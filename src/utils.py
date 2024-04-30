import random
import uuid
import os
import re
import tempfile
import subprocess

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import TextNode

from config import ExpConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import InferenceClient


def get_examplars(cfg: ExpConfig, description: str, d2p: dict[str, str]):
    """Get examplars for the given description."""

    if cfg.model.random_examples:
        """Random example retrieval."""
        examplars = random.sample(d2p.items(), cfg.model.retrieval_top_k)
        return {d: p for p, d in examplars}
    else:
        """description-similarity based example retrieval."""
        Settings.embed_model = HuggingFaceEmbedding(model_name=cfg.model.retriever_model_name)
        Settings.llm = None # not using LLM for embedding.
        
        nodes = [TextNode(id_= idx, text=d) for idx, d in enumerate(d2p.keys())]
        index = VectorStoreIndex(nodes)
        retriever = VectorIndexRetriever(index, cfg.model.retrieval_top_k)
        query_engine = RetrieverQueryEngine(retriever)
        examplars = query_engine.query(description)
        return {n.text: d2p[n.text] for n in examplars.source_nodes}

def run_simulation(program: str):
    """Run simulation for the user to observe"""
    # tmp_scenario_filename = f"tmp_{uuid.uuid4().hex}.scenic"
    # tmp_scenario_filename = os.path.join(tempfile.gettempdir(), tmp_scenario_filename)
    # with open(tmp_scenario_filename, "w") as out_fs:
    #     print(program, file=out_fs)

    tmp_scenario_filename = f"./tmp_{uuid.uuid4().hex}.scenic"
    with open(tmp_scenario_filename, "w") as out_fs:
        out_fs.write(program)
    process = subprocess.Popen(['bash', '../scripts/simulate.sh', tmp_scenario_filename], stdout=subprocess.PIPE)

    # Get the output and error (if any)
    output, error = process.communicate()
    if os.path.exists(tmp_scenario_filename):
        os.remove(tmp_scenario_filename)

def try_compile_and_run(program: str):
    """try to compile the program if correct return 1 else 0 with exception message"""
    
    scenario_filename = f"./tmp_{uuid.uuid4().hex}.scenic"
    scenario_output_filename = scenario_filename+".out"
    with open(scenario_filename, "w") as out_fs:
        out_fs.write(program)
    cmd = f"scenic --gather-stats 5 --time 1000 {scenario_filename} > {scenario_output_filename} 2>&1"
    os.system(cmd)
    with open(scenario_output_filename, "r") as in_scenario_output_fs:
        scenario_output = in_scenario_output_fs.read().strip()
        scenario_error_match = re.search("Traceback(?:.*)\n", scenario_output)
        if scenario_error_match is None:
            rv = {"compiled": 1, "exception": ""}
        else:
            scenario_error = scenario_output[scenario_error_match.span()[-1]:].strip()
            rv = {"compiled": 0, "exception": scenario_error}
    os.remove(scenario_filename)
    os.remove(scenario_output_filename)
    return rv

def code_block(program):
    return f"```scenic\n{program.strip()}\n```"

def make_code_prompt(cfg: ExpConfig, description: str, d2p: dict[str, str]):
    """generate code prompt from the examplars and description."""
    
    examplars = get_examplars(cfg, description, d2p)
    examplars = "".join([f"""**English description**:"{d}"\n**Scenic program**:\n{code_block(p)}\n""" for d,p in examplars.items()])

    return f"""[INST] You are a helpful assistant that translates English descriptions to Scenic programs.
Scenic is a domain-specific probabilistic programming language for creating distributions over specified scenarios.
For driving scenarios, each program has the following blocks:
- MAP AND MODEL: importing town assets and enabling simulator;
- CONSTANTS: specifying vehicle blueprint and other constants like vehicle speed, brake intensity and safety distance;
- AGENT'S BEHAVIOR: describing how individual vehicles behave in the scenario;
- SPATIAL RELATIONS: outlining the type of road the scenario needs to be synthesized in (e.g. having or not having intersections)
- SCENARIO SPECIFICATION: creating individual vehicles and pedestrians in the specified roads, together with constraints that are required to be true for the full simulation as well as the termination condition.
Here are examples of **English descriptions** and **Scenic programs**:
{examplars}
Now, please translate the following description to a program. Add no extra information in your response.
**English description**:"{description}" [/INST]"""

def generate_program(cfg: ExpConfig,
                    prompt:str,
                    description: str,
                    model: AutoModelForCausalLM = None,
                    tokenizer: AutoTokenizer = None) -> tuple[str, bool]:
    
    def postprocess(program, prompt=None):
        """cleanup after generation"""
        if prompt is not None:
            program = program[len(prompt):]
        maybe_program = program
        try:
            generated_code_match = re.search(r'(?s)```(?:scenic)?\s?(.*)```', maybe_program)
            if generated_code_match:
                program = generated_code_match.groups()[0].strip()
            else:
                program = "### ERROR: No code generated"
            #program = program[program.index("#"):]
            #print("=== 0 ===\n"+program, flush=True)
            #end_marker = "terminate when (distance to egoSpawnPt) > TERM_DIST"
            #end_idx = program.find(end_marker) + len(end_marker) - 1
            #program = program[:end_idx+1]
            #print(f"=== 1 ===\n{end_idx}\n{program}", flush=True)
        except Exception as e:
            """if program cannot be extracted 
            by this pattern matching, use error-feeding for correction"""
            #print("==== maybe_program ====\n"+maybe_program, flush=True)
            return maybe_program
        return program
    
    def error_feeding(prediction, exception, description):
        ## Excpetion feeding (in case not compiled program: add it to prompt and re-generate the program)
        instruction = f"""[INST] When trying to run this program, I got the following error:\n{str(exception)}
Can you suggest an updated version?
Please first describe the exception, analyze what caused it and how it could be avoided in this specific case, writing your reasoning in comment lines starting with # then generate the updated program.
Make sure you generate valid Scenic code, according to the examples, not Python code.
Make sure you generate a full Scenic program, not just a snippet.
Enclose the program in a code block:
```scenic
# PROGRAM
```
**English description**:"{description}e [\INST]"""
        return prediction + "\n</s><s>" + instruction

    """Generate Scenic program"""
    num_trials = 0
    if model is None and tokenizer is None:
        client = InferenceClient(model=cfg.model.client)
    else:
        client = None
    while True:
        num_trials += 1

        #print("prompt used")
        #print(prompt)

        if client is None:
            inputs = tokenizer(prompt, return_tensors="pt")
            outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"),
                                    attention_mask=inputs["attention_mask"].to("cuda"),
                                    pad_token_id=tokenizer.eos_token_id,
                                    temperature=cfg.model.temperature,
                                    do_sample=cfg.model.do_sample,
                                    top_k=cfg.model.decoding_top_k,
                                    num_beams=cfg.model.num_beams,
                                    max_new_tokens=cfg.model.max_new_tokens)
            pred_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

            pred_program = postprocess(pred_output, prompt)

        else:
            pred_output = client.text_generation(prompt=prompt,
                                                temperature=cfg.model.temperature,
                                                do_sample=cfg.model.do_sample,
                                                top_k=cfg.model.decoding_top_k,
                                                best_of = cfg.model.num_beams,
                                                max_new_tokens=cfg.model.max_new_tokens)

   
            pred_program = postprocess(pred_output)
        #print("=====raw llm output =======\n"+pred_output, flush=True)
        #print("=====program pred =======")
        #print(pred_program)

        compiled = try_compile_and_run(pred_program)

        print(compiled)

        if not compiled["compiled"] and \
            cfg.model.exceptions.do_feeding and \
            num_trials < cfg.model.exceptions.max_trials:
            error_msg = error_feeding(pred_output, compiled["exception"], description)
            prompt = prompt + error_msg
        else:
            return pred_program, compiled

def make_summ_prompt(description:str, feedback:str) -> str:
    """generate prompt for summarizing description and user's feedback"""

    return f"""[INST] You are a helpful assistant that creates updated driving scenario descriptions.
    Given a driving scenario description and corresponding feedback regarding the simulation, your task is to generate an updated description that incorporates the feedback.
    Please ensure that the updated description accurately reflects the intended action based on the feedback received and does not introduce additional information or lose information, like the number of vehicles or pedestrians in the situation. 
    The description will outline a specific scenario or context, while the feedback will provide information about how the described scenario could have been improved or modified.
    Be sure to maintain the original meaning and intent of both the initial description and the feedback.
    Be brief and only return the updated driving scenario description.
    Description: {description}
    Feedback: {feedback}[/INST]"""

def generate_description(cfg: ExpConfig,
                        prompt: str,
                        model: AutoModelForCausalLM = None,
                        tokenizer: AutoTokenizer = None, 
                        ) -> str:
    """Generate updated description from the prompt"""
    if model is None and tokenizer is None:
        client = InferenceClient(model=cfg.model.client)
        pred_description = client.text_generation(prompt=prompt,
                                                temperature=cfg.model.temperature,
                                                do_sample=cfg.model.do_sample,
                                                top_k=cfg.model.decoding_top_k,
                                                best_of = cfg.model.num_beams,
                                                max_new_tokens=cfg.model.max_new_tokens)

    else:
        inputs = tokenizer(prompt, return_tensors="pt")
        outputs = model.generate(input_ids=inputs["input_ids"].to("cuda"),
                                attention_mask=inputs["attention_mask"].to("cuda"),
                                pad_token_id=tokenizer.eos_token_id,
                                temperature=cfg.model.temperature,
                                do_sample=cfg.model.do_sample,
                                top_k=cfg.model.decoding_top_k,
                                num_beams=cfg.model.num_beams,
                                max_new_tokens=cfg.model.max_new_tokens)
        pred_description = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return pred_description



