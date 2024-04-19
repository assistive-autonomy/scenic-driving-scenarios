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

def retrieve_examplars(description: str,
                       d2p: dict[str,str],
                       retriever_model_name:str,
                       top_k:int=3) -> dict[str, str]:
    """retrieve examplars for the given description"""

    Settings.embed_model = HuggingFaceEmbedding(model_name=retriever_model_name)
    Settings.llm = None # not using LLM for embedding.
    
    # form the nodes
    nodes = [TextNode(id_= idx, text=d) for idx, d in enumerate(d2p.keys())]
    index = VectorStoreIndex(nodes)
    retriever = VectorIndexRetriever(index, top_k)
    query_engine = RetrieverQueryEngine(retriever)
    examplars = query_engine.query(description)

    return {d2p[n.text]: n.text for n in examplars}

def random_examplars(d2p: dict[str,str], top_k:int=3) -> dict[str, str]:
    """retrieve random examplars for the given description"""

    examplars = random.sample(d2p.items(), top_k)
    return {d: p for p, d in examplars}

def make_examplars(cfg: ExpConfig, description: str, d2p: dict[str, str]):
    """generate examplars for the given description"""

    if cfg.model.random_examples:
        return random_examplars(d2p, cfg.model.retrieval_top_k)
    else:
        return retrieve_examplars(description,
                                  d2p,
                                  cfg.model.retriever_model_name,
                                  cfg.model.retrieval_top_k)

def make_code_prompt(description:str, examplars:dict[str, str]):
    """generate code prompt from the examplars and description."""
    
    examplars = "".join([f"Description**:\n {d}\nProgram:\n{p}\n" for d,p in examplars.items()])

    return f'''[INST] You are a helpful assistant that translates English descriptions to Scenic programs.
    Scenic is a domain-specific probabilistic programminglanguage for creating distributions over specified scenarios.
    For driving scenarios, each program has the following blocks:
    - MAP AND MODEL: importing town assets and enabling simulator;
    - CONSTANTS: specifying vehicle blueprint and other constants like vehicle speed, brake intensity and safety distance;
    - AGENT'S BEHAVIOR: describing how individual vehicles behave in the scenario;
    - SPATIAL RELATIONS: outlining the type of road the scenario needs to be synthesized in (e.g. having or not having intersections)
    - SCENARIO SPECIFICATION: creating individual vehicles and pedestrians in the specified roads, together with constraints that are required to be true for the full simulation as well as the termination condition.
    Here are examples of description and programs:
    {examplars}
    Now, please translate the following description to a program.Just give the program. No extra information.
    Description:{description}[/INST]'''

def generate_program_from_prompt(cfg: ExpConfig,
                                 model: AutoModelForCausalLM,
                                 tokenizer: AutoTokenizer,
                                 prompt:str):
    """Generate Scenic program from the prompt using the model and tokenizer."""
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
    pred_program = postprocess_program(pred_output, prompt)

    return pred_program, pred_output

def postprocess_program(program, prompt):
    """cleanup after generation"""
    program = program[len(prompt):]
    try:
        program = program[program.index("#"):]
        end_marker = "terminate when (distance to egoSpawnPt) > TERM_DIST"
        end_idx = program.find(end_marker) + len(end_marker) - 1
        program = program[:end_idx]
    except Exception as e:
        program = "Invalid program"
    return program 
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

## Excpetion feeding (in case not compiled program: add it to prompt and re-generate the program)
def add_exception(prediction, exception, description):
    instruction = f"[INST] When trying to run this program, I got the following error:\n{str(exception)}\n\nCan you suggest an updated version?\nPlease first describe the exception, analyze what caused it and how it could be avoided in this specific case, writing your reasoning in comment lines starting with # then generate the updated program. As a reminder, the **English description** is {description} [\INST]"
    return prediction + "\n</s><s>" + instruction

def run_simulation(program):
    tmp_scenario_filename = f"tmp_{uuid.uuid4().hex}.scenic"
    tmp_scenario_filename = os.path.join(tempfile.gettempdir(), tmp_scenario_filename)
    with open(tmp_scenario_filename, "w") as out_fs:
        print(program, file=out_fs)
    process = subprocess.Popen(['bash', 'scripts/simulate.sh', tmp_scenario_filename], stdout=subprocess.PIPE)

    # Get the output and error (if any)
    output, error = process.communicate()
    if os.path.exists(tmp_scenario_filename):
        os.remove(tmp_scenario_filename)


def try_compile_and_run(program):
    """try to compile the program if correct return 1 else 0 with exception message"""
    
    scenario_filename = f"./tmp_{uuid.uuid4().hex}.scenic"
    scenario_output_filename = scenario_filename+".out"
    with open(scenario_filename, "w") as out_fs:
        out_fs.write(program)
    cmd = f"scenic --gather-stats 5 --time 1000 {scenario_filename} > {scenario_output_filename} 2>&1"
    print("Starting simulation: " + cmd, flush=True)
    os.system(cmd)
    with open(scenario_output_filename, "r") as in_scenario_output_fs:
        scenario_output = in_scenario_output_fs.read().strip()
        #print(scenario_output, flush=True)
        scenario_error_match = re.search("Traceback(?:.*)\n", scenario_output)
        if scenario_error_match is None:
            rv = {"compiled": 1, "exception": ""}
        else:
            scenario_error = scenario_output[scenario_error_match.span()[-1]:].strip()
            rv = {"compiled": 0, "exception": scenario_error}
    os.remove(scenario_filename)
    os.remove(scenario_output_filename)
    return rv