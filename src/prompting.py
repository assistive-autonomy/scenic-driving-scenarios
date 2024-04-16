import uuid
import os
import tempfile
import subprocess

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, VectorStoreIndex
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.schema import TextNode

header = f""" You are a helpful assistant that translates English descriptions to Scenic programs.
Scenic is a domain-specific probabilistic programming language for creating distributions over specified scenarios.
For driving scenarios, each program has the following blocks:
- MAP AND MODEL: importing town assets and enabling simulator;
- CONSTANTS: specifying vehicle blueprint and other constants like vehicle speed, brake intensity and safety distance;
- AGENT'S BEHAVIOR: describing how individual vehicles behave in the scenario;
- SPATIAL RELATIONS: outlining the type of road the scenario needs to be synthesized in (e.g. having or not having intersections)
- SCENARIO SPECIFICATION: creating individual vehicles and pedestrians in the specified roads, together with constraints that are required to be true for the full simulation as well as the termination condition.
Here are examples of **English descriptions** and **Scenic programs**:\n"""

instruction = f"""Now, please translate the following **English description** to **Scenic program**.
Just give the program. No extra information.\n**English description**:\n"""

def retrieve_examplars(description: str, d2p: dict[str,str], retriever_model_name:str, top_k:int=3):
    """retrieve examplars for the given description"""

    Settings.embed_model = HuggingFaceEmbedding(model_name=retriever_model_name)
    Settings.llm = None # not using LLM for embedding.
    
    # form the nodes
    nodes = [TextNode(id_= idx, text=d) for idx, d in enumerate(d2p.keys())]
    index = VectorStoreIndex(nodes)
    retriever = VectorIndexRetriever(index, top_k)
    query_engine = RetrieverQueryEngine(retriever)
    return query_engine.query(description)

def make_prompt(description, examplars, d2p):
    """description: task descriptin
    examplars: examplars to use when constructing the prompt
    d2p: dictionary mapping examplars to programs"""
    examplars = "".join([f"**English description**:\n {node.text}\n**Scenic program**:\n{d2p[node.text]}\n" for node in examplars.source_nodes])
    return f'''[INST]{header}{examplars}{instruction}{description}[/INST]'''

def postprocess(program, prompt):
    """cleanup after generation"""
    program = program[len(prompt):]
    program = program[program.index("#"):]
    end_marker = "terminate when (distance to egoSpawnPt) > TERM_DIST"
    end_idx = program.find(end_marker) + len(end_marker) - 1
    program = program[:end_idx]
    return program 

## Excpetion feeding (in case not compiled program: add it to prompt and re-generate the program)
def add_exception(prediction, exception):
    try:
        instruction = f"\n[INST]When trying to compile, I got a {exception.msg} on line \"{exception.text}\". Can you suggest an updated version?\n [\INST]"
    except:
        instruction = f"\n[INST]When trying to compile, I got an exception. Can you suggest an updated version?\n [\INST]"
    return prediction + instruction

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
