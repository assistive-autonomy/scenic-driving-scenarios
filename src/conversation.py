import random
import subprocess
import pathlib  
import re
import os
import html
import uuid
import tempfile

from PIL import Image
# Experiment tracking
import wandb
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from config import ExpConfig
import gradio as gr
# ML libraries
from huggingface_hub import InferenceClient
from datasets import load_dataset
# prompt engineering
from utils import make_prompt, retrieve_examplars, postprocess, add_exception


MAX_INSTRUCTIONS = 5
TOTAL_INSTRUCTIONS = 0
RAW_CHAT_HISTORY = []
USE_WANDB = False


def generate_program_from_prompt(prompt):
    output = client.text_generation(prompt=prompt,
                                    max_length=6000,
                                    max_new_tokens=1200)
    program = postprocess(output, prompt)
    return program, output

def respond(message, chat_history):
    global TOTAL_INSTRUCTIONS
    global MAX_INSTRUCTIONS
    global RAW_CHAT_HISTORY

    if TOTAL_INSTRUCTIONS >= MAX_INSTRUCTIONS:
        response = "You have reached the maximum number of instructions. Please press next scenario to describe another scenario."
    else:
        TOTAL_INSTRUCTIONS += 1
        
        examplars = retrieve_examplars(message, all_d2p, "BAAI/bge-small-en-v1.5", top_k=3)
        prompt = make_prompt(message, examplars, all_d2p)

        program, output = generate_program_from_prompt(prompt)
        
    #print(response, flush=True)
    html_response = f"<pre><code>{html.escape(program)}</code></pre>"
    chat_history.append((message, html_response))

    # run a simulation
    tmp_scenario_filename = f"tmp_{uuid.uuid4().hex}.scenic"
    tmp_scenario_filename = os.path.join(tempfile.gettempdir(), tmp_scenario_filename)
    with open(tmp_scenario_filename, "w") as out_fs:
        print(program, file=out_fs)
    process = subprocess.Popen(['bash', 'scripts/simulate.sh', tmp_scenario_filename], stdout=subprocess.PIPE)

    # Get the output and error (if any)
    output, error = process.communicate()
    if os.path.exists(tmp_scenario_filename):
        os.remove(tmp_scenario_filename)
        
    return "", chat_history

def next_stimuli(scenarios) -> tuple[gr.Image, bool]:
    """Get the next stimuli."""
    if not scenarios:
        path = pathlib.Path('assets/app_data/done.png')
        done = True
    else:
        path = scenarios.pop(random.randint(0, len(scenarios) - 1))
        done = False
    
    return gr.Image(value=Image.open(path), show_label=False, show_download_button=False, show_share_button=False), done, scenarios


def generate_html_chat(chat_history):
    html = ""
    for i, (user, bot) in enumerate(chat_history):
        user = html.escape(user)
        bot = f"<pre><code>{html.escape(bot)}</code></pre>"
        print(bot, flush=True)
        if i == 0:
            html += f"<p><strong>User:</strong> {user}</p>"
            html += f"<p><strong>Bot:</strong><br />{bot}</p>"
        else:
            html += f"<p><strong>User:</strong> {user}</p>"
            html += f"<p><strong>Bot:</strong><br />{bot}</p>"
    return html

def next_scenario(msg, chat_history, img, check):
    global TOTAL_INSTRUCTIONS
    global RAW_CHAT_HISTORY
    
    # logging
    if USE_WANDB:
        wandb.log({"chat_history": wandb.Html(generate_html_chat(chat_history), inject=False),
                "stimuli": wandb.Image(img),
                "number_of_instructions": TOTAL_INSTRUCTIONS,
                "satisfied": int(check),
                })
    # reset
    TOTAL_INSTRUCTIONS = 0
    msg = 0
    chat_history = []
    RAW_CHAT_HISTORY = []
    img, done = next_stimuli()
    check = False

    return msg, chat_history, img, check

cs = ConfigStore.instance()
cs.store(name="base_config", node=ExpConfig)

@hydra.main(config_path="../config", config_name="rag", version_base=None)
def main(cfg: ExpConfig):

    if cfg.wandb.use_wandb:
        config = OmegaConf.to_container(cfg, resolve=False)
        wandb.init(project=cfg.wandb.project,
                    entity=cfg.wandb.entity,
                    name=f"Conversation Experiments",
                    config=config)
        
    random.seed(cfg.seed)

    dataset = load_dataset(cfg.data.dataset_name,
                           trust_remote_code=True)["train"]
    d2p = {d: p for d, p in zip(dataset["description"], dataset["program"])}

    model = InferenceClient(model=cfg.model.client)

    base_path = pathlib.Path(cfg.conv.scenarios.path)
    bypassing = random.sample(list(base_path/"bypassing".iterdir()),
                              cfg.conv.scenarios.bypassing)
    intersection = random.sample(list(base_path/"intersection".iterdir()),
                                 cfg.conv.scenarios.intersection)
    pedestrian = random.sample(list(base_path/"pedestrian".iterdir()),
                               cfg.conv.scenarios.pedestrian)
    scenarios = bypassing + intersection + pedestrian

    with gr.Blocks() as demo:
        gr.Markdown("# Conversartional Driving Scenario Generation 🚗 💬")
        with gr.Row():
            img, done, scenarios = next_stimuli(scenarios)
            chatbot = gr.Chatbot(avatar_images=(cfg.conv.assets.user,
                                                cfg.conv.assets.bot),
                                                show_label=False)
        
        msg = gr.Textbox(label="Enter your instructions here")
        msg.submit(respond, [msg, chatbot], [msg, chatbot])
        with gr.Row():
            check = gr.Checkbox(label="Satsified with the scenarios produced")
            next_btn = gr.Button(value="Next Scenario")
            next_btn.click(next_scenario,
                           inputs=[msg, chatbot, img, check],
                           outputs=[msg, chatbot, img, check])

    demo.launch(share=cfg.conv.share)

if __name__ == '__main__':
    main()