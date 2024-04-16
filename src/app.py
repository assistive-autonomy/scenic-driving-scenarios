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
from omegaconf import DictConfig, OmegaConf
import gradio as gr
# ML libraries
from huggingface_hub import InferenceClient
from datasets import load_dataset
# prompt engineering
from prompting import make_prompt, retrieve_examplars, postprocess, add_exception

SEED = 42
N_BYPASSING = 2
N_INTERSECTION = 2
N_PEDESTRIAN = 2
MAX_INSTRUCTIONS = 5
TOTAL_INSTRUCTIONS = 0
RAW_CHAT_HISTORY = []
USE_WANDB = False

if SEED:
    random.seed(SEED)

SCENARIOS_BYPASSING = random.sample(list(pathlib.Path('assets/app_data/bypassing').iterdir()), N_BYPASSING)
SCENARIOS_INTERSECTION = random.sample(list(pathlib.Path('assets/app_data/intersection').iterdir()), N_INTERSECTION)
SCENARIOS_PEDESTRIAN = random.sample(list(pathlib.Path('assets/app_data/pedestrian').iterdir()), N_PEDESTRIAN)
SCENARIOS = SCENARIOS_BYPASSING + SCENARIOS_INTERSECTION + SCENARIOS_PEDESTRIAN

client = InferenceClient(model="http://goya:8080")

def generate_program_from_prompt(cfg, prompt):
    output = client.text_generation(prompt=prompt)
    program = postprocess(output, prompt)
    return program, output


def respond(message, chat_history):
    pass
    global TOTAL_INSTRUCTIONS
    global MAX_INSTRUCTIONS
    global RAW_CHAT_HISTORY

    if TOTAL_INSTRUCTIONS >= MAX_INSTRUCTIONS:
        response = "You have reached the maximum number of instructions. Please press next scenario to describe another scenario."
    else:
        TOTAL_INSTRUCTIONS += 1
        prompt = '""" Scenario description\n' + " ".join([m for m, h in RAW_CHAT_HISTORY])
        if (len(RAW_CHAT_HISTORY) > 0):
            prompt += " "
        message = message.strip()
        if not message.endswith("."):
            message += "."
        print(prompt, flush=True)
        program, output = generate_program_from_prompt(prompt)
        
    #print(response, flush=True)
    html_response = f"<pre><code>{html.escape(response)}</code></pre>"
    chat_history.append((message, html_response))
    RAW_CHAT_HISTORY.append((message, response))

    # run a simulation
    tmp_scenario_filename = f"tmp_{uuid.uuid4().hex}.scenic"
    tmp_scenario_filename = os.path.join(tempfile.gettempdir(), tmp_scenario_filename)
    with open(tmp_scenario_filename, "w") as out_fs:
        print(response, file=out_fs)
    process = subprocess.Popen(['bash', 'scripts/simulate.sh', tmp_scenario_filename], stdout=subprocess.PIPE)

    # Get the output and error (if any)
    output, error = process.communicate()
    if os.path.exists(tmp_scenario_filename):
        os.remove(tmp_scenario_filename)
        
#        inputs = tokenizer(message, return_tensors="pt").to("cuda")
#        outputs = model.generate(inputs=inputs['input_ids'],
#                                 max_new_tokens=1000,
#                                 do_sample=True,
#                                 temperature=0.1)
#
#        output = outputs[0].cpu()
#        responce = tokenizer.decode(output, skip_special_tokens=True)
#
#    chat_history.append((message, responce))
#
#    # run a simulation
#    process = subprocess.Popen(['bash', 'scripts/simulate.sh', '19.scenic'], stdout=subprocess.PIPE)
#
#    # Get the output and error (if any)
#    output, error = process.communicate()

    return "", chat_history

def next_stimuli() -> tuple[gr.Image, bool]:
    if not SCENARIOS:
        path = pathlib.Path('assets/app_data/done.png')
        done = True
    else:
        path = SCENARIOS.pop(random.randint(0, len(SCENARIOS) - 1))
        done = False
    
    return gr.Image(value=Image.open(path), show_label=False, show_download_button=False, show_share_button=False), done


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

@hydra.main(config_path="../config", config_name="rag", version_base=None)
def main(cfg: DictConfig):
    pass

if USE_WANDB:
    wandb.init(project="CDSG-experiments",entity="ipab-rad")


with gr.Blocks() as demo:
    gr.Markdown("# Conversartional Driving Scenario Generation 🚗 💬")
    with gr.Row():
        img, done = next_stimuli()
        chatbot = gr.Chatbot(avatar_images=("assets/app_data/user.png", "assets/app_data/bot.png"), show_label=False)
    
    msg = gr.Textbox(label="Enter your instructions here")
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    with gr.Row():
        check = gr.Checkbox(label="Satsified with the scenarios produced")
        next_btn = gr.Button(value="Next Scenario")
        next_btn.click(next_scenario, inputs=[msg, chatbot, img, check], outputs=[msg, chatbot, img, check])

#    demo.launch(share=True)
    demo.launch(share=False)

if __name__ == '__main__':
    main()