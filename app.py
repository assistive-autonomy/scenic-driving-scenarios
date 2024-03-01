import random
import subprocess
import pathlib  

import gradio as gr
from PIL import Image
from transformers import pipeline

from peft import PeftModel, PeftConfig
from transformers import AutoModelForCausalLM, AutoTokenizer



import wandb

SEED = 42
N_BYPASSING = 2
N_INTERSECTION = 2
N_PEDESTRIAN = 2
MAX_INSTRUCTIONS = 5
TOTAL_INSTRUCTIONS = 0
USE_WANDB = False

if SEED:
    random.seed(SEED)

SCENARIOS_BYPASSING = random.sample(list(pathlib.Path('app_data/bypassing').iterdir()), N_BYPASSING)
SCENARIOS_INTERSECTION = random.sample(list(pathlib.Path('app_data/intersection').iterdir()), N_INTERSECTION)
SCENARIOS_PEDESTRIAN = random.sample(list(pathlib.Path('app_data/pedestrian').iterdir()), N_PEDESTRIAN)
SCENARIOS = SCENARIOS_BYPASSING + SCENARIOS_INTERSECTION + SCENARIOS_PEDESTRIAN

tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf", load_in_4bit=True)
config = PeftConfig.from_pretrained("ipab-rad/codellama2-7b-cdsg-tuned", load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-7b-hf", load_in_4bit=True)
model = PeftModel.from_pretrained(model, "ipab-rad/codellama2-7b-cdsg-tuned", load_in_4bit=True)


# pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

def respond(message, chat_history):
    global TOTAL_INSTRUCTIONS
    global MAX_INSTRUCTIONS

    if TOTAL_INSTRUCTIONS >= MAX_INSTRUCTIONS:
        responce = "You have reached the maximum number of instructions. Please press next scenario to describe another scenario."
    else:
        TOTAL_INSTRUCTIONS += 1
        
        inputs = tokenizer(message, return_tensors="pt").to("cuda")
        outputs = model.generate(inputs=inputs['input_ids'],
                                 max_new_tokens=1000,
                                 do_sample=True,
                                 temperature=0.1)

        output = outputs[0].cpu()
        responce = tokenizer.decode(output, skip_special_tokens=True)

    chat_history.append((message, responce))

    # run a simulation
    process = subprocess.Popen(['bash', 'scripts/simulate.sh', '19.scenic'], stdout=subprocess.PIPE)

    # Get the output and error (if any)
    output, error = process.communicate()

    return "", chat_history

def next_stimuli() -> tuple[gr.Image, bool]:
    if not SCENARIOS:
        path = pathlib.Path('app_data/done.png')
        done = True
    else:
        path = SCENARIOS.pop(random.randint(0, len(SCENARIOS) - 1))
        done = False
    
    return gr.Image(value=Image.open(path), show_label=False, show_download_button=False, show_share_button=False), done


def generate_html_chat(chat_history):
    html = ""
    for i, (user, bot) in enumerate(chat_history):
        if i == 0:
            html += f"<p><strong>User:</strong> {user}</p>"
            html += f"<p><strong>Bot:</strong> {bot}</p>"
        else:
            html += f"<p><strong>User:</strong> {user}</p>"
            html += f"<p><strong>Bot:</strong> {bot}</p>"
    return html

def next_scenario(msg, chat_history, img, check):
    global TOTAL_INSTRUCTIONS
    
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
    img, done = next_stimuli()
    check = False

    return msg, chat_history, img, check


if USE_WANDB:

    wandb.init(project="CDSG-experiments",
            entity="ipab-rad",
            )


with gr.Blocks() as demo:
    gr.Markdown("# Conversartional Driving Scenario Generation 🚗 💬")
    # gr.Markdown("This is a chat platform that allows you to generate driving scenarios by provinding scenario descriptions in English. \
    #             You are asked to describe the scenario depicted in an image and the dialogue agent (chatbot) will generate the scenario \
    #             for you using CARLA simulator. If something is wrong with the simulations provided you are encouraged to provide \
    #             follow up instructions (up to 5 times). When you are happy with the simulations produced press next scenario to describe another scenario.\
    #             You will be asked to describe 6 scenarios in total.")
    with gr.Row():
        img, done = next_stimuli()
        chatbot = gr.Chatbot(avatar_images=("app_data/user.png", "app_data/bot.png"), show_label=False)
    
    msg = gr.Textbox(label="Enter your instructions here")
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    with gr.Row():
        check = gr.Checkbox(label="satsified with the scenarios produced")
        next_btn = gr.Button(value="Next Scenario")
        next_btn.click(next_scenario, inputs=[msg, chatbot, img, check], outputs=[msg, chatbot, img, check])

    demo.launch(share=False)