import random
import subprocess
import pathlib  
import re
import os
import html
import uuid
import tempfile

import gradio as gr
from PIL import Image
import torch
from transformers import pipeline, BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig
import wandb

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

SCENARIOS_BYPASSING = random.sample(list(pathlib.Path('app_data/bypassing').iterdir()), N_BYPASSING)
SCENARIOS_INTERSECTION = random.sample(list(pathlib.Path('app_data/intersection').iterdir()), N_INTERSECTION)
SCENARIOS_PEDESTRIAN = random.sample(list(pathlib.Path('app_data/pedestrian').iterdir()), N_PEDESTRIAN)
SCENARIOS = SCENARIOS_BYPASSING + SCENARIOS_INTERSECTION + SCENARIOS_PEDESTRIAN

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    "ipab-rad/codellama2-7b-cdsg-tuned-merged",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    use_auth_token=True,
)

tokenizer = AutoTokenizer.from_pretrained("ipab-rad/codellama2-7b-cdsg-tuned-merged", trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training
eos_token_id = 15945 # Corresponds to ""

pipe = pipeline(
        "text-generation", 
        model=model, 
        tokenizer=tokenizer, 
        do_sample=True, 
        top_p=0.9, 
        temperature=0.3, 
        max_length=2048, 
        eos_token_id=eos_token_id)

prompt_prefix = """# Scenic is a domain-specific probabilistic programming language for modeling the environments of cyber-physical systems like robots and autonomous cars. A Scenic program defines a distribution over scenes, configurations of physical objects and agents; sampling from this distribution yields concrete scenes which can be simulated to produce training or testing data. Scenic can also define (probabilistic) policies for dynamic agents, allowing modeling scenarios where agents take actions over time in response to the state of the world.
#
# Here is a list of Scenic scenarios, each with its corresponding description in English included as a docstring:

"""

add_map_path = os.path.join("/home/amiceli/TAS_Project/", "Scenic/tests/formats/opendrive/maps/CARLA/")
def process_response(message, prompt, response):
    generated_code = prompt + "\n" + response

    map_dir = add_map_path + "/" if (add_map_path[-1] != "/") else add_map_path
    carla_map_match = re.search("param(?:\s+)carla_map(?:\s*)=(?:\s*)['\"](\S+)['\"]", generated_code) # CARLA map name
    if carla_map_match:
        carla_map = carla_map_match.group(1)
        map_str = "param map = localPath('%s')" % (map_dir + carla_map + ".xodr")
        generated_code = re.sub('"""\n', ('"""\n\n%s\n' % map_str), generated_code, count=1) # add after the docstring
    return generated_code

#tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf", load_in_4bit=True)
#config = PeftConfig.from_pretrained("ipab-rad/codellama2-7b-cdsg-tuned", load_in_4bit=True)
#model = AutoModelForCausalLM.from_pretrained("codellama/CodeLlama-7b-hf", load_in_4bit=True)
#model = PeftModel.from_pretrained(model, "ipab-rad/codellama2-7b-cdsg-tuned", load_in_4bit=True)


def respond(message, chat_history):
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
        prompt = prompt_prefix + prompt + message + '\n"""'
        print(prompt, flush=True)
        response = pipe(prompt)[0]['generated_text']
        response = response[len(prompt):]
        response = response.split('""')[0].strip()
        response = process_response(message, prompt, response)

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
        path = pathlib.Path('app_data/done.png')
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

#    demo.launch(share=True)
    demo.launch(share=False)
