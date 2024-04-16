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
from openai import OpenAI
#import backoff

#import wandb

SEED = 42
N_BYPASSING = 2
N_INTERSECTION = 2
N_PEDESTRIAN = 2
MAX_INSTRUCTIONS = 5
MAX_LLM_QUERIES_PER_TURN = 5
TOTAL_INSTRUCTIONS = 0
RAW_CHAT_HISTORY = []

random.seed(SEED)

SCENARIOS_BYPASSING = random.sample(list(pathlib.Path('assets/app_data/bypassing').iterdir()), N_BYPASSING)
SCENARIOS_INTERSECTION = random.sample(list(pathlib.Path('assets/app_data/intersection').iterdir()), N_INTERSECTION)
SCENARIOS_PEDESTRIAN = random.sample(list(pathlib.Path('assets/app_data/pedestrian').iterdir()), N_PEDESTRIAN)
SCENARIOS = SCENARIOS_BYPASSING + SCENARIOS_INTERSECTION + SCENARIOS_PEDESTRIAN


with open("../oai_key", "r") as api_key_fs:
    os.environ["OPENAI_API_KEY"] = api_key_fs.read().strip()
gpt_client = OpenAI(
  api_key=os.environ['OPENAI_API_KEY'],  # this is also the default, it can be omitted
)

#MODEL_NAME = "ft:gpt-3.5-turbo-1106:uedin:scenic-run001:8yq6JXLS"
MODEL_NAME = "ft:gpt-3.5-turbo-1106:uedin:scenic-run002:8ysfl1rR"
#MODEL_NAME= "ft:gpt-3.5-turbo-1106:uedin:scenic-run003:8ytcZljV" 
MODEL_TEMP = 0.1

#@backoff.on_exception(backoff.expo, openai.RateLimitError)
def get_lm_response(prompt):
    response = gpt_client.chat.completions.create(
        model = MODEL_NAME,
        temperature = MODEL_TEMP,
        messages = prompt)
    return response.choices[0].message.content

system_message_str="""You are a helpful agent that generates specifications for car driving scenarios in the Scenic language
Scenic is a domain-specific probabilistic programming language for modeling the environments of cyber-physical systems like robots and autonomous cars. A Scenic program defines a distribution over scenes, configurations of physical objects and agents; sampling from this distribution yields concrete scenes which can be simulated to produce training or testing data. Scenic can also define (probabilistic) policies for dynamic agents, allowing modeling scenarios where agents take actions over time in response to the state of the world.

Your task is to generate Scenic scenarios, each according to its corresponding description in English included as a docstring. Write each scenario in a separate code box."""
system_role = "system"
system_message = {"role": system_role, "content": system_message_str}

add_map_path = os.path.join("/home/amiceli/TAS_Project/", "Scenic/tests/formats/opendrive/maps/CARLA/")
def process_response(prompt_messages, response_str):
    generated_code_raw = response_str
    generated_code_match = re.search(r'(?s)```(?:scenic)?\s?(.*)```', generated_code_raw)
    if generated_code_match:
        generated_code = generated_code_match.groups()[0].strip()
    else:
        generated_code = "### ERROR: No code generated"

    docstring = prompt_messages[1]["content"]
    generated_code = docstring + "\n" + generated_code

    map_dir = add_map_path + "/" if (add_map_path[-1] != "/") else add_map_path
    carla_map_match = re.search("param(?:\s+)carla_map(?:\s*)=(?:\s*)['\"](\S+)['\"]", generated_code) # CARLA map name
    if carla_map_match:
        carla_map = carla_map_match.group(1)
        map_str = "param map = localPath('%s')" % (map_dir + carla_map + ".xodr")
        generated_code = re.sub('"""\n', ('"""\n\n%s\n' % map_str), generated_code, count=1) # add after the docstring
    return generated_code

def first_user_message(user_str):
    return '""" Scenario description\n' + user_str + '\n"""'

def respond(message, chat_history):
    global TOTAL_INSTRUCTIONS
    global MAX_INSTRUCTIONS
    global RAW_CHAT_HISTORY

    if TOTAL_INSTRUCTIONS >= MAX_INSTRUCTIONS:
        response = "You have reached the maximum number of instructions. Please press next scenario to describe another scenario."
    else:
        TOTAL_INSTRUCTIONS += 1
        message = message.strip()
        if not message.endswith("."):
            message += "."
        
        prompt_messages = [system_message]
        for i, e in enumerate(RAW_CHAT_HISTORY):
            user_str, bot_str = e
            if i == 0:
                user_str = first_user_message(user_str)
            prompt_messages.append({"role": "user", "content": user_str})
            prompt_messages.append({"role": "assistant", "content": bot_str})
        user_str = first_user_message(message) if (len(RAW_CHAT_HISTORY) == 0) else message
        prompt_messages.append({"role": "user", "content": user_str})

        print(f"Prompt dict: {prompt_messages}", flush=True)
        response_raw_str = get_lm_response(prompt_messages)
        print(f"GPT Response raw: {response_raw_str}", flush=True)
        response = process_response(prompt_messages, response_raw_str)

    #print(response, flush=True)
    html_response = f"<pre><code>{html.escape(response_raw_str)}</code></pre>"
    chat_history.append((message, html_response))
    RAW_CHAT_HISTORY.append((message, response_raw_str))

    # run a simulation
    tmp_scenario_filename = f"tmp_{uuid.uuid4().hex}.scenic"
    tmp_scenario_filename = os.path.join(tempfile.gettempdir(), tmp_scenario_filename)
    with open(tmp_scenario_filename, "w") as out_fs:
        print(response, file=out_fs)
        print(f"Processed scenario file:\n{response}", flush=True)
    process = subprocess.Popen(['bash', 'scripts/simulate.sh', tmp_scenario_filename], 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE)

    # Get the output and error (if any)
    output, error = process.communicate()
#    print(f"output: {output}\nerror: {error}", flush=True)
    if os.path.exists(tmp_scenario_filename):
        os.remove(tmp_scenario_filename)
    if os.path.exists(tmp_scenario_filename+".out"):
        with open(tmp_scenario_filename+".out") as in_fs:
            output = in_fs.read()
            print(f"output: {output}")
        #os.remove(tmp_scenario_filename+".out")

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
#    wandb.log({"chat_history": wandb.Html(generate_html_chat(chat_history), inject=False),
#               "stimuli": wandb.Image(img),
#               "number_of_instructions": TOTAL_INSTRUCTIONS,
#               "satisfied": int(check),
#               })

    # reset
    TOTAL_INSTRUCTIONS = 0
    msg = 0
    chat_history = []
    RAW_CHAT_HISTORY = []
    img, done = next_stimuli()
    check = False

    return msg, chat_history, img, check



#wandb.init(project="CDSG-experiments",
#           entity="ipab-rad",
#           )


with gr.Blocks() as demo:
    gr.Markdown("# Conversartional Driving Scenario Generation 🚗 💬")
    gr.Markdown("This is a chat platform that allows you to generate driving scenarios by provinding scenario descriptions in English. \
                You are asked to describe the scenario depicted in an image and the dialogue agent (chatbot) will generate the scenario \
                for you using CARLA simulator. If something is wrong with the simulations provided you are encouraged to provide \
                follow up instructions (up to 5 times). When you are happy with the simulations produced press next scenario to describe another scenario.\
                You will be asked to describe 6 scenarios in total.")
    with gr.Row():
        img, done = next_stimuli()
        chatbot = gr.Chatbot(avatar_images=("assets/app_data/user.png", "assets/app_data/bot.png"), show_label=False)
    
    msg = gr.Textbox(label="Enter your instructions here")
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    with gr.Row():
        check = gr.Checkbox(label="satsified with the scenarios produced")
        next_btn = gr.Button(value="Next Scenario")
        next_btn.click(next_scenario, inputs=[msg, chatbot, img, check], outputs=[msg, chatbot, img, check])

    demo.launch(share=True)
