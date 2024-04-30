import random
import pathlib  
from dataclasses import dataclass
# Experiment tracking
import wandb
import hydra
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf
from config import ExpConfig
# Dialogue Interfaca
from PIL import Image
import gradio as gr
# ML libraries
from datasets import load_dataset
from utils import (make_code_prompt,
                   generate_program,
                   run_simulation,
                   make_summ_prompt,
                   generate_description)

@dataclass
class AppStatus:
    """Container to track information during the experiment."""
    scenarios: list[pathlib.Path]
    num: int = 0
    max: int = 5
    description: str = ""
    cfg: ExpConfig = ExpConfig()

def get_scenarios(cfg: ExpConfig) -> list[pathlib.Path]:
    """Get scenarios from the given path."""
    base_path = pathlib.Path(cfg.conv.scenarios.path)
    bypassing = random.sample(list((base_path/"bypassing").iterdir()),
                              cfg.conv.scenarios.bypassing)
    intersection = random.sample(list((base_path/"intersection").iterdir()),
                                 cfg.conv.scenarios.intersection)
    pedestrian = random.sample(list((base_path/"pedestrian").iterdir()),
                               cfg.conv.scenarios.pedestrian)
    scenarios = bypassing + intersection + pedestrian
    return scenarios

def respond(message:str,
            chat_history:list[str],
            ):
    
    global d2p
    global status

    if status.num >= status.max:
        response = """You have reached the maximum number of instructions.
        Please press next scenario to describe another scenario."""
    else:
        status.num += 1

        # if not the first message, generate the descripton
        if status.description != "":
            print("==== changing the description ===")
            summ_prompt = make_summ_prompt(status.description, message)
            status.description = generate_description(status.cfg, summ_prompt)
        else:
            status.description = message

        print(f"====description: {status.description}")
        
        code_prompt = make_code_prompt(status.cfg, status.description, d2p)
        pred_program, compiled = generate_program(status.cfg, code_prompt, status.description)

        if not compiled["compiled"]:
            response = "I did not manage to create a simulation. Can you rephrase it!"
            ## delete the message 
            status.description = ""
        else:
            response = "Understood. Let's generate 3 simulation instances." 

            run_simulation(pred_program)

    chat_history.append((message, response))
        
    return "", chat_history

def next_stimuli(status: AppStatus) -> tuple[gr.Image, AppStatus]:
    """Get the next stimuli."""
    if not status.scenarios:
        path = pathlib.Path('assets/app_data/done.png')
    else:
        path = status.scenarios.pop(random.randint(0, len(status.scenarios) - 1))
    
    return gr.Image(value=Image.open(path),
                    show_label=False, 
                    show_download_button=False,
                    show_share_button=False), status

def next_scenario(msg, chat_history, img, check):

    global status

    def html_chat(chat_history):
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
    
    # logging
    if status.cfg.wandb.use_wandb:
        wandb.log({"dialogue": wandb.Html(html_chat(chat_history),inject=False),
                "stimuli": wandb.Image(img),
                "number_of_instructions": status["num"],
                "satisfied": int(check),
                })
    # reset
    status.num = 0
    msg = 0
    chat_history = []
    img, status = next_stimuli(status)
    check = False

    return msg, chat_history, img, check, status

cs = ConfigStore.instance()
cs.store(name="base_config", node=ExpConfig)

@hydra.main(config_path="../config", config_name="default", version_base=None)
def main(cfg: ExpConfig):

    print(OmegaConf.to_yaml(cfg))

    if cfg.wandb.use_wandb:
        config = OmegaConf.to_container(cfg, resolve=False)
        feed = "-error-feeding" if cfg.model.exceptions.do_feeding else ""
        wandb.init(project=cfg.wandb.project,
                   entity=cfg.wandb.entity,
                   name=f"CONV-{cfg.model.model_name}{feed}",
                   config=config)
        
    random.seed(cfg.seed)

    dataset = load_dataset(cfg.data.dataset_name)["train"]
    global d2p
    d2p = {d: p for d, p in zip(dataset["description"], dataset["program"])}
    
    # marker for state status
    global status
    status = AppStatus(scenarios=get_scenarios(cfg),
                       num=0,
                       max=cfg.conv.max_instructions,
                       description="",
                       cfg=cfg)

    with gr.Blocks() as demo:
        gr.Markdown("# Conversartional Driving Scenario Generation 🚗 💬")
        with gr.Row():
            img, status = next_stimuli(status)
            chatbot = gr.Chatbot(avatar_images=(cfg.conv.assets.user,
                                                cfg.conv.assets.bot),
                                                show_label=False)
        
        msg = gr.Textbox(label="Enter your instructions here")
        msg.submit(respond,
                   inputs=[msg, chatbot],
                   outputs=[msg, chatbot])

        with gr.Row():
            check = gr.Checkbox(label="Satsified with the scenarios produced")
            next_btn = gr.Button(value="Next Scenario")
            next_btn.click(next_scenario,
                           inputs=[msg, chatbot, img, check],
                           outputs=[msg, chatbot, img, check])

    demo.launch(share=cfg.conv.share)

if __name__ == '__main__':
    main()
