import gradio as gr
import random

from PIL import Image

def respond(message, chat_history):
        bot_message = random.choice(["How are you?", "I love you", "I'm very hungry"])
        chat_history.append((message, bot_message))
        return "", chat_history

with gr.Blocks() as demo:
    gr.Markdown("This is a simple demo of Gradio. Enter your name and click 'Run' to see the output.")
    with gr.Row():
        # legeng_img = gr.Image(Image.open("app_data/legend.png"))
        scenario_img = gr.Image(Image.open("app_data/bypassing/0.png"))
        chatbot = gr.Chatbot(avatar_images=("app_data/user.png", "app_data/bot.png"))
    msg = gr.Textbox(label="Enter your instructions here")
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    next_btn = gr.ClearButton([msg, chatbot], value="Next Scenario")

demo.launch()