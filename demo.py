import gradio as gr
from bot import chat

print("\n---Start---\n")

gr.ChatInterface(chat).launch()