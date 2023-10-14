import gradio as gr
from gradio.components import Slider, Dropdown
import torch
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
import jsonlines
import gc
import math
from peft import PeftModel
from datetime import datetime
import os
# base model 
MODEL_LIST = [
      'mistralai/Mistral-7B-v0.1'
]
# peft weight가 들어있는 폴더 만약 오리지날로 하고 싶다면 그냥 빈칸
lora_path='./mistral_7b'

# 대화 데이터 저장 path
os.makedirs('./data',exist_ok=True)
save_file_dir=f'./data/talk{datetime.now().strftime("%m_%d")}.jsonl'

CURRENT_MODEL = model = pipe = tokenizer = None
cur_model_name=None
def load_model(model_name):
    global CURRENT_MODEL, model, pipe, tokenizer, cur_model_name, lora_path

    if (model_name != CURRENT_MODEL):
        cur_model_name=model_name
        print(f"Start loading {model_name}")
        CURRENT_MODEL = model_name
        model = pipe = tokenizer = None
        gc.collect()
        torch.cuda.empty_cache()

        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            load_in_8bit=True,
        )

        if lora_path:
            print('peft 불러오기')
            model = PeftModel.from_pretrained(
            model,
            lora_path,
            torch_dtype=torch.float16,
            )
            print('peft 성공!')
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        pipe = pipeline(
            "text-generation",
            model=model,
            tokenizer=model_name,
            device_map="auto"
        )
        print(f"{model_name} loading done!")

        return clear_history()


def answer(state, state_chatbot, text, temperature, top_p, top_k, rep_penalty):
    global cur_model_name
    state =[]
    messages = state

    system_prompts=""
    conversation_history = "".join(
        [f"[INST]{msg['question']}[\INST]{msg['answer']}" for msg in messages]
    )
    conversation_history=system_prompts+conversation_history
    print(conversation_history)

    load_model(CURRENT_MODEL)
    ans = pipe(
        conversation_history +"[INST]"+text+ "[\INST]",
        do_sample=True,
        max_new_tokens=512,
        temperature=temperature,
        #top_p=top_p,
        #top_k=top_k,
        num_beams = 1,
        repetition_penalty = rep_penalty,
        #no_repeat_ngram_size=3,
        return_full_text=False,
        eos_token_id=2,
    )

    msg = ans[0]["generated_text"]

    #if "##" in msg:
    #    msg = msg.split("##")[0]

    #while ('\\u200b' in msg):
    #    print("replaced!")
    #    msg.replace('\\u200b', '')

    new_state = [{"question": text, "answer": msg}]

    state = state + new_state
    state_chatbot = state_chatbot + [(text, msg)]

    new_talk={"model":cur_model_name,"ask":text,"answer":msg,"temperature":temperature,"top_p":top_p,"top_k":top_k,"rep_penalty":rep_penalty}
    save_file_dir=f'./data/talk{datetime.now().strftime("%m_%d")}.jsonl'
    with jsonlines.open(save_file_dir, mode='a') as writer:
        writer.write(new_talk)
    print(state)
    print(state_chatbot)

    return state, state_chatbot, state_chatbot

def clear_history():
    state = []
    state_chatbot = []
    return state, state_chatbot


with gr.Blocks(css="#chatbot .overflow-y-auto{height:750px}") as demo:
    state = gr.State(
        [
            
        ]
    )
    state_chatbot = gr.State([])

    with gr.Row():
        gr.HTML(
            """<div style="text-align: center; max-width: 500px; margin: 0 auto;">
            <div>
                <h1>ChatKoAlpaca 12.8B (v1.1b-chat-8bit)</h1>
            </div>
            <div>
                Demo runs on RTX 4090 (24GB) with 8bit quantized
            </div>
        </div>"""
        )

    with gr.Row():
        model_selection = Dropdown(choices=MODEL_LIST, label="Select Model", value=MODEL_LIST[0])
        model_selection.change(fn=load_model, inputs=model_selection, outputs=[state, state_chatbot])

    with gr.Row():
        chatbot = gr.Chatbot(elem_id="chatbot")


    with gr.Row():
        txt = gr.Textbox(show_label=False, placeholder="Send a message...").style(
            container=False
        )

    with gr.Row():
        temperature_slider = Slider(minimum=0.1, maximum=1.0, step=0.1, value=0.7, label="Temperature")
        top_p_slider = Slider(minimum=0, maximum=1, step=0.1, value=0.9, label="Top-p")
        top_k_slider = Slider(minimum=0, maximum=100, step=1, value=50, label="Top-k")
        rep_penalty_slider = Slider(minimum=0, maximum=2, step=0.1, value=1, label="rep_penalty")
        clear = gr.ClearButton([txt, chatbot])
        clear.click(fn=clear_history, inputs=[], outputs=[state,state_chatbot])

    txt.submit(answer, [state, state_chatbot, txt, temperature_slider, top_p_slider, top_k_slider, rep_penalty_slider], [state, state_chatbot, chatbot])
    txt.submit(lambda: "", None, txt)

load_model(MODEL_LIST[0])
demo.queue().launch(debug=True, server_name="0.0.0.0", share=True)
