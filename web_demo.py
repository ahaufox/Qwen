# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""A simple web interactive chat demo based on gradio."""
import os
from argparse import ArgumentParser

import gradio as gr
import mdtex2html

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig
import shutil

DEFAULT_CKPT_PATH = 'Qwen/Qwen-7B-Chat-Int4'
CONTENT_DIR = 'content'
block_css = """.importantButton {
    background: linear-gradient(45deg, #7e0570,#5d1c99, #6e00ff) !important;
    border: none !important;
}
.importantButton:hover {
    background: linear-gradient(45deg, #ff00e0,#8500ff, #6e00ff) !important;
    border: none !important;
}"""

webui_title = """
# 🎉ChatPDF WebUI🎉
 PS: 1080Ti 11G显存机器，约1min一条😭
"""


def _get_args():
    parser = ArgumentParser()
    parser.add_argument("-c", "--checkpoint-path", type=str, default=DEFAULT_CKPT_PATH,
                        help="Checkpoint name or path, default to %(default)r")
    parser.add_argument("--cpu-only", action="store_true", help="Run demo with CPU only")

    parser.add_argument("--share", action="store_true", default=False,
                        help="Create a publicly shareable link for the interface.")
    parser.add_argument("--inbrowser", action="store_true", default=False,
                        help="Automatically launch the interface in a new tab on the default browser.")
    parser.add_argument("--server-port", type=int, default=8000,
                        help="Demo server port.")
    parser.add_argument("--server-name", type=str, default="127.0.0.1",
                        help="Demo server name.")

    args = parser.parse_args()
    return args


def _load_model_tokenizer(args):
    tokenizer = AutoTokenizer.from_pretrained(
        args.checkpoint_path, trust_remote_code=True, resume_download=True,
    )

    if args.cpu_only:
        device_map = "cpu"
    else:
        device_map = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint_path,
        device_map=device_map,
        trust_remote_code=True,
        resume_download=True,
    ).eval()

    config = GenerationConfig.from_pretrained(
        args.checkpoint_path, trust_remote_code=True, resume_download=True,
    )

    return model, tokenizer, config


def postprocess(self, y):
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert(message),
            None if response is None else mdtex2html.convert(response),
        )
    return y


# gr.Chatbot.postprocess = postprocess

def get_file_list():
    if not os.path.exists("content"):
        return []
    return [f for f in os.listdir("content") if
            f.endswith(".txt") or f.endswith(".pdf") or f.endswith(".docx") or f.endswith(".md")]


file_list = get_file_list()


def upload_file(file):
    if not os.path.exists(CONTENT_DIR):
        os.mkdir(CONTENT_DIR)
    filename = os.path.basename(file.name)
    shutil.move(file.name, os.path.join(CONTENT_DIR, filename))
    # file_list首位插入新上传的文件
    file_list.insert(0, filename)
    return gr.Dropdown.update(choices=file_list, value=filename)


def _parse_text(text):
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split("`")
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f"<br></code></pre>"
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", r"\`")
                    line = line.replace("<", "&lt;")
                    line = line.replace(">", "&gt;")
                    line = line.replace(" ", "&nbsp;")
                    line = line.replace("*", "&ast;")
                    line = line.replace("_", "&lowbar;")
                    line = line.replace("-", "&#45;")
                    line = line.replace(".", "&#46;")
                    line = line.replace("!", "&#33;")
                    line = line.replace("(", "&#40;")
                    line = line.replace(")", "&#41;")
                    line = line.replace("$", "&#36;")
                lines[i] = "<br>" + line
    text = "".join(lines)
    return text


def save_history(task_history):
    if task_history is not None:
        with open(os.path.join('history.txt'),'w+') as f:
            f.writelines(task_history)
            f.close()
        return


def _launch_demo(args, model, tokenizer, config):
    def predict(_query, _chatbot, _task_history):
        user_input = _parse_text(_query)
        save_history(user_input)
        print(f"用户: {user_input}")
        _chatbot.append((user_input, ""))
        full_response = ""

        for response in model.chat_stream(tokenizer, _query, history=_task_history, generation_config=config):
            responses = _parse_text(response)
            _chatbot[-1] = (user_input, responses)
            yield _chatbot
            full_response = responses

        # print(f"History: {_task_history}")
        _task_history.append((_query, full_response))
        print(f"小黑: {full_response}")
        save_history(user_input)

    def regenerate(_chatbot, _task_history):
        if not _task_history:
            yield _chatbot
            return
        item = _task_history.pop(-1)
        _chatbot.pop(-1)
        yield from predict(item[0], _chatbot, _task_history)

    def reset_user_input():
        return gr.update(value="")

    def reset_state(_chatbot, _task_history):
        _task_history.clear()
        _chatbot.clear()
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        return _chatbot

    with gr.Blocks(css=block_css) as demo:
        file_status = gr.State("")

        demo.title = "qwen-demo"
        gr.Markdown("""<center><font size=8>Qwen-Chat Bot</center>\n""")
        gr.Markdown(webui_title)

        task_history = gr.State([])
        # '你是一个专业的数据分析师'
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label='Qwen-Chat', elem_classes="control-height")
                query = gr.Textbox(lines=2, label='Input')

                submit_btn = gr.Button("🚀 Submit (发送)")
                regen_btn = gr.Button("🤔️ Regenerate (重试)")
                empty_btn = gr.Button("🧹 Clear History (清除历史)")
                submit_btn.click(predict, [query, chatbot, task_history], [chatbot], show_progress=True)
                submit_btn.click(reset_user_input, [], [query])
                empty_btn.click(reset_state, [chatbot, task_history], outputs=[chatbot], show_progress=True)
                regen_btn.click(regenerate, [chatbot, task_history], [chatbot], show_progress=True)
            with gr.Column(scale=1):
                with gr.Tab("upload"):
                    file = gr.File(
                        label="content file",
                        file_types=['.txt', '.md', '.docx', '.pdf']
                    )
                load_file_button = gr.Button("加载文件")
                with gr.Tab("select"):
                    selectFile = gr.Dropdown(
                        file_list,
                        label="content file",
                        interactive=True,
                        value=file_list[0] if len(file_list) > 0 else None
                    )
            # 将上传的文件保存到content文件夹下,并更新下拉框
            file.upload(upload_file, inputs=file, outputs=selectFile)
            # local_file_path = os.path.join(CONTENT_DIR, selectFile)
            load_file_button.click(
                # get_vector_store,
                show_progress=True,
                # inputs=[selectFile, chatbot, embedding_model],
                outputs=[selectFile, chatbot],
            )

    demo.queue().launch(
        share=args.share,
        inbrowser=args.inbrowser,
        server_port=args.server_port,
        server_name=args.server_name,
    )


def main():
    args = _get_args()

    model, tokenizer, config = _load_model_tokenizer(args)

    _launch_demo(args, model, tokenizer, config)


if __name__ == '__main__':
    main()
