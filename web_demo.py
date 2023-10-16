# Copyright (c) Alibaba Cloud.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""A simple web interactive chat demo based on gradio."""
import os
import gradio as gr
import mdtex2html
from tools import extract_text_from_excle, extract_text_from_pdf, extract_text_from_txt, _get_args, get_file_list, \
    upload_file, save_history, load_doc_files
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BloomForCausalLM, BloomTokenizerFast, AutoModel
from transformers.generation import GenerationConfig
import shutil
import warnings

warnings.filterwarnings('ignore')

# DEFAULT_CKPT_PATH = 'Qwen/Qwen-7B-Chat-Int4'T

block_css = """.importantButton {
    background: linear-gradient(45deg, #7e0570,#5d1c99, #6e00ff) !important;
    border: none !important;
}
.importantButton:hover {
    background: linear-gradient(45deg, #ff00e0,#8500ff, #6e00ff) !important;
    border: none !important;
}"""
webui_title = """"""
MODEL_CLASSES = {
    "chatglm": (AutoModel, AutoTokenizer),
    "llama": (AutoModelForCausalLM, AutoTokenizer),
    "Qwen": (AutoModelForCausalLM, AutoTokenizer),
    'auto': (AutoModel, AutoTokenizer)
}

"""Override Chatbot.postprocess"""


def postprocess(self, y):
    # [(None, '你好', '')]
    if y is None:
        return []
    for i, (message, response) in enumerate(y):
        y[i] = (
            None if message is None else mdtex2html.convert((message)),
            None if response is None else mdtex2html.convert(response),
        )
    return y


gr.Chatbot.postprocess = postprocess


def _load_model_tokenizer(args):
    if args.cpu_only:
        device_map = "cpu"
    else:
        device_map = "auto"
    global webui_title
    model_type_keys = MODEL_CLASSES.keys()
    model_class, tokenizer_class = [], []
    for k in model_type_keys:
        if k in args.checkpoint_path:
            model_class, tokenizer_class = MODEL_CLASSES[k]
            break
    if model_class == []:
        # config = GenerationConfig.from_pretrained(
        #     args.checkpoint_path, trust_remote_code=True, resume_download=True,
        # )
        model_class, tokenizer_class = MODEL_CLASSES['auto']
    model = model_class.from_pretrained(
        args.checkpoint_path,
        device_map=device_map,
        trust_remote_code=True,
        resume_download=True,
    )
    model = model if '4' in args.checkpoint_path else model.half()
    model = model.eval()
    tokenizer = tokenizer_class.from_pretrained(
        args.checkpoint_path, trust_remote_code=True, resume_download=True,
    )
    config = GenerationConfig.from_pretrained(
        args.checkpoint_path, trust_remote_code=True, resume_download=True,
    )
    webui_title = """🎉WebUI🎉\n
    PS:Qwen/Qwen-7B-Chat-Int4 7G左右显存1080Ti约 30s 一条🚀 ;\n
    Qwen/Qwen-7B-Chat 10G左右显存 1080Ti约 2min 一条😭;\n
    Qwen/Qwen-14B-Chat-Int4 10G左右显存 1080Ti约 1min 一条😭\n
    当前模型: <font size=6>{}</font>
    """.format(args.checkpoint_path)
    return model, tokenizer, config


file_list = get_file_list()


def _parse_text(text):
    """copy from https://github.com/GaiZhenbiao/ChuanhuChatGPT/"""
    lines = text.split("\n")
    lines = [line for line in lines if line != ""]
    count = 0
    for i, line in enumerate(lines):
        if "```" in line:
            count += 1
            items = line.split('`')
            if count % 2 == 1:
                lines[i] = f'<pre><code class="language-{items[-1]}">'
            else:
                lines[i] = f'<br></code></pre>'
        else:
            if i > 0:
                if count % 2 == 1:
                    line = line.replace("`", "\`")
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


def _launch_demo(args, model, tokenizer, config):
    def predict(_query, _chatbot, _task_history):

        user_input = _parse_text(_query)
        print(f"用户: {user_input}")
        save_history(f"用户: {user_input}")
        _chatbot.append(( user_input, ""))
        if 'llm' in args.checkpoint_path:
            print('llm模型……')
            print(f"小黑:")
            for response, _task_history in model.stream_chat(tokenizer, _query, history=_task_history):
                response = _parse_text(response)
                _chatbot[-1] = (user_input, response)
                yield _chatbot
        else:
            print(f"小黑:")
            for response in model.chat_stream(tokenizer, _query, history=_task_history, generation_config=config):
                response = _parse_text(response)
                _chatbot[-1] = (user_input, response)
                yield _chatbot
            print(f" {response}")
        # fresponses = response
        save_history(f"小黑: {response}")
        # print(f"History: {_task_history}")
        _task_history.append((_query, response))

    def reset_user_input():
        return gr.update(value='')

    def regenerate(_chatbot, _task_history):
        if not _task_history:
            yield _chatbot
            return
        item = _task_history.pop(-1)
        _chatbot.pop(-1)
        yield from predict(item[0], _chatbot, _task_history, file_status)

    with gr.Blocks(css=block_css) as demo:


        demo.title = "demo"
        gr.Markdown(webui_title)
        task_history = gr.State([])
        # '你是一个专业的数据分析师'
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label='Chat', elem_classes="control-height")
                query = gr.Textbox(lines=2, label='Input')

                submit_btn = gr.Button("🚀 Submit (发送)")
                regen_btn = gr.Button("🤔️ Regenerate (重试)")
                # empty_btn = gr.Button("🧹 Clear History (清除历史)")
                submit_btn.click(predict, [query, chatbot, task_history], [chatbot], show_progress=True)
                submit_btn.click(reset_user_input, [], [query])
                regen_btn.click(regenerate, [chatbot, task_history], [chatbot], show_progress=True)

            with gr.Column(scale=1):
                with gr.Tab("upload"):
                    file = gr.File(
                        label="content",
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
                # load_file_button.click(
                #     # get_vector_store,
                #     show_progress=True,
                #     # inputs=[selectFile, chatbot, embedding_model],
                #     outputs=[selectFile, select],
                # )

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
