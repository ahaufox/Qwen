import pandas as pd
import os
from argparse import ArgumentParser
import shutil

DEFAULT_CKPT_PATH = 'Qwen/Qwen-7B-Chat-Int4'
CONTENT_DIR = 'content'



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


def extract_text_from_excle(excelFile):
    # 数据文件
    excel_file = excelFile
    excel_file_ = os.path.basename(excelFile).split('.')[0]
    excel = pd.read_excel(excel_file)  # 读取Excel表格
    excel_table_head = list(excel.columns.values)  # 读取表头
    table_head = '|'.join(excel_table_head) + "\n"  # 拼接表头
    # 获取表格主体
    excel_table_body = list(excel.iloc[0:].values)
    new_table_body = []
    # 将每一个列表项转换为字符串
    for i in excel_table_body:
        row = []
        for j in i:  # 对这一行的遍历
            row.append(str(j))  # 转换为字符串并加入row列表
        new_table_body.append(row)  # 再将row加入new_table_body
    # 拼接列表主体
    table_body = '\n'.join(['|'.join(i) for i in new_table_body])
    # 制作列表分隔符
    table_split = '-|' * len(excel_table_head) + "\n"
    # 拼接成table变量
    table = table_head + table_split + table_body
    # 输出到文件
    with open("./content/{}.txt".format(excel_file_), "w", encoding="UTF-8") as f:
        f.write(table)


def extract_text_from_txt(file_path: str):
    """Extract text content from a TXT file."""
    contents = []
    with open(file_path, 'r', encoding='utf-8') as f:
        contents = [text.strip() for text in f.readlines() if text.strip()]
    return contents


def extract_text_from_pdf(file_path: str):
    """Extract text content from a PDF file."""
    import PyPDF2
    contents = []
    with open(file_path, 'rb') as f:
        pdf_reader = PyPDF2.PdfReader(f)
        for page in pdf_reader.pages:
            page_text = page.extract_text().strip()
            raw_text = [text.strip() for text in page_text.splitlines() if text.strip()]
            new_text = ''
            for text in raw_text:
                new_text += text
                if text[-1] in ['.', '!', '?', '。', '！', '？', '…', ';', '；', ':', '：', '”', '’', '）', '】', '》', '」',
                                '』', '〕', '〉', '》', '〗', '〞', '〟', '»', '"', "'", ')', ']', '}']:
                    contents.append(new_text)
                    new_text = ''
            if new_text:
                contents.append(new_text)
    return contents


def save_history(task_history):
    with open(os.path.join('history.txt'), 'a') as f:
        f.writelines(task_history)
        f.writelines('\n')
        f.close()

def get_file_list():
    if not os.path.exists(CONTENT_DIR):
        return []
    return [f for f in os.listdir(CONTENT_DIR) if
            f.endswith(".txt") or f.endswith(".pdf") or f.endswith(".docx") or f.endswith(".md")]


file_list = get_file_list()


def upload_file(file):
    if not os.path.exists(CONTENT_DIR):
        os.mkdir(CONTENT_DIR)
    filename = os.path.basename(file.name)
    shutil.move(file.name, os.path.join(CONTENT_DIR, filename))
    # file_list首位插入新上传的文件
    file_list.insert(0, filename)
    return file_list

def load_doc_files(doc_files):
    """Load document files."""
    corpus = []
    if isinstance(doc_files, str):
        doc_files = [doc_files]
    if doc_files is None:
        pass
        return None
    else:
        for doc_file in doc_files:
            if doc_file.endswith('.pdf'):
                corpus.append(extract_text_from_pdf(doc_file))
            # elif doc_file.endswith('.docx'):
            #     corpus = self.extract_text_from_docx(doc_file)
            # elif doc_file.endswith('.md'):
            #     corpus = self.extract_text_from_markdown(doc_file)
            else:
                corpus.append(extract_text_from_txt(doc_file))
            # sim_model.add_corpus(corpus)
        return corpus
if __name__ == '__main__':
    import time

    # y=[(None, '你好', '')]
    # print(enumerate(y))
    # for i, (message,response,h) in enumerate(y):
    #     print(i,message,response,h)
    # time.sleep(101)
    file_name = './content/满意度参与详情列表2022_12_02_17_17_39.xls'
    import pandas as pd

    df = pd.read_excel(file_name)
    df.to_csv('t.csv')
    # 创建一个新的Document对象
    # 将数据写入MD文件

    extract_text_from_excle(file_name)
