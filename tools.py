import pandas as pd
import os
from argparse import ArgumentParser

def _get_args():
    parser = ArgumentParser()
    parser.add_argument("-c", "--checkpoint-path", type=str, default='Qwen/Qwen-7B-Chat-Int4',
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
    excel_file_=os.path.basename(excelFile).split('.')[0]
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


if __name__ == '__main__':
    file_name = './content/满意度参与详情列表2022_12_02_17_17_39.xls'
    extract_text_from_excle(file_name)
