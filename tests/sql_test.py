from langchain.document_loaders import UnstructuredPDFLoader
from langchain.document_loaders.csv_loader import UnstructuredCSVLoader

f = "../content/满意度参与详情列表2022_12_02_17_17_39.xls"
csv_name = '../content/满意度参与详情列表2022_12_02_17_17_39.csv'
import pandas as pd

df = pd.read_excel(f, sheet_name=0)
df.to_csv(csv_name, encoding='utf-8')

loader = UnstructuredCSVLoader(csv_name, mode="elements")
docs = loader.load()
print(docs)
for i in docs:
    if loader.mode == 'elements':
        print(i.page_content)
    if loader.mode == 'single':
        x = i.page_content
        for j in x.split('\n\n'):
            print(j)
# from pdfminer.pdfinterp import PDFPageInterpreter, PDFResourceManager

loader = UnstructuredPDFLoader(
    "../content/self-instruct 论文提出使用现有的强大语言模型自动生成指令数据.pdf", mode="single", strategy="fast",
)
docs = loader.load()

# print(docs)
for i in docs:
    if loader.mode == 'elements':
        print(i.page_content, i.metadata['page_number'], i.metadata['category'])
    if loader.mode == 'single':
        x = i.page_content
        for j in x.split('\n\n'):
            print(j)
