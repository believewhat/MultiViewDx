import pandas as pd
import os
import requests
from concurrent.futures import ThreadPoolExecutor

# 读取CSV文件
csv_file = '20240801_radiopaedia.csv'  # 请将其替换为实际的CSV文件路径
data = pd.read_csv(csv_file)

# 替换非法字符的函数
def sanitize_folder_name(name):
    return name.replace('/', '_').replace('?', '_').replace(' ', '_')

import sys
import json
import json
import json
import ipdb
import glob
import os
import ipdb
from openai import OpenAI
import os
import json
import csv
import pandas as pd
import argparse
import base64
import time
import concurrent.futures
os.environ["OPENAI_API_KEY"]=''

client = OpenAI()

data = pd.read_csv('20240801_radiopaedia.csv')

data2 = pd.read_csv('radiopaedia[不含图片信息].csv')

import ipdb
ipdb.set_trace()

system_prompt