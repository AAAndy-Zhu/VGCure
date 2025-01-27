import json
from pathlib import Path

def replace_first_slash(input_string):
    # 找到第一个 '/' 的位置
    first_slash_index = input_string.find('/')
    
    # 如果找到了 '/'，替换第一个 '/' 为 '_'
    if first_slash_index != -1:
        input_string = input_string[:first_slash_index] + '_' + input_string[first_slash_index+1:]
    
    return input_string

path = Path('understanding')
for file in path.rglob('*'):  # rglob('*') 会递归查找所有文件
    all_data = []
    with open(file, 'r') as f:
        for line in f:
            data = json.loads(line)
            data['graph_image'] = replace_first_slash(data['graph_image'])
            all_data.append(data)
    with open(file, 'w') as f:
        for d in all_data:
            f.write(json.dumps(d))
            f.write('\n')
