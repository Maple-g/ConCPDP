import time
import os
from Tools import *
from ParsingSource import *
from shutil import copyfile


def copy_javafile(project_root_path, handcraft_file_names, package_heads, project_name):
    result = {}
    count = 0
    existed_file_names = []
    file_list = ''
    try:
        os.mkdir("./" + project_name)
    except FileExistsError:
        return

    for dir_path, dir_names, file_names in os.walk(project_root_path):

        if len(file_names) == 0:
            continue

        index = -1
        for _head in package_heads:
            index = int(dir_path.find(_head))
            if index >= 0:
                break
        if index < 0:
            continue
        package_name = dir_path[index:]
        package_name = package_name.replace(os.sep, '.')

        for file in file_names:
            file_list = os.path.join(dir_path, file)
            if file.endswith('java'):  # 检查文件名是否为java 后缀文件
                if str(package_name + "." + str(file)) not in handcraft_file_names:
                    continue

                copyfile(file_list, "./" + project_name + "/" + file)
                count += 1

    print("data size : " + str(count))
    return result


now_time = str(int(time.time()))
dump_path = '../data/balanced_dump_data_' + now_time
os.mkdir(dump_path)

root_path_source = '../data/projects/'
root_path_csv = '../data/csvs/'
package_heads = ['org', 'gnu', 'bsh', 'javax', 'com']

path_train_and_test = []
with open('../data/pairs.txt', 'r') as file_obj:
    for line in file_obj.readlines():
        line = line.strip('\n')
        line = line.strip(' ')
        path_train_and_test.append(line.split(','))

# Loop each pair of combinations
for path in path_train_and_test:
    # File
    path_train_source = root_path_source + path[0]
    path_train_handcraft = root_path_csv + path[0] + '.csv'
    path_test_source = root_path_source + path[1]
    path_test_handcraft = root_path_csv + path[1] + '.csv'

    train_file_instances = extract_handcraft_instances(path_train_handcraft)
    test_file_instances = extract_handcraft_instances(path_test_handcraft)

    # Generate Token
    print(path[0] + "===" + path[1])
    train_project_name = path_train_source.split('/')[3]
    test_project_name = path_test_source.split('/')[3]

    dict_token_train = copy_javafile(path_train_source, train_file_instances, package_heads, train_project_name)
    dict_token_test = copy_javafile(path_test_source, test_file_instances, package_heads, test_project_name)
