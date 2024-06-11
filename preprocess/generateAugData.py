import random

from CNN_Test import *
from tool.imblearn.over_sampling import RandomOverSampler
from tool.imblearn.under_sampling import RandomUnderSampler
from tool.imblearn.over_sampling import SMOTE
from tool.imblearn.over_sampling import BorderlineSMOTE
from tool.imblearn.over_sampling import ADASYN
import time
from Tools import *
from ParsingSource import *

now_time = str(int(time.time()))
dump_path = '../data/balanced_dump_data_' + now_time
os.mkdir(dump_path)

root_path_source = '../data/projects_trans/'
root_path_aug = '../data/augment/'
root_path_csv = '../data/csvs/'
package_heads = ['org', 'gnu', 'bsh', 'javax', 'com']

IMBALANCE_PROCESSOR_DEFAULT = RandomOverSampler()  # RandomOverSampler(), RandomUnderSampler(), None, 'cost'
IMBALANCE_PROCESSOR = SMOTE(ratio='auto', random_state=0, k_neighbors=2, m_neighbors=10, out_step=0.5)
# IMBALANCE_PROCESSOR = RandomUnderSampler(ratio=1.0)
# IMBALANCE_PROCESSOR = ADASYN()
# Analyze source and target projects
path_train_and_test = []
with open('../data/pairs-CPDP.txt', 'r') as file_obj:
    for line in file_obj.readlines():
        line = line.strip('\n')
        line = line.strip(' ')
        path_train_and_test.append(line.split(','))

# Loop each pair of combinations
for path in path_train_and_test:

    # File
    path_train_source = root_path_source + path[0]
    path_train_aug_list = get_aug_list(root_path_aug, path[0])
    path_train_handcraft = root_path_csv + path[0] + '.csv'
    path_test_source = root_path_source + path[1]
    path_test_aug_list = get_aug_list(root_path_aug, path[1])
    path_test_handcraft = root_path_csv + path[1] + '.csv'

    # Generate Token
    print(path[0] + "===" + path[1])
    train_project_name = path_train_source.split('/')[3]
    test_project_name = path_test_source.split('/')[3]

    # Get a list of instances of the training and test sets
    train_file_instances = extract_handcraft_instances(path_train_handcraft)
    test_file_instances = extract_handcraft_instances(path_test_handcraft)

    # Get tokens
    dict_token_train = parse_for_aug(path_train_source, train_file_instances, package_heads)
    dict_token_train_aug = parse_aug(path_train_aug_list, train_file_instances, package_heads)
    dict_token_test = parse_for_aug(path_test_source, test_file_instances, package_heads)
    dict_token_test_aug = parse_aug(path_test_aug_list, test_file_instances, package_heads)

    # Turn tokens into numbers
    list_dict, vector_len, vocabulary_size = transform_token_to_number([dict_token_train, dict_token_test])
    list_dict_aug = [0, 0, 0, 0, 0, 0]
    vector_len_aug = [0, 0, 0, 0, 0, 0]
    vocabulary_size_aug = [0, 0, 0, 0, 0, 0]
    dict_encoding_train_aug = []
    dict_encoding_test_aug = []
    for i in range(6):
        list_dict_aug[i], vector_len_aug[i], vocabulary_size_aug[i] = transform_token_to_number([dict_token_train_aug[i], dict_token_test_aug[i]])
        dict_encoding_train_aug.append(list_dict_aug[i][0])
        dict_encoding_test_aug.append(list_dict_aug[i][1])
    dict_encoding_train = list_dict[0]
    dict_encoding_test = list_dict[1]


    # Take out data that can be used for training
    train_ast, train_hand_craft, train_label = extract_data(path_train_handcraft, dict_encoding_train)
    test_ast, test_hand_craft, test_label = extract_data(path_test_handcraft, dict_encoding_test)
    train_ast_aug = [0, 0, 0, 0, 0, 0]
    train_hand_craft_aug = [0, 0, 0, 0, 0, 0]
    train_label_aug = [0, 0, 0, 0, 0, 0]
    test_ast_aug = [0, 0, 0, 0, 0, 0]
    test_handcraft_aug = [0, 0, 0, 0, 0, 0]
    test_label_aug = [0, 0, 0, 0, 0, 0]
    for i in range(6):
        train_ast_aug[i], train_hand_craft_aug[i], train_label_aug[i] = extract_data(path_train_handcraft, dict_encoding_train_aug[i])
        test_ast_aug[i], test_handcraft_aug[i], test_label_aug[i] = extract_data(path_test_handcraft, dict_encoding_test_aug[i])

    train_aug1 = train_ast
    train_aug2 = train_ast
    test_aug1 = test_ast
    test_aug2 = test_ast
    max = 0
    for i in train_ast_aug:
        if i.shape[1]>max:
            max = i.shape[1]
    for i in range(len(train_ast)):
        rand1 = random.randint(0, 5)
        rand2 = random.randint(0, 5)
        while rand1 == rand2:
            rand2 = random.randint(0, 5)
        try:
            train_aug1[i] = train_ast_aug[rand1][i]
        except ValueError:
            train_aug1 = np.pad(train_aug1, ((0, 0), (0, max-train_aug1.shape[1])), 'constant')
            train_ast_aug[rand1] = np.pad(train_ast_aug[rand1], ((0,0),(0, max-train_ast_aug[rand1].shape[1])), 'constant')
            train_aug1[i] = train_ast_aug[rand1][i]
        try:
            train_aug2[i] = train_ast_aug[rand2][i]
        except ValueError:
            train_aug2 = np.pad(train_aug2, ((0, 0), (0, max - train_aug2.shape[1])), 'constant')
            train_ast_aug[rand2] = np.pad(train_ast_aug[rand2], ((0, 0), (0, max - train_ast_aug[rand2].shape[1])), 'constant')
            train_aug2[i] = train_ast_aug[rand2][i]
    max = 0
    for i in test_ast_aug:
        if i.shape[1] > max:
            max = i.shape[1]
    for i in range(len(test_ast)):
        rand1 = random.randint(0, 5)
        rand2 = random.randint(0, 5)
        while rand1 == rand2:
            rand2 = random.randint(0, 5)
        try:
            test_aug1[i] = test_ast_aug[rand1][i]
        except ValueError:
            test_aug1 = np.pad(test_aug1, ((0, 0), (0, max - test_aug1.shape[1])), 'constant')
            test_ast_aug[rand1] = np.pad(test_ast_aug[rand1], ((0, 0), (0, max - test_ast_aug[rand1].shape[1])),
                                          'constant')
            test_aug1[i] = test_ast_aug[rand1][i]
        try:
            test_aug2[i] = test_ast_aug[rand2][i]
        except ValueError:
            test_aug2 = np.pad(test_aug2, ((0, 0), (0, max - test_aug2.shape[1])), 'constant')
            test_ast_aug[rand2] = np.pad(test_ast_aug[rand2], ((0, 0), (0, max - test_ast_aug[rand2].shape[1])),
                                          'constant')
            test_aug2[i] = test_ast_aug[rand2][i]

    if train_aug1.shape[0] < 100:
        train_ast, train_hand_craft, train_aug1, train_aug2, train_label = resam(train_ast, train_hand_craft, train_aug1, train_aug2, train_label)
    if test_aug1.shape[0] < 100:
        test_ast, test_hand_craft, test_aug1, test_aug2, test_label = resam(test_ast, test_hand_craft, test_aug1, test_aug2, test_label)

    # Imbalanced processing
    train_ast, train_aug1, train_aug2, train_hand_craft, train_label, test_ast, test_aug1, test_aug2, test_hand_craft, test_label = imbalance_process_for_aug(train_ast, train_aug1, train_aug2, test_ast, test_aug1, test_aug2, train_hand_craft, test_hand_craft, train_label, test_label, IMBALANCE_PROCESSOR, IMBALANCE_PROCESSOR_DEFAULT)


    # Saved to dump_data
    path_train_and_test_dump = dump_path + '/' + train_project_name + '_to_' + test_project_name
    obj = [train_ast, train_hand_craft, train_label, test_ast, test_hand_craft, test_label, train_aug1, train_aug2, test_aug1, test_aug2, vector_len, vocabulary_size]
    dump_data(path_train_and_test_dump, obj)

    # Saved to csv file
    train_project_name = train_project_name.replace('-', '_')
    train_project_name = train_project_name.replace('.', '_')
    test_project_name = test_project_name.replace('-', '_')
    test_project_name = test_project_name.replace('.', '_')

    path_train_csv = dump_path + '/' + train_project_name + '_to_' + test_project_name + '_of_' + train_project_name + '.csv'
    path_test_csv = dump_path + '/' + train_project_name + '_to_' + test_project_name + '_of_' + test_project_name + '.csv'

    # All labels are 1 and converted to 1,0 to -1
    train_label = np.where(train_label > 0, 1, -1)
    test_label = np.where(test_label > 0, 1, -1)
    # Merge features and labels
    train_hand_craft_data = np.hstack((train_hand_craft,train_label))
    test_hand_craft_data = np.hstack((test_hand_craft, test_label))

    df = pd.DataFrame(data=train_hand_craft_data)
    df.to_csv(path_train_csv, header=None, index=None)
    df = pd.DataFrame(data=test_hand_craft_data)
    df.to_csv(path_test_csv, header=None, index=None)
