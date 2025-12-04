import json
import os
import re
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support


def compute_metrics(labels, preds):
    # Calculate precision, recall, F1 score, and accuracy
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0.0)
    acc = accuracy_score(labels, preds)
    class_rep = classification_report(labels, preds, target_names=["0", "1"], output_dict=True, zero_division=0.0)

    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall, "class_rep": class_rep}

def split_lists(input_list):
    result = []

    for sublist in input_list:
        # Create a temporary list to store the split results of the current sublist
        temp_list = []
        for item in sublist:
            if 'and' in item:
                # If the element contains 'and', split it into two elements
                parts = item.split(' and ')

                if len(parts) > 1:  # If there is content after 'and'
                    temp_list.append(parts[0])  # Add the part before 'and'
                    temp_list.append(parts[1])  # Add the part after 'and'
                else:
                    temp_list.append(parts[0])
            else:
                # Otherwise, add the element directly
                temp_list.append(item)
        
        # Split the result into two sublists as needed
        result.append(temp_list[:len(temp_list)//2])
        result.append(temp_list[len(temp_list)//2:])
    
    return result

def extract_after_is_are(text):
    # Extract the content after 'is' or 'are'
    for i in range(len(text) - 1, -1, -1):
        if text[i:i+2] == 'is' or text[i:i+3] == 'are':
            return text[i + 2:].strip() if text[i:i+2] == 'is' else text[i + 3:].strip()
    return text


def remove_punctuation(text):
    # Remove specific phrases and punctuation
    if 'via' in text:
        text = text.split('via', 1)[1]
    if 'a neighbor of ' in text:
        text = text.replace('a neighbor of ', "")
    return re.sub(r'[^\w\s]', '', text).strip()  # 删除所有标点符号


def match_brackets(text):
    # Count the number of left and right brackets
    left_count = text.count('[')
    right_count = text.count(']')

    # If there are more left brackets, add the missing right brackets
    if left_count > right_count:
        text += ']' * (left_count - right_count)

    return text


def parse_list_or_return_string(text):
    # Parse a list or return the string as is
    if text.startswith('[[') and text.endswith(', ...]'):
        text = text.replace(', ...]', ']')
    if text.startswith('[[') and ']]' in text:
        while not text.endswith(']]'):
            text = text[:-1]
    if text.startswith('[[') and text.endswith(']]') or text.startswith('[') and text.endswith(']'):
        final_list = []
        for text_list in text.split('], ['):
            l = []
            for element in text_list.split(','):
                l.append(remove_punctuation(element))
            final_list.append(l)
        return final_list
    else:
        return text


def check_format_for_path(text):
    # Define allowed prefixes and suffixes
    prefixes = [
        "Yes. The paths are[",
        "Yes. The paths are [",
        "Yes. The shortest paths are[",
        "Yes. The shortest paths are [",
        "Yes.\n[",
        "Yes. [",
    ]
    suffix = "]"

    for prefix in prefixes:
        if text.startswith(prefix):
            if text.endswith(suffix):
                return text[len(prefix) - 1:-len(suffix)]
            elif text[len(prefix) - 1:].endswith(suffix):
                return text[len(prefix) - 1:-len(suffix)]
            else:
                return text[len(prefix) - 1:]

    return text


def extract_content(text):
    # Extract content within square brackets
    match = re.search(r'\[(.*?)\]', text)
    if match:
        return match.group(1)
    else:
        return text.split('[', 1)[1] if '[' in text else extract_after_is_are(text)  # If no closing bracket, extract content after the opening bracket


def extract_number_from_words(text):
    words_to_numbers = {
        "zero": 0, "one": 1, "two": 2, "three": 3,
        "four": 4, "five": 5, "six": 6, "seven": 7,
        "eight": 8, "nine": 9, "ten": 10
    }

    match = re.search(r'\b(?:is|are)\s+(\w+)', text)
    if match:
        word = match.group(1).lower()  # Get the word and convert to lowercase
        return str(words_to_numbers.get(word))  # Return the corresponding number
    return text


def extract_number(text):
    if 'is' in text or 'are' in text:
        match = re.search(r'\b(?:is|are)\s+(\d+)', text)
        if match:
            return str(match.group(1))
        else:
            return str(extract_number_from_words(text))
    else:
        match = re.search(r'has a degree of (\d+)', text)
        if match:
            return str(match.group(1))
        else:
            return text
        # return text

def reasoning_metrics(model, reasoning_file_path):
    precisions = {}
    recalls = {}
    hits1 = {}
    fc_gold = {}
    fc_pred = {}
    common_neighbor_check = []
    path_label_gold = []
    path_label_pred = []
    path_em_precisions = []
    path_em_recalls = []
    with open(reasoning_file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            for kt, vt in data['task'].items():
                if kt == 'common_out_neighbor_check':
                    kt = 'common_neighbor_check'
                if not vt['qa']:
                    pass
                else:
                    if kt == 'common_neighbor_check' or kt == 'common_out_neighbor_check':
                        d = vt['qa']
                        if d['prediction'] == d['answer']:
                            common_neighbor_check.append(1)
                        else:
                            common_neighbor_check.append(0)
                    elif kt == 'shortest_connective_path_query' or kt == 'connective_path_query':
                        d = vt['qa']
                        if model == 'internvl':
                            if d['prediction'].startswith('No'):
                                path_ = 'No'
                            else:
                                for sent in d['prediction'].split('\n\n'):
                                    if sent.startswith('[['):
                                        path = check_format_for_path(sent.replace('] and [', ', '))
                                    else:
                                        path = check_format_for_path(sent.replace(' and ', ', '))
                                    while path.endswith('.') or path.endswith(',') or path.endswith(' '):
                                        path = path[:-1]
                                    path = match_brackets(path)
                                    path_ = parse_list_or_return_string(path)
                                    if isinstance(path_, list):
                                        break
                                if not isinstance(path_, list) and path_ != 'No':
                                    path_ = d['prediction']
                                    if 'Yes' in path_.split('\n')[-1]:
                                        path_ = path_.split('\n')[-1]
                                    elif 'Yes' in path_.split('\n')[-2]:
                                        path_ = path_.split('\n')[-2]
                                    elif 'Yes' in path_.split('\n')[-3]:
                                        path_ = path_.split('\n')[-3]
                                    else:
                                        print(path_)
                                    path_ = check_format_for_path(path_.replace('] and [', ', '))
                                    while path_.endswith('.') or path_.endswith(',') or path_.endswith(' '):
                                        path_ = path_[:-1]
                                    path_ = match_brackets(path_)
                                    path_ = parse_list_or_return_string(path_)
                        else:
                            path = check_format_for_path(d['prediction'])
                            while path.endswith('.') or path.endswith(',') or path.endswith(' '):
                                path = path[:-1]
                            path = match_brackets(path)
                            path_ = parse_list_or_return_string(path)
                        gold_path = check_format_for_path(d['answer'])
                        gold_path = match_brackets(gold_path)
                        gold_path_ = parse_list_or_return_string(gold_path)
                        if isinstance(path_, str) and path_.startswith('No'):
                            path_ = 'No'
                        if path_ == 'No':
                            path_label_pred.append(0)
                            if gold_path_ != 'No':
                                path_em_precisions.append(0)
                                path_em_recalls.append(0)
                        else:
                            path_label_pred.append(1)
                            if gold_path_ == 'No':
                                path_em_precisions.append(0)
                                path_em_recalls.append(0)
                            else:
                                correct_path = 0
                                for p in path_:
                                    if p in gold_path_:
                                        correct_path += 1
                                path_em_precisions.append(correct_path / len(path_))
                                path_em_recalls.append(correct_path / len(gold_path_))

                        if gold_path_ == 'No':
                            path_label_gold.append(0)
                        else:
                            path_label_gold.append(1)

                    else:
                        correct_ans = 0
                        if kt not in precisions:
                            precisions[kt] = []
                        if kt not in recalls:
                            recalls[kt] = []
                        if kt not in hits1:
                            hits1[kt] = []

                        d = vt['qa']
                        pred = extract_content(d["prediction"]).replace('.', '').replace(' and ', ', ')
                        gold = d['answer']

                        pred = pred.split(', ')
                        # print(pred)
                        if remove_punctuation(pred[0]) in gold:
                            hits1[kt].append(1)
                        else:
                            hits1[kt].append(0)
                        for p in pred:
                            p_ = remove_punctuation(p)
                            # print(p_)
                            if p_ in gold:
                                correct_ans += 1
                        recalls[kt].append(correct_ans / len(gold))
                        precisions[kt].append(correct_ans / len(pred))

                if not vt['fc_sup']:
                    pass
                else:
                    d = vt['fc_sup']
                    if kt not in fc_gold:
                        fc_gold[kt] = []
                    if kt not in fc_pred:
                        fc_pred[kt] = []
                    # print(type(d['label']))
                    if d['label'] == 'True':
                        fc_gold[kt].append(1)
                    elif d['label'] == 'False':
                        fc_gold[kt].append(0)
                    else:
                        print(data['id'])
                        print(vt['fc_sup'])

                    if 'is' in d['prediction']:
                        d['prediction'] = extract_after_is_are(d['prediction']).replace(".", "").strip()
                    if 'Solution:' in d['prediction']:
                        d['prediction'] = d['prediction'].split('Solution:', 1)[1].replace(".", "").strip()
                    if 'be' in d['prediction']:
                        d['prediction'] = remove_punctuation(d['prediction'].split('be', 1)[1].replace(".", "")).strip()

                    if remove_punctuation(d['prediction']).lower() == 'true' or remove_punctuation(d['prediction']) == '1' or remove_punctuation(d['prediction']).lower() == 'yes':
                        fc_pred[kt].append(1)
                    elif remove_punctuation(d['prediction']).lower() == 'false' or remove_punctuation(d['prediction']) == '0':
                        fc_pred[kt].append(0)
                    else:
                        if d['label'] == 'True':
                            fc_pred[kt].append(0)
                        elif d['label'] == 'False':
                            fc_pred[kt].append(1)

                if not vt['fc_ref']:
                    pass
                else:
                    d = vt['fc_ref']
                    if kt not in fc_gold:
                        fc_gold[kt] = []
                    if kt not in fc_pred:
                        fc_pred[kt] = []
                    # print(type(d['label']))
                    if d['label'] == 'True':
                        fc_gold[kt].append(1)
                    elif d['label'] == 'False':
                        fc_gold[kt].append(0)
                    else:
                        print(data['id'])
                        print(vt['fc_ref'])

                    if 'is' in d['prediction']:
                        d['prediction'] = extract_after_is_are(d['prediction']).replace(".", "").strip()
                    if 'Solution:' in d['prediction']:
                        d['prediction'] = d['prediction'].split('Solution:', 1)[1].replace(".", "").strip()
                    if 'be' in d['prediction']:
                        d['prediction'] = remove_punctuation(d['prediction'].split('be', 1)[1].replace(".", "")).strip()

                    if remove_punctuation(d['prediction']).lower() == 'true' or remove_punctuation(d['prediction']) == '1' or remove_punctuation(d['prediction']).lower() == 'yes':
                        fc_pred[kt].append(1)
                    elif remove_punctuation(d['prediction']).lower() == 'false' or remove_punctuation(d['prediction']) == '0':
                        fc_pred[kt].append(0)
                    else:
                        if d['label'] == 'True':
                            fc_pred[kt].append(0)
                        elif d['label'] == 'False':
                            fc_pred[kt].append(1)

    precisions_final = {}
    recalls_final = {}
    f1_final = {}
    hits1_final = {}

    fc_precisions_final = {}
    fc_recalls_final = {}
    fc_f1_final = {}
    fc_acc_final = {}

    for key in precisions.keys():
        p = sum(precisions[key]) / len(precisions[key])
        precisions_final[key] = p
        r = sum(recalls[key]) / len(recalls[key])
        recalls_final[key] = r
        if p + r != 0:
            f1_final[key] = 2 * (p * r) / (p + r)
        else:
            f1_final[key] = 0
        hits1_final[key] = sum(hits1[key]) / len(hits1[key])

    for key in fc_pred.keys():
        fc_metrics = compute_metrics(fc_gold[key], fc_pred[key])
        fc_acc_final[key] = fc_metrics['accuracy']
        fc_f1_final[key] = fc_metrics['f1']
        fc_precisions_final[key] = fc_metrics['precision']
        fc_recalls_final[key] = fc_metrics['recall']

    final_path_em_precision = sum(path_em_precisions) / len(path_em_precisions)
    final_path_em_recall = sum(path_em_recalls) / len(path_em_recalls)
    if final_path_em_precision + final_path_em_recall != 0:
        final_path_em_f1 = 2 * (final_path_em_precision * final_path_em_recall) / (final_path_em_precision + final_path_em_recall)
    else:
        final_path_em_f1 = 0

    final_path_label_metrics = compute_metrics(path_label_gold, path_label_pred)
    final_path_label_acc = final_path_label_metrics['accuracy']
    final_path_label_f1 = final_path_label_metrics['f1']
    final_path_label_precision = final_path_label_metrics['precision']
    final_path_label_recall = final_path_label_metrics['recall']

    print(f'Precision: {precisions_final}')
    print(f'Recall: {recalls_final}')
    print(f'F1: {f1_final}')
    print(f'Hits@1: {hits1_final}')
    print(f'common_neighbor_check_acc: {sum(common_neighbor_check) / len(common_neighbor_check)}')

    print(f'FC_Precision: {fc_precisions_final}')
    print(f'FC_Recall: {fc_recalls_final}')
    print(f'FC_F1: {fc_f1_final}')
    print(f'FC_Accuracy: {fc_acc_final}')

    print(f'Path_EM_Precision: {final_path_em_precision}')
    print(f'Path_EM_Recall: {final_path_em_recall}')
    print(f'Path_EM_F1: {final_path_em_f1}')

    print(f'Path_Label_Precision: {final_path_label_precision}')
    print(f'Path_Label_Recall: {final_path_label_recall}')
    print(f'Path_Label_F1: {final_path_label_f1}')
    print(f'Path_Label_Accuracy: {final_path_label_acc}')

    return precisions_final, recalls_final, f1_final, hits1_final, fc_precisions_final, fc_recalls_final, fc_f1_final, fc_acc_final, common_neighbor_check, final_path_em_precision, final_path_em_recall, final_path_em_f1, final_path_label_precision, final_path_label_recall, final_path_label_f1, final_path_label_acc


def understanding_metrics(model, understanding_file_path):
    precisions = {}
    recalls = {}
    hits1 = {}
    qa_acc = {}
    fc_gold = {}
    fc_pred = {}

    with open(understanding_file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            for kt, vt in data['task'].items():
                if not vt['qa']:
                    pass
                else:
                    if kt == 'neighbor_query':
                        correct_ans = 0
                        if kt not in precisions:
                            precisions[kt] = []
                        if kt not in recalls:
                            recalls[kt] = []
                        if kt not in hits1:
                            hits1[kt] = []

                        d = vt['qa']
                        pred = extract_content(d["prediction"]).replace('.', '').replace(' and ', ', ')
                        gold = d['answer']

                        pred = pred.split(', ')
                        # print(pred)
                        if remove_punctuation(pred[0]) in gold:
                            hits1[kt].append(1)
                        else:
                            hits1[kt].append(0)
                        for p in pred:
                            p_ = remove_punctuation(p)
                            # print(p_)
                            if p_ in gold:
                                correct_ans += 1
                        recalls[kt].append(correct_ans / len(gold))
                        precisions[kt].append(correct_ans / len(pred))
                    else:
                        d = vt['qa']
                        if kt not in qa_acc:
                            qa_acc[kt] = []

                        if kt == 'directed_check':
                            if d['prediction'] != "Yes" and d['prediction'] != "No":
                                d['prediction'] = d['prediction'].split(':', 1)[-1].strip()
                                if d['prediction'] != "Yes" and d['prediction'] != "No":
                                    # print(d['prediction'])
                                    pass
                        else:
                            if not d['prediction'].isdigit():
                                d['prediction'] = extract_number(d['prediction'])
                                if not d['prediction'].isdigit():
                                    print(data['graph_image'])
                                    print(kt)
                                    print(d['prediction'])

                        assert type(d['prediction']) == type(d['answer'])
                        if d['prediction'] == d['answer']:
                            qa_acc[kt].append(1)
                        else:
                            qa_acc[kt].append(0)

                if not vt['fc_sup']:
                    pass
                else:
                    d = vt['fc_sup']
                    if kt not in fc_gold:
                        fc_gold[kt] = []
                    if kt not in fc_pred:
                        fc_pred[kt] = []
                    # print(type(d['label']))
                    if d['label'] == 'True':
                        fc_gold[kt].append(1)
                    elif d['label'] == 'False':
                        fc_gold[kt].append(0)
                    else:
                        print(data['id'])
                        print(vt['fc_sup'])

                    if 'is' in d['prediction']:
                        d['prediction'] = extract_after_is_are(d['prediction']).replace(".", "").strip()
                    if 'Solution:' in d['prediction']:
                        d['prediction'] = d['prediction'].split('Solution:', 1)[1].replace(".", "").strip()
                    if 'be' in d['prediction']:
                        d['prediction'] = remove_punctuation(d['prediction'].split('be', 1)[1].replace(".", "")).strip()

                    if remove_punctuation(d['prediction']).lower() == 'true' or remove_punctuation(
                            d['prediction']) == '1' or remove_punctuation(d['prediction']).lower() == 'yes':
                        fc_pred[kt].append(1)
                    elif remove_punctuation(d['prediction']).lower() == 'false' or remove_punctuation(
                            d['prediction']) == '0':
                        fc_pred[kt].append(0)
                    else:
                        if d['label'] == 'True':
                            fc_pred[kt].append(0)
                        elif d['label'] == 'False':
                            fc_pred[kt].append(1)

                if not vt['fc_ref']:
                    pass
                else:
                    d = vt['fc_ref']
                    if kt not in fc_gold:
                        fc_gold[kt] = []
                    if kt not in fc_pred:
                        fc_pred[kt] = []
                    # print(type(d['label']))
                    if d['label'] == 'True':
                        fc_gold[kt].append(1)
                    elif d['label'] == 'False':
                        fc_gold[kt].append(0)
                    else:
                        print(data['id'])
                        print(vt['fc_ref'])

                    if 'is' in d['prediction']:
                        d['prediction'] = extract_after_is_are(d['prediction']).replace(".", "").strip()
                    if 'Solution:' in d['prediction']:
                        d['prediction'] = d['prediction'].split('Solution:', 1)[1].replace(".", "").strip()
                    if 'be' in d['prediction']:
                        d['prediction'] = remove_punctuation(d['prediction'].split('be', 1)[1].replace(".", "")).strip()

                    if remove_punctuation(d['prediction']).lower() == 'true' or remove_punctuation(
                            d['prediction']) == '1' or remove_punctuation(d['prediction']).lower() == 'yes':
                        fc_pred[kt].append(1)
                    elif remove_punctuation(d['prediction']).lower() == 'false' or remove_punctuation(
                            d['prediction']) == '0':
                        fc_pred[kt].append(0)
                    else:
                        if d['label'] == 'True':
                            fc_pred[kt].append(0)
                        elif d['label'] == 'False':
                            fc_pred[kt].append(1)

    precisions_final = {}
    recalls_final = {}
    f1_final = {}
    hits1_final = {}
    qa_acc_final = {}
    fc_precisions_final = {}
    fc_recalls_final = {}
    fc_f1_final = {}
    fc_acc_final = {}

    for key in precisions.keys():
        p = sum(precisions[key]) / len(precisions[key])
        precisions_final[key] = p
        r = sum(recalls[key]) / len(recalls[key])
        recalls_final[key] = r
        if p + r != 0:
            f1_final[key] = 2 * (p * r) / (p + r)
        else:
            f1_final[key] = 0
        hits1_final[key] = sum(hits1[key]) / len(hits1[key])

    for key in fc_pred.keys():
        fc_metrics = compute_metrics(fc_gold[key], fc_pred[key])
        fc_acc_final[key] = fc_metrics['accuracy']
        fc_f1_final[key] = fc_metrics['f1']
        fc_precisions_final[key] = fc_metrics['precision']
        fc_recalls_final[key] = fc_metrics['recall']

    for key in qa_acc.keys():
        qa_acc_final[key] = sum(qa_acc[key]) / len(qa_acc[key])

    print(f'Precision: {precisions_final}')
    print(f'Recall: {recalls_final}')
    print(f'F1: {f1_final}')
    print(f'Hits@1: {hits1_final}')
    print(f'QA_Acc: {qa_acc_final}')

    print(f'FC_Precision: {fc_precisions_final}')
    print(f'FC_Recall: {fc_recalls_final}')
    print(f'FC_F1: {fc_f1_final}')
    print(f'FC_Accuracy: {fc_acc_final}')

    return precisions_final, recalls_final, f1_final, hits1_final, fc_precisions_final, fc_recalls_final, fc_f1_final, fc_acc_final, qa_acc_final


if __name__ == '__main__':
    reasoning_file_path = 'xxx'  # replace with your file path of reasoning results
    understanding_file_path = 'xxx'  # replace with your file path of understanding results
    model_name = 'qwen2vl'

    os.makedirs('./results', exist_ok=True)
    f = open(f'./results/results_{model_name}.txt', 'w')

    # Reasoning
    precisions_final, recalls_final, f1_final, hits1_final, fc_precisions_final, fc_recalls_final, fc_f1_final, fc_acc_final, common_neighbor_check, final_path_em_precision, final_path_em_recall, final_path_em_f1, final_path_label_precision, final_path_label_recall, final_path_label_f1, final_path_label_acc = reasoning_metrics(
        model_name, reasoning_file_path)
    f.write('Reasoning:\n')
    f.write("QA:\n")
    f.write(f'Precision: {precisions_final}\n')
    f.write(f'Recall: {recalls_final}\n')
    f.write(f'F1: {f1_final}\n')
    f.write(f'Hits@1: {hits1_final}\n')
    f.write('\n')
    f.write(f'common_neighbor_check_acc: {sum(common_neighbor_check) / len(common_neighbor_check)}\n')
    f.write('\n')
    f.write('FC:\n')
    f.write(f'FC_Precision: {fc_precisions_final}\n')
    f.write(f'FC_Recall: {fc_recalls_final}\n')
    f.write(f'FC_F1: {fc_f1_final}\n')
    f.write(f'FC_Accuracy: {fc_acc_final}\n')
    f.write('\n')
    f.write('Path:\n')
    f.write(f'Path_EM_Precision: {final_path_em_precision}\n')
    f.write(f'Path_EM_Recall: {final_path_em_recall}\n')
    f.write(f'Path_EM_F1: {final_path_em_f1}\n')
    f.write('\n')
    f.write(f'Path_Label_Precision: {final_path_label_precision}\n')
    f.write(f'Path_Label_Recall: {final_path_label_recall}\n')
    f.write(f'Path_Label_F1: {final_path_label_f1}\n')
    f.write(f'Path_Label_Accuracy: {final_path_label_acc}\n')
    f.write('\n\n')
    #
    # Understanding
    precisions_final, recalls_final, f1_final, hits1_final, fc_precisions_final, fc_recalls_final, fc_f1_final, fc_acc_final, qa_acc_final = understanding_metrics(
        model_name, understanding_file_path)
    f.write('Understanding:\n')
    f.write("QA:\n")
    f.write(f'Precision: {precisions_final}\n')
    f.write(f'Recall: {recalls_final}\n')
    f.write(f'F1: {f1_final}\n')
    f.write(f'Hits@1: {hits1_final}\n')
    f.write('\n')
    f.write(f'QA_Acc: {qa_acc_final}\n')
    f.write('\n')
    f.write('FC:\n')
    f.write(f'FC_Precision: {fc_precisions_final}\n')
    f.write(f'FC_Recall: {fc_recalls_final}\n')
    f.write(f'FC_F1: {fc_f1_final}\n')
    f.write(f'FC_Accuracy: {fc_acc_final}\n')
    f.write('\n')

    f.close()