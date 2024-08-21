import random

super_label_map = {
    0: "airplane",
    1: "bird",
    2: "car",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "horse",
    7: "monkey",
    8: "ship",
    9: "truck"
}


def find_avmis(main_class, predicted_class):
    type1_super_groups = ['car', 'deer', 'horse', 'truck']
    type2_super_groups = ['airplane', 'bird', 'cat', 'dog', 'monkey', 'ship']
    main_super_label = super_label_map[main_class]
    predicted_super_label = super_label_map[predicted_class]

    if main_super_label != predicted_super_label:
        if main_super_label in type1_super_groups and predicted_super_label in type1_super_groups:
            return True
        elif main_super_label in type1_super_groups and predicted_super_label in type2_super_groups:
            return True
        else:
            return False
    else:
        return False


def get_statistics_without_fault_injection(original_label_list, predicted_label_list):
    correct_classification = 0
    misclassified_avmis = 0
    misclassified_non_avmis = 0
    correct_indices = []
    avmis_indexes = []
    non_avmis_indexes = []
    for i in range(len(original_label_list)):
        org_val = original_label_list[i]
        pred_val = predicted_label_list[i]
        if org_val != pred_val:
            if find_avmis(org_val, pred_val):
                misclassified_avmis += 1
                avmis_indexes.append(i)
            else:
                misclassified_non_avmis += 1
                non_avmis_indexes.append(i)
        else:
            correct_classification += 1
            correct_indices.append(i)
    return correct_indices, avmis_indexes, non_avmis_indexes, correct_classification, misclassified_avmis, misclassified_non_avmis


def get_statistics_per_image_with_fault_injection(previous_predicted_label_list, faulty_label_list):
    benign_count = 0
    faulty_avmis = 0
    faulty_non_avmis = 0
    for i in range(len(previous_predicted_label_list)):
        prev_pred_val = previous_predicted_label_list[i]
        faulty_val = faulty_label_list[i]
        if prev_pred_val != faulty_val:
            if find_avmis(prev_pred_val, faulty_val):
                faulty_avmis += 1
            else:
                faulty_non_avmis += 1
        else:
            benign_count += 1
    return benign_count, faulty_avmis, faulty_non_avmis


def get_string(data_list, index, text, model_count):
    s = text + '\t'
    for i in range(0, model_count):
        s += '{:.6f}'.format(data_list[i][index]) + '\t'
    return s + '\n'


def main():
    dataset_name = 'stl-10'
    model_names = ['ResNet50']
    model_list = []
    data_list = []
    for model_name in model_names:
        model_list.append(model_name)
        data = []
        file1 = open('../dataset/stl-10/' + dataset_name + '_final_log_' + model_name + '.txt', 'r')
        lines = file1.readlines()
        lines.pop(0)
        if dataset_name == 'imagenet':
            lines.pop(0)
        y_val = []
        predicted_label_list = []
        for i in range(8000):
            line_parts = [x.strip() for x in lines[0].split(':')]
            y_val.append(int(line_parts[1]))
            predicted_label_list.append(int(line_parts[2]))
            lines.pop(0)

        correct_indices, avmis_indexes, non_avmis_indexes, correct_classification, misclassified_avmis, misclassified_non_avmis \
            = get_statistics_without_fault_injection(y_val, predicted_label_list)
        data.append(correct_classification)
        data.append(misclassified_avmis)
        data.append(misclassified_non_avmis)

        fi_results = []
        fi_taken = []
        for i in range(8000):
            fi_taken.append(0)
            fi_results.append([])

        while len(lines) > 0:
            line_parts = [x.strip() for x in lines[0].split(':')]
            index = int(line_parts[0])
            fi_results[index].append([int(line_parts[1]), int(line_parts[3])])
            lines.pop(0)

        prev_pred_list = []
        faulty_list = []
        for i in range(len(correct_indices)):
            while True:
                image_index = correct_indices[i]
                fi_taken_image_index = fi_taken[image_index]
                if fi_taken_image_index < 5:
                    break
            fi_data = fi_results[image_index][fi_taken_image_index]
            faulty_list.append(fi_data[1])
            prev_pred_list.append(fi_data[0])
            fi_taken[image_index] += 1

        benign_count, faulty_avmis, faulty_non_avmis = get_statistics_per_image_with_fault_injection(prev_pred_list,
                                                                                                     faulty_list)
        data.append(benign_count)
        data.append(faulty_avmis)
        data.append(faulty_non_avmis)

        prev_pred_list = []
        faulty_list = []
        for i in range(len(avmis_indexes)):
            while True:
                image_index = avmis_indexes[i]
                fi_taken_image_index = fi_taken[image_index]
                if fi_taken_image_index < 30:
                    break
            fi_data = fi_results[image_index][fi_taken[image_index]]
            faulty_list.append(fi_data[1])
            prev_pred_list.append(fi_data[0])
            fi_taken[image_index] += 1

        benign_count, faulty_avmis, faulty_non_avmis = get_statistics_per_image_with_fault_injection(prev_pred_list,
                                                                                                     faulty_list)

        data.append(benign_count)
        data.append(faulty_avmis)
        data.append(faulty_non_avmis)

        prev_pred_list = []
        faulty_list = []
        for i in range(len(non_avmis_indexes)):
            while True:
                image_index = non_avmis_indexes[i]
                fi_taken_image_index = fi_taken[image_index]
                if fi_taken_image_index < 30:
                    break
            fi_data = fi_results[image_index][fi_taken[image_index]]
            faulty_list.append(fi_data[1])
            prev_pred_list.append(fi_data[0])
            fi_taken[image_index] += 1

        benign_count, faulty_avmis, faulty_non_avmis = get_statistics_per_image_with_fault_injection(prev_pred_list,
                                                                                                     faulty_list)
        data.append(benign_count)
        data.append(faulty_avmis)
        data.append(faulty_non_avmis)

        misclassified_indices = []
        misclassified_indices.extend(avmis_indexes)
        misclassified_indices.extend(non_avmis_indexes)
        misclassified_indices.sort()
        prev_pred_list = []
        faulty_list = []
        for i in range(len(misclassified_indices)):
            while True:
                image_index = misclassified_indices[i]
                fi_taken_image_index = fi_taken[image_index]
                if fi_taken_image_index < 30:
                    break
            fi_data = fi_results[image_index][fi_taken[image_index]]
            faulty_list.append(fi_data[1])
            prev_pred_list.append(fi_data[0])
            fi_taken[image_index] += 1

        benign_count, faulty_avmis, faulty_non_avmis = get_statistics_per_image_with_fault_injection(prev_pred_list,
                                                                                                     faulty_list)
        data.append(benign_count)
        data.append(faulty_avmis)
        data.append(faulty_non_avmis)

        data_list.append(data)

    print(model_list)
    print(data_list)



if __name__ == '__main__':
    main()
