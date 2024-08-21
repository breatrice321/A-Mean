def main():
    model_names = ['ResNet50']
    for model_name in model_names:
        partitions = 16
        data_per_partition = 500
        data_list = []
        data_list.append("Model name: " + model_name + "\n")
        lines_list = []
        for i in range(partitions):
            file1 = open('../dataset/stl-10/result/' + model_name + '/' + 'stl-10_final_log_' + model_name + '_' + str(i) + '.txt', 'r')
            lines = file1.readlines()
            lines.pop(0)
            lines.pop(0)
            data_list.extend(lines[:data_per_partition])
            lines_list.append(lines[data_per_partition:])

        for lines in lines_list:
            data_list.extend(lines)

        file_object = open('../dataset/stl-10/stl-10_final_log_' + model_name + '.txt', 'a')
        for data in data_list:
            file_object.write(data)
        file_object.close()


if __name__ == '__main__':
    main()
