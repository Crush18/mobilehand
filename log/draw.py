import matplotlib.pyplot as plt

total_train_loss = []
total_test_loss = []
total_test_auc = []

with open('train_log.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        if line.startswith("训练次数"):
            train_loss = line.split(',')[1].split(':')[1]
            total_train_loss.append(train_loss)
        elif line.startswith("整体测试集上Loss"):
            test_loss = line.split(':')[1]
            total_test_loss.append(test_loss)
        elif line.startswith("整体测试集上AUC"):
            test_auc = line.split(':')[1]
            total_test_auc.append(test_auc)
print(len(total_train_loss))
print(len(total_test_loss))
print(len(total_test_auc))
