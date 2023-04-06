import torch
import pandas as pd

from data.dataset import DyDataset
from models.progressive_sage import GraphSage
from utils.get_params import get_args
from utils.device import get_device
from utils.set_seed import set_random_seed

import warnings
import os

warnings.filterwarnings('ignore')

args = get_args()
device = get_device(args.cuda_enable)
set_random_seed(args.seed)

dataset = DyDataset(dataset_name=args.dataset_name, edge_type=args.edge_type, m_size=args.m_size)  # memory_size=args.m_size
in_dim = dataset.in_dim
out_dim = dataset.labels_num

model = GraphSage(in_dim=in_dim, out_dim=out_dim, multi_threading=args.multi_threading,normalize=args.normalize)


def calculate_test_acc(graph):
    model.eval()
    y = graph.y
    out = model.inference(graph, "test")
    test_mask = graph.test_mask
    _, pred = out.max(dim=1)
    if test_mask.sum().item() > 0:
        correct = int(pred[test_mask].eq(y[test_mask]).sum().item())
        acc = correct / len(y[test_mask])
        return acc
    else:
        return 0.


def calculate_new_nodes_test_acc(graph):
    model.eval()
    y = graph.y
    out = model.inference(graph, "test")
    test_mask = graph.new_nodes_test_mask
    _, pred = out.max(dim=1)
    if test_mask.sum().item() > 0:
        correct = int(pred[test_mask].eq(y[test_mask]).sum().item())
        acc = correct / len(y[test_mask])
        return acc
    else:
        return 0.


def calculate_PM():
    aa = sum(acc_list) / len(acc_list)
    return aa


def calculate_FM():
    tmp_list = [final_acc_list[i] - acc_list[i] for i in range(len(acc_list))]
    FM = sum(tmp_list)/(len(tmp_list)-1)
    return FM


acc_list = []
final_acc_list = []
average_time_list = []

for id in range(len(dataset)):
    (rec_graph, ret_graph, now_graph) = dataset[id]
    print(f'task id is {id}')
    if id == 0:
        print("-"*10+"in init epochs"+"-"*10)
        training_time = 0
        for epoch in range(args.init_epochs):
            loss, time = model.init_train(rec_graph, phase='init', args=args)
            training_time += time
            if epoch % 10 == 0:
                print(f"epoch:\t{epoch} and loss:\t{loss}")

        acc = calculate_test_acc(rec_graph)
        print('Accuracy: {:.4f}'.format(acc))

        acc_list.append(acc)
        average_time_list.append(training_time/args.init_epochs)

        save_model_path = os.path.join(args.save_models_path, args.dataset_name)
        torch.save(model, os.path.join(save_model_path, f'{id}.pt'))
    else:
        model.open_parameters()
        rectify_time = 0
        for rectify_epoch in range(args.rectify_epochs):
            loss, time = model.rectify(rectify_graph=rec_graph, phase='rectify', epoch=rectify_epoch, args=args)
            rectify_time += time
            if loss == 0:
                break

        average_rectify_time = rectify_time / args.rectify_epochs
        model.expand()
        model.init_expanded_parameters()
        model.isolate_parameters()

        print("-" * 20 + "in retrain epochs" + "-" * 20)

        retrain_time = 0
        for retrain_epoch in range(args.retrain_epochs):
            loss, time = model.retrain(retrain_graph=ret_graph, phase='retrain', epoch=retrain_epoch, args=args)
            retrain_time += time
            if loss == 0:
                break

        average_retrain_time = retrain_time/args.retrain_epochs
        print("-" * 20 + "in retrain epochs" + "-" * 20)
        model.combine()

        acc = calculate_test_acc(now_graph)
        print('Accuracy: {:.4f}'.format(acc))

        acc_list.append(acc)
        average_time_list.append((rectify_time + retrain_time)/(args.rectify_epochs + args.retrain_epochs))

        save_model_path = os.path.join(args.save_models_path, args.dataset_name)
        torch.save(model, os.path.join(save_model_path, f'{id}.pt'))

for id in range(len(dataset)):
    (rec_graph, ret_graph, now_graph) = dataset[id]
    if id == 0:
        final_acc = calculate_test_acc(rec_graph)
        final_acc_list.append(final_acc)
    else:
        final_acc = calculate_test_acc(now_graph)
        task_acc = calculate_new_nodes_test_acc(now_graph)
        final_acc_list.append(final_acc)

PM = calculate_PM()
FM = calculate_FM()

print(f'average training time is: {sum(average_time_list)/len(average_time_list)}')
print("PM is:")
print(PM)
print("FM is:")
print(FM)
print("acc is:")
print(acc_list)


def evaluate(t, model_path, graph):
    model = torch.load(os.path.join(model_path, f'{t}.pt'))
    model.eval()
    y = graph.y
    out = model.inference(graph, "test")
    test_mask = graph.test_mask
    _, pred = out.max(dim=1)
    if test_mask.sum().item() > 0:
        correct = int(pred[test_mask].eq(y[test_mask]).sum().item())
        acc = correct / len(y[test_mask])
        return acc
    else:
        return 0.


print("begin evaluate forgetting!")
print("#"*20)
dataset = DyDataset(dataset_name=args.dataset_name, edge_type=args.edge_type, m_size=args.m_size)
data,_,_ = dataset[0]
first_task_acc_list = []
model_path = os.path.join(args.save_models_path, args.dataset_name)
print(len(dataset))
for i in range(len(dataset)):
    first_task_acc = evaluate(i, model_path, data)
    first_task_acc_list.append(first_task_acc)
print("Accuracy on the first task:")
print(first_task_acc_list)