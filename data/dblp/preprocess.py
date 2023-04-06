import os
import sys
import random
import numpy as np 

random.seed(1)
np.random.seed(1)   

def process_edges():
    in_file = open('meta/edges.txt')
    out_file = open('edges', 'w')
    new_flag = True
    
    for line in in_file.readlines():
        items = line.strip().split()
        if new_flag:
            new_flag = False
        elif len(items) < 2:
            new_flag = True
        else:
            out_file.write(line)
    
    in_file.close()
    out_file.close()

# process_edges()


def process_labels():
    in_file = open('meta/labels.txt')
    out_file = open('labels', 'w')
    
    for line in in_file.readlines():
        items = line.strip().split()
        if len(items) < 2:
            pass
        elif int(items[1]) == 10:
            continue
        else:
            out_file.write(items[0] + '\t' + str(int(items[1]) - 1) + '\n')
    
    in_file.close()
    out_file.close()

# process_labels()



def generate_labeled_nodes():
    ori_file_name = os.path.join('labels')
    nodes = np.loadtxt(ori_file_name, dtype = np.int64)[:, 0]

    train_file = open('train_nodes', 'w')
    val_file = open('val_nodes', 'w')
    test_file = open('test_nodes', 'w')
    for node in nodes:
        r = random.random()
        if r < 0.6:
            train_file.write(str(node) + '\n')
        elif r < 0.8:
            val_file.write(str(node) + '\n')
        else:
            test_file.write(str(node) + '\n')
    train_file.close()
    val_file.close()
    test_file.close()

generate_labeled_nodes()


def process_dyn_edges():
    for i, year in enumerate(range(1991, 2017)):
        in_file = open('meta/edges' + str(year) + '.txt')
        out_file = open(os.path.join('stream_edges_26', str(i)), 'w')
        new_flag = True
        
        for line in in_file.readlines():
            items = line.strip().split()
            if new_flag:
                new_flag = False
            elif len(items) < 2:
                new_flag = True
            else:
                out_file.write(line)
        
        in_file.close()
        out_file.close()

# process_dyn_edges()


def check_label_distribution():
    in_file = open('meta/labels.txt')
    node2label = dict()
    for line in in_file.readlines():
        items = line.strip().split()
        if len(items) < 2:
            pass
        else:
            node2label[items[0]] = int(items[1]) - 1
    in_file.close()

    time_label_cnt = np.zeros((26, 10), dtype=np.int32)
    for i, year in enumerate(range(1991, 2017)):
        in_file = open('meta/edges' + str(year) + '.txt')
        new_flag = True
        
        for line in in_file.readlines():
            items = line.strip().split()
            if new_flag:
                new_flag = False
            elif len(items) < 2:
                new_flag = True
            else:
                if items[0] in node2label:
                    time_label_cnt[i, node2label[items[0]]] += 1
                if items[1] in node2label:
                    time_label_cnt[i, node2label[items[1]]] += 1
        
        in_file.close()
    
    print(time_label_cnt)
    print(np.divide(time_label_cnt, np.sum(time_label_cnt, 1)[:, None]))
    print(np.sum(time_label_cnt, 1))

# check_label_distribution()


def process_dyn_edges_rev():
    for i, year in enumerate(range(2016, 1990, -1)):
        in_file = open('meta/edges' + str(year) + '.txt')
        out_file = open(os.path.join('stream_edges_26_rev', str(i)), 'w')
        new_flag = True
        
        for line in in_file.readlines():
            items = line.strip().split()
            if new_flag:
                new_flag = False
            elif len(items) < 2:
                new_flag = True
            else:
                out_file.write(line)
        
        in_file.close()
        out_file.close()

# process_dyn_edges_rev()

