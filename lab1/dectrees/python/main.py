import monkdata as mk
import dtree as dt
import drawtree_qt5 as dq
import random
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

def assignment1():
    print('Assignment 1:')

    print(dt.entropy(mk.monk1))
    print(dt.entropy(mk.monk2))
    print(dt.entropy(mk.monk3))

assignment1()

def assignment3():
    print('Assignment 3:')
    # for i in range(0,6):
    #     print("monk1 attribute", i + 1 ,":", dt.averageGain(mk.monk1, mk.attributes[i]))
    #     print("monk2 attribute", i + 1 ,":", dt.averageGain(mk.monk2, mk.attributes[i]))
    #     print("monk3 attribute", i + 1 ,":", dt.averageGain(mk.monk3, mk.attributes[i]))
    for a in mk.attributes:
        print("monk1 attribute", a ,":", dt.averageGain(mk.monk1, a))
        print("monk2 attribute", a ,":", dt.averageGain(mk.monk2, a))
        print("monk3 attribute", a ,":", dt.averageGain(mk.monk3, a))
        
    # MONK1: A5
    datasets = [
        dt.select(mk.monk1, mk.attributes[4], 1),
        dt.select(mk.monk1, mk.attributes[4], 2),
        dt.select(mk.monk1, mk.attributes[4], 3),
        dt.select(mk.monk1, mk.attributes[4], 4)
    ]

    attributes_remain = [a for a in mk.attributes if a != mk.attributes[4]]

    for j, subset in enumerate(datasets, start=1):
        print(f"Subset {j} (A5={j}):")
        for i, attr in enumerate(mk.attributes):
            gain = dt.averageGain(subset, attr)
            print(f"  A{i+1}: {gain:.4f}")
        print()
    
    rows = []
    for j, subset in enumerate(datasets, start=1):
        row = [f"A5={j}"]
        for attr in mk.attributes:
            row.append(round(dt.averageGain(subset, attr), 4))
        rows.append(row)
    headers = ["Subset"] + [f"A{i+1}" for i in range(len(mk.attributes))]
    # print table
    print(tabulate(rows, headers=headers, tablefmt="grid"))
    # For A5 == 1: The dataset is pure, no need to test any attributes
    # dataset 1 is pure
    print("--- A5 == 1:")
    print(dt.mostCommon(datasets[0]))

    # datasets[1]: attribute 4
    print("--- A5 == 2:")
    attr = mk.attributes[3]
    print("{} == 1:".format(attr))
    print(dt.mostCommon(dt.select(datasets[1], attr, 1)))
    print("{} == 2:".format(attr))
    print(dt.mostCommon(dt.select(datasets[1], attr, 2)))
    print("{} == 3:".format(attr))
    print(dt.mostCommon(dt.select(datasets[1], attr, 3)))
    
    # datasets[2]: attribute 6
    print("--- A5 == 3:")
    attr = mk.attributes[5]
    print("{} == 1:".format(attr))
    print(dt.mostCommon(dt.select(datasets[2], attr, 1)))
    print("{} == 2:".format(attr))
    print(dt.mostCommon(dt.select(datasets[2], attr, 2)))

    # datasets[3]: attribute 1
    print("--- A5 == 4:")
    attr = mk.attributes[0]
    print("{} == 1:".format(attr))
    print(dt.mostCommon(dt.select(datasets[3], attr, 1)))
    print("{} == 2:".format(attr))
    print(dt.mostCommon(dt.select(datasets[3], attr, 2)))
    print("{} == 3:".format(attr))
    print(dt.mostCommon(dt.select(datasets[3], attr, 3)))
    print()
    # t1 = dt.buildTree(mk.monk1, mk.attributes)
    # dq.drawTree(t1)
    # t1 = dt.buildTree(mk.monk1, mk.attributes,maxdepth=3)
    # dq.drawTree(t1)
    # t2 = dt.buildTree(mk.monk2, mk.attributes)
    # dq.drawTree(t2)
    # t3 = dt.buildTree(mk.monk3, mk.attributes)
    # dq.drawTree(t3)
    # For A5 == 2: attribute A4 should be tested
    # For A5 == 3: attribute A6 should be tested
    # For A5 == 4: attribute A1 should be tested
    # MONK2: A5
    # MONK3: A2
assignment3()

def assignment5():
    print('Assignment 5:')

    # Build full trees
    t1 = dt.buildTree(mk.monk1, mk.attributes)
    t2 = dt.buildTree(mk.monk2, mk.attributes)
    t3 = dt.buildTree(mk.monk3, mk.attributes)

    # dq.(t1)
    # dq.drawTree(t2)
    # dq.drawTree(t3)

    # Compute accuracies
    acc1_train = dt.check(t1, mk.monk1)
    acc1_test  = dt.check(t1, mk.monk1test)

    acc2_train = dt.check(t2, mk.monk2)
    acc2_test  = dt.check(t2, mk.monk2test)

    acc3_train = dt.check(t3, mk.monk3)
    acc3_test  = dt.check(t3, mk.monk3test)

    # Print as errors (1 - accuracy)
    print("MONK-1: E_train =", 1 - acc1_train, ", E_test =", 1 - acc1_test)
    print("MONK-2: E_train =", 1 - acc2_train, ", E_test =", 1 - acc2_test)
    print("MONK-3: E_train =", 1 - acc3_train, ", E_test =", 1 - acc3_test)
    print()
assignment5()

def partition(data, fraction):
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]

def pruned_error(data, test):
    fractions = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    pruned_trees_error = []

    for fraction in fractions:
        train, val = partition(data, fraction)
        tree = dt.buildTree(train, mk.attributes)
        ptrees = dt.allPruned(tree)
        
        best_acc = dt.check(tree, val)
        best_tree = tree

        for ptree in ptrees:
            temp_acc = dt.check(ptree, val)
            if best_acc < temp_acc:
                best_acc = temp_acc
                best_tree = ptree
        
        best_performance = dt.check(best_tree,test)
        pruned_trees_error.append(round(1 - best_performance,5))
    return pruned_trees_error
def assignment7():
    
    epoch = 500
    fractions = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    monk1_pruned =  np.transpose([pruned_error(mk.monk1, mk.monk1test) for i in range(epoch)])
    monk3_pruned =  np.transpose([pruned_error(mk.monk3, mk.monk3test) for i in range(epoch)])

    mean1 = np.around(np.mean(monk1_pruned, axis=1), decimals=5)
    mean3 = np.around(np.mean(monk3_pruned, axis=1), decimals=5)
    std1 = np.around(np.std(monk1_pruned, axis=1), decimals=5)
    std3 = np.around(np.std(monk3_pruned, axis=1), decimals=5)

    # Print results
    print("Assignment 7:")
    print("Fractions:", fractions)
    print("Total Runs:", epoch)
    print("MONK1 mean:", np.round(mean1, 5))
    print("MONK1 std :", np.round(std1, 5))
    print("MONK3 mean:", np.round(mean3, 5))
    print("MONK3 std :", np.round(std3, 5))


    plt.figure()
    plt.title('Mean of Errors - {} runs'.format(epoch))
    plt.xlabel('Fractions')
    plt.ylabel('Mean Value')
    plt.plot(fractions, mean1, label="MONK-1", marker='o')
    plt.plot(fractions, mean3, label="MONK-3", marker='o')
    plt.legend()
    plt.show()    
    
    plt.title('Standard Deviation of Errors - {} runs'.format(epoch))
    plt.xlabel('Fractions')
    plt.ylabel('Standard Deviation Value')
    plt.plot(fractions, std1, label="MONK-1", marker='o')
    plt.plot(fractions, std3, label="MONK-3", marker='o')
    plt.legend()
    plt.show()    

assignment7()
    
