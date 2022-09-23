"""--------------------------------------
Name: Rajkumar Conjeevaram Mohan
Email: rajkumarcm@yahoo.com
BSc Hons (UK), MSc AI (UK), MS DS (US)
Decision Tree
--------------------------------------"""
import numpy as np

class Node:
    def __init__(self, label, measure, measure_type):
        self.label = label
        self.measure = measure
        self.measure_type = measure_type
        self.children = {}

    def add_child(self, label, node):
        self.children[label] = node

class DecisionTreeClassifier:

    def __entropy_of_an_attribute(self, attribute, y_subset):
        values = np.unique(attribute)
        entropy_value = 0
        for value in values:
            indices = np.where(attribute == value)
            attribute_value = attribute[indices]
            attribute_target = y_subset[indices]
            number_of_value_v_in_attribute = len(attribute_value)
            prior_prob = number_of_value_v_in_attribute / len(attribute)

            # Entropy of each value in the attribute
            # the value that results from the loop is for instance E(D_sunny)
            for class_label in np.unique(y_subset):
                # tmp_n refers to the number of class i
                tmp_n = np.sum(attribute_target == class_label)
                tmp_prob = tmp_n / number_of_value_v_in_attribute
                entropy_value += -(tmp_prob * np.log2(tmp_prob))

            entropy_value += prior_prob * entropy_value
        return entropy_value

    def __entropy_of_a_dataset(self, y_subset):
        n = len(y_subset)
        entropy_val = 0
        for class_label in np.unique(y_subset):
            n_class_i_in_target = np.sum(y_subset == class_label)
            prob = n_class_i_in_target/n
            entropy_val += -(prob * np.log2(prob))
        return entropy_val

    def entropy(self, x_subset, y_subset):
        class_labels = np.unique(y_subset)
        n_classes = len(class_labels)
        entropy_value = 0
        entropy_value = self.__entropy_of_an_attribute(x_subset, y_subset)
        # You can enable this when you have access to labels. So for instance if you are using Pandas that has column
        # names then the following snippet of code can be enabled under which case a dictionary of entropies with
        # key as labels can be returned...
        # else:
        #     for i in range(x_subset.shape[1]):
        #         entropy_value += self.__entropy_of_an_attribute(x_subset[:, i], y_subset)
        entropy_dataset = self.__entropy_of_a_dataset(y_subset)
        ig = entropy_dataset - entropy_value

    def create_decision_tree(self, X, y):


