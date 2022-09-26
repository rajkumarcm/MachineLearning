"""--------------------------------------
Name: Rajkumar Conjeevaram Mohan
Email: rajkumarcm@yahoo.com
BSc Hons (UK), MSc AI (UK), MS DS (US)
Decision Tree
--------------------------------------"""
import numpy as np
import pandas as pd

class Node:
    def __init__(self, label, n_samples, info_gain=None, gini=None):
        self.label = label
        self.info_gain = info_gain
        self.gini = gini
        self.n_samples = n_samples
        self.children = {}

    def add_child(self, label, node):
        self.children[label] = node

class DecisionTreeClassifier:

    def entropy_of_an_attribute(self, attribute, y_subset):
        values = np.unique(attribute)
        entropy_value = 0
        for value in values:
            indices = np.where(attribute == value)[0]
            attribute_target = y_subset[indices]
            number_of_value_v_in_attribute = len(indices)
            prior_prob = number_of_value_v_in_attribute / len(attribute)

            # Entropy of each value in the attribute
            # the value that results from the loop is for instance E(D_sunny)
            for class_label in y_subset.unique():
                number_of_class_i_in_value_v = np.sum(attribute_target == class_label)
                tmp_prob = number_of_class_i_in_value_v / number_of_value_v_in_attribute
                entropy_value -= (tmp_prob * np.log2(tmp_prob))
            entropy_value += prior_prob * entropy_value
        return entropy_value

    def __entropy_of_a_dataset(self, y_subset):
        n = len(y_subset)
        entropy_val = 0
        for class_label in y_subset.unique():
            n_class_i_in_target = np.sum(y_subset == class_label)
            prob = n_class_i_in_target/n
            entropy_val -= (prob * np.log2(prob))
        return entropy_val

    def information_gain(self, x_subset, y_subset):
        class_labels = np.unique(y_subset)
        # n_classes = len(class_labels)
        info_gain = {}
        entropy_dataset = self.__entropy_of_a_dataset(y_subset)
        for col in x_subset.columns:
            info_gain[col] = entropy_dataset - self.entropy_of_an_attribute(x_subset.loc[:, col], y_subset)
        return info_gain

    def create_decision_tree(self, X, y, n_total_attributes, class_label, mode_of_root_node):
        leaf_class = None
        node = None
        n_samples = X.shape[0]

        # When there is only one attribute, then there is no point in passing this onto a recursive
        # function to decide the best attribute. This is the only attribute we have got.
        if len(X.columns) == 1 or X.shape[0]:
            leaf_class = y.mode().max()
            # Leaf node
            node = Node(label=class_label, n_samples=n_samples)
            child = Node(label=leaf_class, n_samples=n_samples)
            node.add_child(label=mode_of_root_node, node=child)

        if n_total_attributes == X.shape[1]:
            # Since the n_total_attributes value doesn't saturate over the iterations, we can
            # use this to verify whether this is the initial iteration or a subsequent by
            # comparing the remaining features of the subset against the total number of attributes
            # the dataset originally has.
            # Root node
            node = Node(label=class_label, n_samples=n_samples)

        # Time to recursively partition the data and create the rest of the tree
        info_gain = self.information_gain(X, y)
        best_attribute = max(info_gain, key=info_gain.get)
        max_ig = info_gain[best_attribute]

        # Partition the data as per the values of the attribute
        for val_of_attribute in X.loc[:, best_attribute].unique(): # This is where ID3 and C4.5 differs
            indices = X.loc[:, best_attribute] == val_of_attribute
            X_subset = X.loc[indices,].drop(columns=best_attribute)
            y_subset = y.loc[indices,]

            # Create the descendant nodes
            child = self.create_decision_tree(X=X_subset, y=y_subset, n_total_attributes=n_total_attributes,
                                              class_label=best_attribute, mode_of_root_node=y.mode().max())
            child.information_gain = max_ig
            node.children[val_of_attribute] = child
            return node


if __name__ == '__main__':

    df = pd.read_csv('car.data', header=None, names=['buying', 'class_', 'maint', 'doors', 'persons', 'lug_boot', 'safety'],
                     skip_blank_lines=True, on_bad_lines='error')
    dt = DecisionTreeClassifier()
    y = df.buying
    X = df.drop(columns=['buying'])
    tree = dt.create_decision_tree(X=X, y=y, n_total_attributes=len(df.columns),
                                   class_label='buying', mode_of_root_node=y.mode().max())
    print('debug breakpoint...')








