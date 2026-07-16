import numpy as np
import pandas as pd
from itertools import combinations
np.random.seed(123)

class DecisionTreeClassifier:
    def __init__(self, 
                 max_depth, 
                 max_features, 
                 min_info_gain,
                 min_samples_split):
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_info_gain = min_info_gain
        self.min_samples_split = min_samples_split
        self.tree = None

    def mse(self, y):
        # Ensure `y` is a 1-D array/Series of labels (not a DataFrame)
        # if isinstance(y, (pd.DataFrame, pd.Series)):
        #     # squeeze to 1-D
        #     y = y.values
        y = np.asarray(y).ravel()
        return np.var(y)
    
    def mse_split(self, left, right):
        n_left = left.shape[0]
        n_right = right.shape[0]
        n = n_left + n_right
        if n==0: return 0
        mse_left = self.mse(left)
        mse_right = self.mse(right)
        return ((n_left / n) * mse_left) + ((n_right / n) * mse_right)

    # Run this for each feature
    def find_best_split(self, df, column, y):
        values = np.unique(df[column])
        if len(values) <= 1:
            return None, float('inf')

        if np.issubdtype(df[column].dtype, np.number):
            # continuous feature
            _, bins = np.histogram(df[column])
            candidate_thresholds = [(bins[i-1]+bins[i])/2 for i in range(1, len(bins))]
            gini_vals = {}
            for thresh in candidate_thresholds:
                left = y[df[column] < thresh]
                right = y[df[column] >= thresh]
                gini_vals[thresh] = self.mse_split(left, right)

            optimal_thresh = min(gini_vals, key=gini_vals.get)
            return optimal_thresh, gini_vals[optimal_thresh]
        else:
            # categorical feature
            candidate_thresholds = []
            # for i in range(1, len(values)):
            #     candidate_thresholds += list(combinations(values, i))
            for i in range(1, (len(values) // 2) + 1):
                for comb in combinations(values, i):
                    # To avoid mirroring when splitting an even number of items exactly in half:
                    # e.g., for [A, B, C, D], forcing 'A' to always be in the left chunk prevents ([A, B] vs [C, D]) 
                    # from repeating later as ([C, D] vs [A, B]).
                    if i == len(values) / 2 and comb[0] != values[0]:
                        continue
                    candidate_thresholds.append(comb)
            gini_vals = {}
            for thresh in candidate_thresholds:
                # left = y[df[column].isin(thresh)]
                # indices = range(len(y))
                is_left = df[column].isin(thresh).values
                left = y[is_left]
                right = y[~is_left]
                thresh_str = ", ".join(thresh)
                gini_vals[thresh_str] = self.mse_split(left, right)
            
            optimal_thresh = min(gini_vals, key=gini_vals.get)
            return optimal_thresh, gini_vals[optimal_thresh]
    
    def find_best_attribute(self, df, y):
        cnames = df.columns
        att_mse = {}
        att_threshold = {}
        for cname in cnames:
            optimal_thresh, mse_val = self.find_best_split(df, cname, y)
            att_mse[cname] = mse_val
            att_threshold[cname] = optimal_thresh
        best_attribute = min(att_mse, key=att_mse.get)
        best_threshold = att_threshold[best_attribute]
        mse_val = att_mse[best_attribute]
        return best_attribute, best_threshold, mse_val

    def build_tree(self, df, y, depth=0):

        n_labels = len(np.unique(y))
        n_samples = df.shape[0]
        parent_mse = self.mse(y)

        target_value = np.mean(y.values.ravel())

        # 1. Stopping criteria
        if (depth >= self.max_depth or 
            n_labels == 1 or 
            n_samples < self.min_samples_split):
           
           return {
            "is_leaf": True,
            "value": target_value ,
            "left": None,
            "right": None,
            "feature": None,
            "is_feature_numerical": None,
            "threshold": None,
            "mse": parent_mse,
            "samples": n_samples
           }

        # 2. Subsample features
        max_features = self.max_features
        if self.max_features == "sqrt":
            max_features = int(np.sqrt(df.shape[1]))

        max_features = max(1, max_features) # To handle 0 case
        sub_feats = np.random.choice(df.columns, max_features, replace=False)
        df_sub = df.loc[:, sub_feats]
        
        # 3. Select the best attribute for split
        best_att, best_thresh, best_att_mse = self.find_best_attribute(df_sub, y)

        # 4. If no attribute can be picked as the best, then default to the leaf node
        info_gain = parent_mse - best_att_mse
        if best_att is None or info_gain < self.min_info_gain:
            return {
                "is_leaf": True,
                "value": target_value,
                "left": None,
                "right": None,
                "feature": None,
                "is_feature_numerical": None,
                "threshold": None,
                "gini": parent_gini,
                "samples": n_samples
            }
        
        # 5. Partition the data according to the best_thresh for branches
        # further down. 

        # 5.a. First find out whether the column is numerical or categorical
        left_df = None
        right_df = None
        if np.issubdtype(df[best_att].dtype, np.number):
            # If numerical
            # left_indices = df.loc[df_sub[best_att] < best_thresh, :]
            left_boolean_indices = df[best_att] < best_thresh
        else:
            # If categorical
            best_thresh = best_thresh.split(", ")
            left_boolean_indices = df[best_att].isin(best_thresh).values
        right_boolean_indices = ~left_boolean_indices
        
        # 5.b. Partition the data into left and right branches
        left_df = df.loc[left_boolean_indices, :]
        right_df = df.loc[right_boolean_indices, :]
        left_y = y[left_boolean_indices]
        right_y = y[right_boolean_indices]
        
        left_child = self.build_tree(left_df, left_y, depth=depth+1)
        right_child = self.build_tree(right_df, right_y, depth=depth+1)

        return {
            "is_leaf": False,
            "value": None,
            "feature": best_att,
            "is_feature_numerical": np.issubdtype(df[best_att].dtype, np.number),
            "threshold": best_thresh,
            "gini": best_att_mse,
            "samples": n_samples,
            "left": left_child,
            "right": right_child
        }

    def fit(self, df, y):
        # Ensure `y` is a 1-D array-like (not a single-column DataFrame)
        # if isinstance(y, pd.DataFrame):
        #     if y.shape[1] == 1:
        #         y = y.iloc[:, 0].values
        #     else:
        #         y = y.values
        # elif isinstance(y, pd.Series):
        #     y = y.values

        self.tree = self.build_tree(df, y)
        
    def traverse_tree(self, df, tree):
        # df is assumed to contain a single instance
        if tree['is_leaf'] is True:
            return tree['value']
        
        feature = tree['feature']
        threshold = tree['threshold']
        is_feat_numerical = tree['is_feature_numerical']
        branch = None
        value = df[feature]
        if is_feat_numerical:
            if value < threshold:
                branch = tree['left']
            else:
                branch = tree['right']
        else: # When categorical
            # threshold is a tuple
            if np.isin(value, threshold):
                branch = tree['left']
            else:
                branch = tree['right']
        return self.traverse_tree(df, branch)

    def predict(self, df):
        predicted = []
        for i in range(df.shape[0]):
            predicted.append(self.traverse_tree(df.iloc[i], self.tree))
        return predicted


if __name__ == "__main__":
    df_train = pd.read_csv("tree_models/processed_X_train.csv", header=0)
    df_val = pd.read_csv("tree_models/processed_X_val.csv", header=0)
    y_train = pd.read_csv("tree_models/y_train.csv")
    y_val = pd.read_csv("tree_models/y_val.csv")

    df_train['type'] = y_train.values.ravel()
    df_val['type'] = y_val.values.ravel()
    
    df_train = df_train.melt(id_vars=['bone_length', 'hair_length', 
                                      'has_soul', 'rotting_flesh',
                                      'type'], 
                            value_vars=['color_black', 'color_blood', 
                                        'color_blue', 'color_clear', 
                                        'color_green', 'color_white'], 
                            var_name='color', 
                            ignore_index=True)
    df_train['color'] = df_train.color.apply(lambda x: x.split("_")[1])
    df_train = df_train.loc[df_train.value == True, :]
    y_train = df_train['type']
    df_train = df_train.drop(columns=['value', 'type'])
    df_train = df_train.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)

    df_val = df_val.melt(id_vars=['bone_length', 'hair_length', 
                                    'has_soul', 'rotting_flesh',
                                    'type'], 
                         value_vars=['color_black', 'color_blood', 
                                     'color_blue', 'color_clear', 
                                     'color_green', 'color_white'], 
                         var_name='color', 
                         ignore_index=True)
    df_val['color'] = df_val.color.apply(lambda x: x.split("_")[1])
    df_val = df_val.loc[df_val.value == True, :]
    y_val = df_val['type']
    df_val = df_val.drop(columns=['value', 'type'])
    df_val = df_val.reset_index(drop=True)
    y_val = y_val.reset_index(drop=True)

    max_features = df_train.shape[1]
    dt_model = DecisionTreeClassifier(max_depth=20, 
                                      max_features="sqrt",
                                      min_info_gain = 1e-6,
                                      min_samples_split = 20
                                      )
    dt_model.fit(df_train, y_train)
    predictions = dt_model.predict(df_val)
    print("checkpoint...")