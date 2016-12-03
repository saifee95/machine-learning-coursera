#Authors
#Mohammad Saifee (201301146)
#Shivang Agarwal (201301148)
#Kundan Arya (201301196)

# ## Implementing binary decision trees

# The goal of this project is to implement our own binary decision tree classifier. We will:
#     
# * Use SFrames to do some feature engineering.
# * Transform categorical variables into binary variables.
# * Write a function to compute the number of misclassified examples in an intermediate node.
# * Write a function to find the best feature to split on.
# * Build a binary decision tree from scratch.
# * Make predictions using the decision tree.
# * Evaluate the accuracy of the decision tree.
# * Visualize the decision at the root node.

import graphlab
import math as mt

# Load the lending club dataset
print "Loading the data"
loans = graphlab.SFrame('lending-club-data.gl/')


# We reassign the labels to have +1 for a safe loan, and -1 for a risky (bad) loan.

loans['safe_loans'] = loans['bad_loans'].apply(lambda x : +1 if x==0 else -1)
loans = loans.remove_column('bad_loans')

# Instead of the entire feature set, we will just be using 4 categorical
# features: 
# 
# 1. grade of the loan 
# 2. the length of the loan term
# 3. the home ownership status: own, mortgage, rent
# 4. number of years of employment.
# 
# Since we are building a binary decision tree, we will have to convert these categorical features to a binary representation.

print "Extracting the features"
features = ['grade',              # grade of the loan
            'term',               # the term of the loan
            'home_ownership',     # home_ownership status: own, mortgage or rent
            'emp_length',         # number of years of employment
           ]
target = 'safe_loans'
loans = loans[features + [target]]

# Let's explore what the dataset looks like.

loans.print_rows(10,5)

# Subsample dataset to make sure classes are balanced
# We will undersample the larger class (safe loans) in order to balance out our dataset.
print "Subsampling the data"

safe_loans_raw = loans[loans[target] == 1]
risky_loans_raw = loans[loans[target] == -1]

print "Percentage of safe loans                 :", len(safe_loans_raw) / float(len(loans))
print "Percentage of risky loans                :", len(risky_loans_raw) / float(len(loans))

print "After Subsampling\n"
# Since there are less risky loans than safe loans, we find the ratio of the sizes
# and use that percentage to undersample the safe loans.

percentage = len(risky_loans_raw)/float(len(safe_loans_raw))
safe_loans = safe_loans_raw.sample(percentage, seed = 1)
risky_loans = risky_loans_raw
loans_data = risky_loans.append(safe_loans)

print "Percentage of safe loans                 :", len(safe_loans) / float(len(loans_data))
print "Percentage of risky loans                :", len(risky_loans) / float(len(loans_data))
print "Total number of loans in our new dataset :", len(loans_data)


# Transforming categorical data into binary features

# Since all of our features are currently categorical features, we want to turn them into binary features.

print "Transforming categorical data into binary features"
loans_data = risky_loans.append(safe_loans)
for feature in features:
    loans_data_one_hot_encoded = loans_data[feature].apply(lambda x: {x: 1})    
    loans_data_unpacked = loans_data_one_hot_encoded.unpack(column_name_prefix=feature)
    
    # Change None's to 0's
    for column in loans_data_unpacked.column_names():
        loans_data_unpacked[column] = loans_data_unpacked[column].fillna(0)

    loans_data.remove_column(feature)
    loans_data.add_columns(loans_data_unpacked)


# Let's see what the feature columns look like now:

features = loans_data.column_names()
features.remove('safe_loans')
print "Number of features (after binarizing categorical variables) = %s" % len(features)

# Let's explore what one of these columns looks like:
print "Column grade.A"
print loans_data['grade.A']

# We split the data into a train test split with 80% of the data in the training set and 20% of the data in the test set.
print "Splitting the data into training and test sets"
train_data, test_data = loans_data.random_split(0.8, seed=1)

# Decision tree implementation
# Now, we have wrote the function `intermediate_node_num_mistakes` which computes
# the number of misclassified examples of an intermediate node given the set of labels (y values) of the data points contained in the node.

def intermediate_node_num_mistakes(labels_in_node):
    # Corner case: If labels_in_node is empty, return 0
    if len(labels_in_node) == 0:
        return 0
    
    # Count the number of 1's (safe loans)
    pos = len(labels_in_node[labels_in_node == 1])
    
    # Count the number of -1's (risky loans)
    neg = len(labels_in_node[labels_in_node == -1])
                
    # Return the number of mistakes that the majority classifier makes.
    if pos > neg :
        return neg
    return pos 
    
# Function to calculate entropy at an intermediate node.

def calculate_entropy(labels_node) :
    
    tot = float(len(labels_node))
    if len(labels_node) == 0:
        return 0
    p1 = (len(labels_node[labels_node == 1])/tot)
    p2 = (len(labels_node[labels_node == -1])/tot)
    
    if (p1 > 0) and (p2 > 0) :
        entropy = -1*(p1*mt.log(p1,2) + p2*mt.log(p2,2))
    elif (p1 <= 0) :
        entropy = -1*(p2*mt.log(p2,2))
    else :
        entropy = -1*(p1*mt.log(p1,2))
    
    return round(entropy,2)


# Testing the function 

# Test case 1
example_labels = graphlab.SArray([-1, -1, 1, 1, 1])
if (intermediate_node_num_mistakes(example_labels) == 2) and (calculate_entropy(example_labels) == 0.97) :
    print 'Test passed!'
else:
    print 'Test 1 failed... try again!'

# Test case 2
example_labels = graphlab.SArray([-1, -1, 1, 1, 1, 1, 1])
if (intermediate_node_num_mistakes(example_labels) == 2) and (calculate_entropy(example_labels) == 0.86):
    print 'Test passed!'
else:
    print 'Test 2 failed... try again!'
    
# Test case 3
example_labels = graphlab.SArray([-1, -1, -1, -1, -1, 1, 1])
if intermediate_node_num_mistakes(example_labels) == 2:
    print 'Test passed!'
else:
    print 'Test 3 failed... try again!'


# Function to pick best feature to split on

def best_splitting_feature(data, features, target):
    
    best_feature = None # Keep track of the best feature 
    best_error = 10     # Keep track of the best error so far 
    # Note: Since error is always <= 1, we should intialize it with something larger than 1.

    # Convert to float to make sure error gets computed correctly.
    num_data_points = float(len(data))  
    
    # Loop through each feature to consider splitting on that feature
    for feature in features:
        
        # The left split will have all data points where the feature value is 0
        left_split = data[data[feature] == 0]
        
        # The right split will have all data points where the feature value is 1
        right_split = data[data[feature] == 1] 
            
        # Calculate the number of misclassified examples in the left split.
        left_mistakes = intermediate_node_num_mistakes(left_split['safe_loans'])            

        # Calculate the number of misclassified examples in the right split.
        right_mistakes = intermediate_node_num_mistakes(right_split['safe_loans'])
            
        # Compute the  of this split.
        # Error = (# of mistakes (left) + # of mistakes (right)) / (# of data points)
        error = ((left_mistakes + right_mistakes)/num_data_points)

        # If this is the best error we have found so far, store the feature as best_feature and the error as best_error
        if error < best_error:
            best_error = error
            best_feature = feature
        
    
    return best_feature # Return the best feature we found


def best_splitting_feature_entropy(data, features, target):
    
    best_feature = None # Keep track of the best feature 
    best_entropy = 10     # Keep track of the best error so far 
    # Note: Since error is always <= 1, we should intialize it with something larger than 1.

    # Convert to float to make sure error gets computed correctly.
    num_data_points = float(len(data))  
    
    # Loop through each feature to consider splitting on that feature
    for feature in features:
        
        # The left split will have all data points where the feature value is 0
        left_split = data[data[feature] == 0]
        
        # The right split will have all data points where the feature value is 1
        right_split = data[data[feature] == 1] 
            
        # Calculate the entropy in the left split.
        left_entropy = calculate_entropy(left_split['safe_loans'])            

        # Calculate the entropy in the right split.
        right_entropy = calculate_entropy(right_split['safe_loans'])
            
        # calculate entropy of split
        entropy_split = (((len(left_split)/num_data_points)*left_entropy) + ((len(right_split)/num_data_points)*right_entropy))

        # If this is the best entropy we have found so far, store the feature as best_feature and the entropy as best_entropy
        if entropy_split < best_entropy:
            best_entropy = entropy_split
            best_feature = feature
        
    
    return best_feature # Return the best feature we found


# Testing the function

if best_splitting_feature(train_data, features, 'safe_loans') == 'term. 36 months':
    print 'Test passed!'
else:
    print 'Test failed... try again!'


# Building the tree
# 
# With the above functions implemented correctly, we are now ready to build our decision tree. Each node in the decision tree is represented as a dictionary which contains the following keys and possible values:
# 
#     { 
#        'is_leaf'            : True/False.
#        'prediction'         : Prediction at the leaf node.
#        'left'               : (dictionary corresponding to the left tree).
#        'right'              : (dictionary corresponding to the right tree).
#        'splitting_feature'  : The feature that this node splits on.
#     }
# 

# Function to create leaf node

def create_leaf(target_values):
    
    # Create a leaf node
    leaf = {'splitting_feature' : None,
            'left' : None,
            'right' : None,
            'is_leaf':True     }
    
    # Count the number of data points that are +1 and -1 in this node.
    num_ones = len(target_values[target_values == +1])
    num_minus_ones = len(target_values[target_values == -1])
    
    # For the leaf node, set the prediction to be the majority class.
    # Store the predicted class (1 or -1) in leaf['prediction']
    if num_ones > num_minus_ones:
        leaf['prediction'] = 1
    else:
        leaf['prediction'] = -1
        
    # Return the leaf node        
    return leaf 


# Function that learns the decision tree recursively and implements 5 stopping conditions:
# 1. Stopping condition 1: All data points in a node are from the same class.
# 2. Stopping condition 2: No more features to split on.
# 3. Stopping condition 3: max_depth of the tree.
# 4. Stopping condition 4: minimum node size.
# 5. Stopping condition 5: possible error reduction.

def reached_minimum_node_size(data, min_node_size):
    # Return True if the number of data points is less than or equal to the minimum node size.
    if len(data) <= min_node_size :
        return True
    return False


def error_reduction(error_before_split, error_after_split):
    # Return the error before the split minus the error after the split.
    return (error_before_split - error_after_split)

# Skeleton of the tree
def decision_tree_create(data, features, target, current_depth = 0, max_depth = 10, min_node_size = 1, min_error_reduction = 0.0):
    
    remaining_features = features[:] # Make a copy of the features.
    
    target_values = data[target]
    print "--------------------------------------------------------------------"
    print "Subtree, depth = %s (%s data points)." % (current_depth, len(target_values))
    
    
    # Stopping condition 1: All nodes are of the same type.
    if intermediate_node_num_mistakes(target_values) == 0:
        print "Stopping condition 1 reached. All data points have the same target value."                
        return create_leaf(target_values)
    
    # Stopping condition 2: No more features to split on.
    if remaining_features == []:
        print "Stopping condition 2 reached. No remaining features."                
        return create_leaf(target_values)    
    
    # Stopping condition 3: Reached max depth limit.
    if current_depth >= max_depth:
        print "Early stopping condition 1 reached. Reached maximum depth."
        return create_leaf(target_values)
    
    # stopping condition 4: Reached the minimum node size.
    # If the number of data points is less than or equal to the minimum size, return a leaf.
    if reached_minimum_node_size(data,min_node_size) : 
        print "Early stopping condition 2 reached. Reached minimum node size."
        return create_leaf(target_values)
    
    # Find the best splitting feature
    splitting_feature = best_splitting_feature(data, features, target)
    
    # Split on the best feature that we found. 
    left_split = data[data[splitting_feature] == 0]
    right_split = data[data[splitting_feature] == 1]
    
    # stopping condition 5: Minimum error reduction
    # Error before splitting (number of misclassified examples 
    # divided by the total number of examples)
    error_before_split = intermediate_node_num_mistakes(target_values) / float(len(data))
    
    # Error after splitting (number of misclassified examples 
    # in both groups divided by the total number of examples)
    left_mistakes = intermediate_node_num_mistakes(left_split[target])
    right_mistakes = intermediate_node_num_mistakes(right_split[target])
    error_after_split = (left_mistakes + right_mistakes) / float(len(data))
    
    # If the error reduction is LESS THAN OR EQUAL TO min_error_reduction, return a leaf.
    if error_reduction(error_before_split,error_after_split) <= min_error_reduction :
        print "Early stopping condition 3 reached. Minimum error reduction."
        return create_leaf(target_values)
    
    
    remaining_features.remove(splitting_feature)
    print "Split on feature %s. (%s, %s)" % (splitting_feature, len(left_split), len(right_split))
    
    
    # Repeat (recurse) on left and right subtrees
    left_tree = decision_tree_create(left_split, remaining_features, target, current_depth + 1, max_depth, min_node_size, min_error_reduction)        
    
    right_tree = decision_tree_create(right_split, remaining_features, target, current_depth + 1, max_depth, min_node_size, min_error_reduction)
    
    
    return {'is_leaf'          : False, 
            'prediction'       : None,
            'splitting_feature': splitting_feature,
            'left'             : left_tree, 
            'right'            : right_tree}




def decision_tree_create_entropy(data, features, target, current_depth = 0, max_depth = 10, min_node_size = 1, min_error_reduction = 0.0):
    remaining_features = features[:] # Make a copy of the features.
    
    target_values = data[target]
    print "--------------------------------------------------------------------"
    print "Subtree, depth = %s (%s data points)." % (current_depth, len(target_values))
    

    # Stopping condition 1
    # (Check if there are mistakes at current node.
    # Recall you wrote a function intermediate_node_num_mistakes to compute this.)
    if calculate_entropy(target_values) == 0 :
        print "Stopping condition 1 reached."     
        # If not mistakes at current node, make current node a leaf node
        return create_leaf(target_values)
    
    # Stopping condition 2 (check if there are remaining features to consider splitting on)
    if remaining_features == [] :
        print "Stopping condition 2 reached."    
        # If there are no remaining features to consider, make current node a leaf node
        return create_leaf(target_values)    
    
    # Additional stopping condition (limit tree depth)
    if current_depth >= max_depth :
        print "Reached maximum depth. Stopping for now."
        # If the max tree depth has been reached, make current node a leaf node
        return create_leaf(target_values)

    # stopping condition 4: Reached the minimum node size.
    # If the number of data points is less than or equal to the minimum size, return a leaf.
    if reached_minimum_node_size(data,min_node_size) : 
        print "stopping condition 4 reached. Reached minimum node size."
        return create_leaf(target_values)

    # Find the best splitting feature (recall the function best_splitting_feature implemented above)
    splitting_feature = best_splitting_feature_entropy(data,features,target)

    
    # Split on the best feature that we found. 
    left_split = data[data[splitting_feature] == 0]
    right_split = data[data[splitting_feature] == 1]

    # stopping condition 5: Minimum error reduction
    # Error before splitting (number of misclassified examples 
    # divided by the total number of examples)
    error_before_split = intermediate_node_num_mistakes(target_values) / float(len(data))
    
    # Error after splitting (number of misclassified examples 
    # in both groups divided by the total number of examples)
    left_mistakes = intermediate_node_num_mistakes(left_split[target])
    right_mistakes = intermediate_node_num_mistakes(right_split[target])
    error_after_split = (left_mistakes + right_mistakes) / float(len(data))
    
    # If the error reduction is LESS THAN OR EQUAL TO min_error_reduction, return a leaf.
    if error_reduction(error_before_split,error_after_split) <= min_error_reduction :
        print "stopping condition 5 reached. Minimum error reduction."
        return create_leaf(target_values)

    remaining_features.remove(splitting_feature)
    print "Split on feature %s. (%s, %s)" % (splitting_feature, len(left_split), len(right_split))
    
    # Create a leaf node if the split is "perfect"
    if len(left_split) == len(data):
        print "Creating leaf node."
        return create_leaf(left_split[target])
    if len(right_split) == len(data):
        print "Creating leaf node."
        return create_leaf(right_split[target])

        
    # Repeat (recurse) on left and right subtrees
    left_tree = decision_tree_create_entropy(left_split, remaining_features, target, current_depth + 1, max_depth, min_node_size, min_error_reduction)        
    
    right_tree = decision_tree_create_entropy(right_split, remaining_features, target, current_depth + 1, max_depth, min_node_size, min_error_reduction)

    return {'is_leaf'          : False, 
            'prediction'       : None,
            'splitting_feature': splitting_feature,
            'left'             : left_tree, 
            'right'            : right_tree}



def count_nodes(tree):
    if tree['is_leaf']:
        return 1
    return 1 + count_nodes(tree['left']) + count_nodes(tree['right'])


# Testing the function

small_decision_tree = decision_tree_create(train_data, features, 'safe_loans', max_depth = 2, min_node_size = 10, min_error_reduction=0.0)
if count_nodes(small_decision_tree) == 7:
    print 'Test passed!'
else:
    print 'Test failed... try again!'
    print 'Number of nodes found                :', count_nodes(small_decision_tree)
    print 'Number of nodes that should be there : 7'


# Building the tree

print "Building the tree\n"
my_decision_tree_old = decision_tree_create(train_data, features, 'safe_loans', max_depth = 6, min_node_size = 0, min_error_reduction=-1)

# Making predictions with a decision tree

def classify(tree, x, annotate = False):   
    # if the node is a leaf node.
    if tree['is_leaf']:
        if annotate: 
            print "At leaf, predicting %s" % tree['prediction']
        return tree['prediction'] 
    else:
        # split on feature.
        split_feature_value = x[tree['splitting_feature']]
        if annotate: 
            print "Split on %s = %s" % (tree['splitting_feature'], split_feature_value)
        if split_feature_value == 0:
            return classify(tree['left'], x, annotate)
        else:
            return classify(tree['right'], x, annotate)


# Now, let's consider the first example of the test set and see what `my_decision_tree` model predicts for this data point.

print "Actual target value of test_data[0]"
print test_data[0]['safe_loans']

print "Predicted value on test_data[0]"
print classify(my_decision_tree_old, test_data[0], annotate = False)

# Evaluating the tree

def evaluate_tree(tree, data, target, flag = False):
    
    # Apply the classify(tree, x) to each row in your data
    prediction = data.apply(lambda x: classify(tree, x))
    
    tot = float(len(data))
    mistakes = (prediction != data[target])
    no_mistakes = len(mistakes[mistakes == 1])
    tp = float(((prediction == 1) & (data[target] == 1)).sum())
    tn = float(((prediction == -1) & (data[target] == -1)).sum())
    fp = float(((prediction == 1) & (data[target] == -1)).sum())
    fn = float(((prediction == -1) & (data[target] == 1)).sum())
    accuracy = ((tp + tn)/(tot))*100
    precision = (tp/(tp + fp))*100
    recall = (tp/(tp + fn))*100
    F1 = (2*precision*recall)/(precision + recall)
    
    if flag == False :

        print "Accuracy is: " + str(accuracy) + " %"
        print "Precision is: " + str(precision) + " %"
        print "Recall is: " + str(recall) + " %"
        print "F1 Score is: : %f" %F1

    else :

        print "Accuracy is: " + str(accuracy) + " %"



# Now, let's use this function to evaluate the `my_decision_tree_new` on the test_data.
print "Evaluating the tree on entire test_data\n"
evaluate_tree(my_decision_tree_old, test_data, target)

# Exploring the effect of max_depth
 # Train three models with these parameters:
# 
# 1. model_1: max_depth = 2 (too small)
# 2. model_2: max_depth = 6 (just right)
# 3. model_3: max_depth = 14 (may be too large)
# For each of these three, we set `min_node_size = 0` and `min_error_reduction = -1`.
print "Exploring the effect of max_depth\n"

model_1 = decision_tree_create(train_data, features, 'safe_loans', max_depth = 2, min_node_size = 0, min_error_reduction=-1)
model_2 = decision_tree_create(train_data, features, 'safe_loans', max_depth = 4, min_node_size = 0, min_error_reduction=-1)
model_3 = decision_tree_create(train_data, features, 'safe_loans', max_depth = 6, min_node_size = 0, min_error_reduction=-1)
model_4 = decision_tree_create(train_data, features, 'safe_loans', max_depth = 8, min_node_size = 0, min_error_reduction=-1)
model_5 = decision_tree_create(train_data, features, 'safe_loans', max_depth = 10, min_node_size = 0, min_error_reduction=-1)


# Evaluating the models
print "Training data,(model 1):"
evaluate_tree(model_1, train_data,target,True)
print "Training data,(model 2):"
evaluate_tree(model_2, train_data,target,True)
print "Training data,(model 3):"
evaluate_tree(model_3, train_data,target,True)
print "Training data,(model 4):"
evaluate_tree(model_4, train_data,target,True)
print "Training data,(model 5):"
evaluate_tree(model_5, train_data,target,True)

# Now evaluate on the test data.
print "Test data,(model 1):"
evaluate_tree(model_1, test_data,target,True)
print "Test data,(model 2):"
evaluate_tree(model_2, test_data,target,True)
print "Test data,(model 3):"
evaluate_tree(model_3, test_data,target,True)
print "Test data,(model 4):"
evaluate_tree(model_4, test_data,target,True)
print "Test data,(model 5):"
evaluate_tree(model_5, test_data,target,True)

# Measuring the complexity of the tree
# complexity(T) = number of leaves in the tree T

def count_leaves(tree):
    if tree['is_leaf']:
        return 1
    return count_leaves(tree['left']) + count_leaves(tree['right'])

print "Measuring the complexity of the tree"
print "complexity(T) = number of leaves in the tree T"

print "model_1" + str(count_leaves(model_1))
print "model_2" + str(count_leaves(model_2))
print "model_3" + str(count_leaves(model_3))
print "model_4" + str(count_leaves(model_4))
print "model_5" + str(count_leaves(model_5))



# Building the best model
print "Building the best model"

best_model = decision_tree_create(train_data, features, 'safe_loans', max_depth = 6, min_node_size = 2000, min_error_reduction=0)

print "Training data,(best_model):"
evaluate_tree(best_model, train_data,target,False)
print "Test data,(best_model):"
evaluate_tree(best_model, test_data,target,False)


# Printing out a decision stump

def print_stump(tree, name = 'root'):
    split_name = tree['splitting_feature']
    if split_name is None:
        print "(leaf, label: %s)" % tree['prediction']
        return None
    split_feature, split_value = split_name.split('.')
    print '                       %s' % name
    print '         |---------------|----------------|'
    print '         |                                |'
    print '         |                                |'
    print '         |                                |'
    print '  [{0} == 0]               [{0} == 1]    '.format(split_name)
    print '         |                                |'
    print '         |                                |'
    print '         |                                |'
    print '    (%s)                         (%s)'         % (('leaf, label: ' + str(tree['left']['prediction']) if tree['left']['is_leaf'] else 'subtree'),
           ('leaf, label: ' + str(tree['right']['prediction']) if tree['right']['is_leaf'] else 'subtree'))


print "\n"
print_stump(my_decision_tree_old)
print "\n"
print "left subtree"
print_stump(my_decision_tree_old['left'], my_decision_tree_old['splitting_feature'])
print "\n"
print "right subtree"
print_stump(my_decision_tree_old['right'], my_decision_tree_old['splitting_feature'])
