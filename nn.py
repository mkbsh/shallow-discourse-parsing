from __future__ import print_function

from confusion_matrix import ConfusionMatrix, Alphabet

import tensorflow as tf
import json
import gensim, logging
import numpy as np
import argparse
from collections import OrderedDict
import ast

train_file = "train/relations.json"
test_file = "test/relations.json"
eval_file = "dev/relations.json"
results_file = "relations.json"
# Parameters
learning_rate = 0.0001
num_steps = 100000
batch_size = 50
n_hidden_1 = 100
n_hidden_2 = 100
#n_hidden_3 = 300

def getWord2VecModel(cache_word2vec, word2vec_file):
    if cache_word2vec:
        return trainWord2VecModel(word2vec_file)
    return gensim.models.Word2Vec.load(word2vec_file)

def trainWord2VecModel(word2vec_file):
    lines = open(train_file)
    sentences = []
    count = 0
    for line in lines:
        relation = ast.literal_eval(line)
        sentence = relation["Arg1"]["RawText"] + ' ' + relation["Connective"]["RawText"] + ' ' + relation["Arg2"][
            "RawText"]
        tokenized = sentence.split()
        for i, word in enumerate(tokenized):
            count += 1
            if (count == 5000):
                tokenized[i] = 'UNK'
                count = 0
        sentences.append(tokenized)

    word2vec = gensim.models.Word2Vec(sentences, iter=200, min_count=1)
    word2vec.save(word2vec_file)

    return word2vec

def get_label_dict():
    label_index = {}
    index_label = {}

    get_label_dict_for_file(train_file, label_index, index_label)
    get_label_dict_for_file(test_file, label_index, index_label)
    get_label_dict_for_file(eval_file, label_index, index_label)

    return label_index, index_label

def get_label_dict_for_file(file_name, label_index, index_label):
    lines = open(file_name)
    for line in lines:
        relation = ast.literal_eval(line)
        sense = relation["Sense"][0]
        if (sense not in label_index):
            count = len(label_index)
            label_index[sense] = count
            index_label[count] = sense

def neural_net(feature_dict):
    x = feature_dict['wordvecs']
    layer_1 = tf.layers.dense(x, n_hidden_1, activation=tf.nn.relu)
    layer_2 = tf.layers.dense(layer_1, n_hidden_2, activation=tf.nn.relu)
    # layer_3 = tf.layers.dense(layer_2, n_hidden_3, activation=tf.nn.relu)
    # The output will be a tensor of size [batch_size, num_classes] where there is going to be a number associated with
    # each class.
    out_layer = tf.layers.dense(layer_2, len(label_index))
    return out_layer

def model_fn(features, labels, mode):
    # Build the neural network
    out_layer = neural_net(features)

    # Take the raw output and predictions
    pred_classes = tf.argmax(out_layer, axis=1)
    # Take the raw outputs and convert to probabilities for each class
    pred_probs = tf.nn.softmax(out_layer)

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode, predictions=pred_classes)


    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=out_layer, labels=tf.cast(labels, dtype=tf.int32)))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train = optimizer.minimize(loss,
                               global_step=tf.train.get_global_step())

    acc_op = tf.metrics.accuracy(labels=labels, predictions=pred_classes)

    estim_specs = tf.estimator.EstimatorSpec(
        mode=mode,
        predictions=pred_classes,
        loss=loss,
        train_op=train,
        eval_metric_ops={'accuracy': acc_op})

    return estim_specs

# Returns the list of labels and features from the data in the file specified.
def get_feat_labels(file_name):
    lines = open(file_name)
    labels = []
    embd = []

    # Get word embeddings of training data
    for line in lines:
        relation = ast.literal_eval(line)
        # For each of arg1, arg2, connective, get average of the word embeddings for each word.
        arg1v = arg2v = connectv = np.asarray(word2vec.wv['UNK'])
        if (relation["Arg1"]["RawText"] != ''):
            arg1v = np.mean([word2vec.wv[word] if word in word2vec.wv.vocab else word2vec.wv['UNK']
                             for word in relation["Arg1"]["RawText"].split()], axis=0)
        if (relation["Arg2"]["RawText"] != ''):
            arg2v = np.mean([word2vec.wv[word] if word in word2vec.wv.vocab else word2vec.wv['UNK']
                             for word in relation["Arg2"]["RawText"].split()], axis=0)
        if (relation["Connective"]["RawText"] != ''):
            connectv = np.mean([word2vec.wv[word] if word in word2vec.wv.vocab else word2vec.wv['UNK']
                                for word in relation["Connective"]["RawText"].split()], axis=0)
        # Concatenate the three vectors from arg1, arg2, connective.
        vec = np.concatenate((arg1v, connectv, arg2v))
        embd.append(vec)
        labels.append(label_index[relation["Sense"][0]])

    return np.asarray(embd), np.asarray(labels)

def write_results(predictions):
    results = open(results_file, 'w+')
    skeleton = open(test_file, 'r')
    for i, line in enumerate(skeleton):
        relation = ast.literal_eval(line)
        # Extract old info from test file and fill in with new predicted info
        new_relation = OrderedDict()
        new_relation["Arg1"] = {}
        new_relation["Arg2"] = {}
        new_relation["Connective"]= {}
        new_relation["Arg1"]["TokenList"] = [tokens[2] for tokens in relation["Arg1"]["TokenList"]]
        new_relation["Arg2"]["TokenList"] = [tokens[2] for tokens in relation["Arg2"]["TokenList"]]
        new_relation["Connective"]["TokenList"] = [] if len(relation["Connective"]["TokenList"]) <= 0 \
                            else [relation["Connective"]["TokenList"][0][2]]
        new_relation["DocID"] = relation["DocID"],
        new_relation["Sense"] = [index_label[predictions[i]]]
        new_relation["Type"] = relation["Type"]
        results.write(json.dumps(new_relation) + "\n")


# Parse command line arguments
parser = argparse.ArgumentParser(description='Optional app description')
parser.add_argument('--word2vec_file', action='store',
                    help='File containing word2vec file', default='word2vec')
parser.add_argument('--cache_word2vec', action='store_true',
                    help='Newly cache trained word2vec model to file', default=False)
parser.add_argument('--log', action='store_true',
                    help='Display training logs', default=False)
args = parser.parse_args()

# Set logging for training
if (args.log):
    tf.logging.set_verbosity(tf.logging.INFO)

print('\nconstructing label dictionary ...\n')
# Get label dictionary
label_index, index_label = get_label_dict()

# Get trained word2vec model
print('\nloading word2vec model ...\n')
word2vec = getWord2VecModel(args.cache_word2vec, args.word2vec_file)

train_features, train_labels = get_feat_labels(train_file)
eval_features, eval_labels = get_feat_labels(eval_file)
test_features, test_labels = get_feat_labels(test_file)

#batch_size = len(train_features)
np.set_printoptions(threshold=np.inf)

model = tf.estimator.Estimator(model_fn)

input_fn = tf.estimator.inputs.numpy_input_fn(
    x={'wordvecs': train_features}, y=train_labels, batch_size=batch_size, num_epochs=None, shuffle=True)

eval_fn = tf.estimator.inputs.numpy_input_fn(
    x={'wordvecs': eval_features}, y=eval_labels, batch_size=batch_size, num_epochs=1, shuffle=False)

test_fn = tf.estimator.inputs.numpy_input_fn(
    x={'wordvecs': test_features}, y=test_labels, batch_size=batch_size, num_epochs=1, shuffle=False)

print('\ntraining ...\n')
# Train the Model with 1000 iterations
model.train(input_fn=input_fn, steps=num_steps)

print('\nevaluating on dev set ...\n')
# Evaluate the Model. Use the Estimator 'evaluate' method
e_metrics = model.evaluate(eval_fn)
print("\ntesting Accuracy: %s\n" % e_metrics)

print('\npredicting on test set ...\n')
predictions = list(model.predict(test_fn))
accuracy = len([predictions[i] for i in range(0, len(test_labels)) if test_labels[i] == predictions[i]]) \
           / len(test_labels)
print("accuracy: ", accuracy)

print('\nwriting results to file ...\n')
write_results(predictions)
