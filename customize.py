import tensorflow as tf
import numpy as np
import pandas as pd 
from sklearn.model_selection import train_test_split

def my_model(features, labels, mode, params):
	# Create three fully connected layers each layer having a dropout
	# probability of 0.1.

	# print('hi')
	# print(features,labels)
	net = tf.feature_column.input_layer(features, params['feature_columns'])
	for units in params['hidden_units']:
		net = tf.layers.dense(net, units=units, activation=tf.nn.relu)

	# Compute logits (1 per class).
	logits = tf.layers.dense(net, params['n_classes'], activation=None)
	print(labels)
	# Compute predictions.
	# predicted_classes = tf.argmax(logits, 1)
	if mode == tf.estimator.ModeKeys.PREDICT:
		print('asdf',logits)
		predictions = {
			
			'logits': logits
		}
		return tf.estimator.EstimatorSpec(mode, predictions=predictions)

	# Compute loss.
	loss = tf.losses.mean_squared_error(labels=labels, predictions=logits)

	# Compute evaluation metrics.
	accuracy = tf.metrics.mean_squared_error(labels=labels,
								   predictions=logits,
								   name='acc_op')
	metrics = {'accuracy': accuracy}
	tf.summary.scalar('accuracy', accuracy[1])

	if mode == tf.estimator.ModeKeys.EVAL:
		return tf.estimator.EstimatorSpec(
			mode, loss=loss, eval_metric_ops=metrics)

	# Create training op.
	assert mode == tf.estimator.ModeKeys.TRAIN

	optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)
	train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
	return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)

def main():
	# args =parser.parse_args(argv[1:])

	# # Fetch the data
	d = pd.read_excel('data-mat.xls',dtype=np.float32)
	np.random.seed(0)
	train_i = train_test_split(range(163),train_size=0.75) 
	train= d.iloc[train_i[0]]
	test= d.iloc[train_i[1]]
	X_train = train.iloc[:,:16]
	Y_train = train.iloc[:,16:17]

	X_test = train.iloc[:,:16]
	Y_test = train.iloc[:,16:17]


	# (train_x, train_y), (test_x, test_y) = iris_data.load_data()

	# Feature columns describe how to use the input.
	my_feature_columns = [tf.feature_column.numeric_column(key="x",shape=[16])]
	# for key in range(X_train.shape[1]):
	# 	my_feature_columns.append()

	# Build 2 hidden layer DNN with 10, 10 units respectively.
	classifier = tf.estimator.Estimator(
		model_fn=my_model,
		params={
			'feature_columns': my_feature_columns,
			# Two hidden layers of 10 nodes each.
			'hidden_units': [10, 10],
			# The model must choose between 3 classes.
			'n_classes': 1,
		},
		model_dir='./model')


	train_input_fn = tf.estimator.inputs.numpy_input_fn(
	  x={"x":np.array(X_train)},
	  y=np.array(Y_train),
	  num_epochs= None,
	  shuffle=True
	  )
	classifier.train(
		input_fn=train_input_fn,
		steps=10000)

	return classifier

	 
def test(classifier):	 
	# Train the Model.
	d = pd.read_excel('data-mat.xls',dtype=np.float32)
	np.random.seed(0)
	train_i = train_test_split(range(163),train_size=0.75) 
	train= d.iloc[train_i[0]]
	test= d.iloc[train_i[1]]
	X_train = test.iloc[:,:16]
	Y_train = test.iloc[:,16:17]

	X_test = train.iloc[:,:16]
	Y_test = train.iloc[:,16:17]

	test_input_fn = tf.estimator.inputs.numpy_input_fn(
	  x={"x":np.array(X_test)},
	  y=np.array(Y_test),
	  num_epochs= 1,
	  shuffle=True
	  )

	train_input_fn = tf.estimator.inputs.numpy_input_fn(
	  x={"x":np.array(X_train)},
	  y=np.array(Y_train),
	  num_epochs= 1,
	  shuffle=True
	  )

	# Evaluate the model.
	eval_result = classifier.evaluate(
	input_fn=train_input_fn)

	print('\nTrain set accuracy: {accuracy:0.3f}\n'.format(**eval_result))
	eval_result = classifier.evaluate(
		input_fn=test_input_fn)

	print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))


def predict(classifier):
	d = pd.read_excel('data-mat.xls',dtype=np.float32)
	np.random.seed(0)
	train_i = train_test_split(range(163),train_size=0.75) 
	train= d.iloc[train_i[0]]
	test= d.iloc[train_i[1]]
	X_train = train.iloc[:,:16]
	Y_train = train.iloc[:,16:17]

	X_test = train.iloc[:,:16]
	Y_test = train.iloc[:,16:17]

	pred = X_test[0:3]
	predy = Y_test[0:3]
	pred_input_fn = tf.estimator.inputs.numpy_input_fn(
	  x={"x":np.array(pred)},
	  y=np.array(predy),
	  
	  shuffle=True
	  )
	prediction = classifier.predict(input_fn = pred_input_fn)
	return prediction,np.array(predy)
	# for x,y in zip(prediction,predy):
	# 	print(x,y)
	# # Generate predictions from the model
	# expected = ['Setosa', 'Versicolor', 'Virginica']
	# predict_x = {
	#     'SepalLength': [5.1, 5.9, 6.9],
	#     'SepalWidth': [3.3, 3.0, 3.1],
	#     'PetalLength': [1.7, 4.2, 5.4],
	#     'PetalWidth': [0.5, 1.5, 2.1],
	# }

	# predictions = classifier.predict(
	#     input_fn=lambda:iris_data.eval_input_fn(predict_x,
	#                                             labels=None,
	#                                             batch_size=args.batch_size))

	# for pred_dict, expec in zip(predictions, expected):
	#     template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

	#     class_id = pred_dict['class_ids'][0]
	#     probability = pred_dict['probabilities'][class_id]

	#     print(template.format(iris_data.SPECIES[class_id],
	#                           100 * probability, expec))