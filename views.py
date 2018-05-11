import matplotlib
matplotlib.use('Agg')
import tensorflow as tf
import numpy as np
import own_model		
import pandas as pd
from sklearn.model_selection import train_test_split
from app import app
from flask import render_template,request,send_from_directory
from werkzeug import secure_filename
from matplotlib.pyplot import hist
import pickle
import matplotlib.pyplot as plt
import PIL
# import lsh.lshparser as lshparser
import warnings
from flask import make_response
from functools import wraps, update_wrapper
from datetime import datetime
user =''
passw = ''
check = True
headVal,bodyVal,m='','',''
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

loss,a2,xP,yP,scale,Aloss,Aa2,AxP,AyP,Ascale = 10*[0]

model1_graph = tf.Graph()
sess = tf.Session(graph =model1_graph,config=config)

model2_graph= tf.Graph()
sess1 = tf.Session(graph =model2_graph,config=config)

with sess.as_default():
	with model1_graph.as_default():
		global loss,a2,xP,yP,scale,sess
		new_restore = tf.train.import_meta_graph('newAbu/abu_model-25000.meta')
		new_restore.restore(sess,tf.train.latest_checkpoint('./newAbu/'))	
		graph = tf.get_default_graph()
		scale = pickle.load(open('normAbuNewScaler.p','rb'))
		xP = graph.get_tensor_by_name("x:0")
		yP = graph.get_tensor_by_name("y:0")
		a2 = graph.get_tensor_by_name("a4:0")
		loss = tf.losses.mean_squared_error(yP,a2)



with sess1.as_default():
	with model2_graph.as_default():
		global Aloss,Aa2,AxP,AyP,Ascale,sess1
		sess1 = tf.Session(config=config)
		outlier = pickle.load(open('AjmanOutlier.p','rb'))
		new_restore = tf.train.import_meta_graph('ajman/ajman_model-20000.meta')
		new_restore.restore(sess1,tf.train.latest_checkpoint('./ajman/'))
		pd.DataFrame()
		# new_restore = tf.train.import_meta_graph('newAjman/ajman_model-95000.meta')
		# new_restore.restore(sess1,tf.train.latest_checkpoint('./newAjman/'))	
		graph = tf.get_default_graph()
		Ascale = pickle.load(open('newAjmanScaler.p','rb'))
		# Ascale = pickle.load(open('normAjmanScaler.p','rb'))

		AxP = graph.get_tensor_by_name("x:0")
		AyP = graph.get_tensor_by_name("y:0")
		Aa2 = graph.get_tensor_by_name("a4:0")
		Aloss = tf.losses.mean_squared_error(AyP,Aa2)
option= 0

# warnings.filterwarnings("ignore")
# sess = tf.Session()
# new_restore = tf.train.import_meta_graph('my_model-10000.meta')
# new_restore.restore(sess,tf.train.latest_checkpoint('./'))	
# graph = tf.get_default_graph()
# xP = graph.get_tensor_by_name("x:0")
# yP = graph.get_tensor_by_name("y:0")
# a2 = graph.get_tensor_by_name("a2:0")

# CHANGES - 15-sep-17 -- DIRECT MATCH






@app.route("/answer", methods=['GET', 'POST'])
def predict():
	if request.method == 'POST':
		train_batch,train_label_batch,test_batch,test_label_batch = getData()
		cost,ans = sess.run([loss,a2],feed_dict={xP:X_test,yP:Y_test})

	else:
		print ('no post')
	return render_template('result_page.html',tables=[final_result.to_html()],thequery=printquery, 
										synonymquery=synonymquery, val=val, val1=val1)

@app.route('/uploader', methods = [ 'POST'])
def upload_file():
	print('helLO',request.method)
	global headVal,bodyVal,m
	global check
	if request.method == 'POST':
		try:
			print('get printed')
			f = request.files['file']
			print('SA',f.filename)
			f.save(secure_filename('./file.csv'))
			# return 'file uploaded successfully'
			return render_template('index.html',article='<p>File Uploaded</p>')
		except Exception:
			check = False
			return render_template('index.html',article='<p>Reupload! File Not Uploaded</p>')
	else:
		return render_template('result.html',head=headVal,body = bodyVal,final = m)	
		# if check:
		# 	return render_template('index.html',article='<p>File Uploaded</p>')
		# else:	
		# 	return render_template('index.html',article='<p>Reupload! File Not Uploaded</p>')		
def nocache(view):
	@wraps(view)
	def no_cache(*args, **kwargs):
		response = make_response(view(*args, **kwargs))
		response.headers['Last-Modified'] = datetime.now()
		response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
		response.headers['Pragma'] = 'no-cache'
		response.headers['Expires'] = '-1'
		return response
		
	return update_wrapper(no_cache, view)
@app.route('/component.css')
@nocache
def sendcssa3():
	return send_from_directory('./static', 'component.css')


@app.route('/demo.css')
def sendcssa4():
	return send_from_directory('./static', 'demo.css')


@app.route('/normalize.css')
def sendcssa1():
	return send_from_directory('./static', 'normalize.css')		

@app.route('/jquery-v1.min.js')
def sendjqeury():
	return render_template('jquery-3.2.1.min.js')

@app.route('/custom-file-input.js')
def sendjqeury1():
	return render_template('custom-file-input.js')	

@app.route('/bootstrap.min.css')
def sendcss1():
	return send_from_directory('./static', 'bootstrap.min.css')	
@app.route('/animate.css')
def sendcss2():
	return send_from_directory('./static', 'animate.css')	
@app.route('/select2.min.css')
def sendcss3():
	return send_from_directory('./static', 'select2.min.css')	
@app.route('/perfect-scrollbar.css')
def sendcss4():
	return send_from_directory('./static', 'perfect-scrollbar.css')	

@app.route('/style.css')
def sendercss7():
	return send_from_directory('./static', 'style.css')	
@app.route('/util.css')
def sendcss5():
	return send_from_directory('./static', 'util.css')	
@app.route('/main.css')
def sendcss6():
	return send_from_directory('./static', 'main.css')	


@app.route('/codropsicons.eot')
def sendFont():
	return send_from_directory('./static', 'codropsicons.eot')

@app.route('/codropsicons.svg')
def sendFont1():
	return send_from_directory('./static', "codropsicons.svg")

@app.route('/codropsicons.ttf')
def sendFont2():
	return send_from_directory('./static', "codropsicons.ttf")



@app.route('/codropsicons.woff')
def sendFont3():
	return send_from_directory('./static',  "codropsicons.woff")		
@app.route('/plot.png')
@nocache
def sendImage1():
	return send_from_directory('./templates',  "plot.png")
@app.route('/new.png')
@nocache
def sendImage2():
	return send_from_directory('./templates',  "new.png")	

@app.route('/logo.jpg')
def sendImage():
	return send_from_directory('./static',  "logo.jpg")	

@app.route('/open', methods = [ 'POST'])
def sendOutput():
	global option,headVal,bodyVal,m
	print("methods",request.method)
	# print("methods",request.method)
	if request.method == 'POST':
		print('notworking')
		option = request.form['x']
		# print('this is+ the val',x)
		# return 'done'
	# elif  request.method == 'GET':
	# if request.method == 'GET':
		global loss,a2,xP,yP,scale,sess,fig,Aloss,Aa2,AxP,AyP,Ascale,sess1,outlier

		# sess = tf.Session()
		if option == 'abu':
			with sess.as_default():
				test_X,test_Y = own_model.getEvaluateData(scale)
				cost,ans = sess.run([loss,a2],feed_dict={xP:test_X,yP:test_Y})
				print(cost,'abu loss')

		elif option == 'ajman':
			with sess1.as_default():	
				test_X,test_Y = own_model.getEvaluateData(Ascale)
				val = outlier.predict(test_X)
				args = np.where(val == 1)
				print(type(args))

				# test_X= test_X.iloc[args[0],:]
				# test_Y = test_Y.iloc[args[0],:]
				cost,ans = sess1.run([Aloss,Aa2],feed_dict={AxP:test_X,AyP:test_Y})
				

				print(cost,'ajman loss')
		# print(cost,ans-Y_test)
		# sess.close()


		val  = np.shape(test_Y)[0]
		val =val -  val/20
		print('these many removed',val)
		test_Y= np.array(test_Y)[:val,:]
		ans = np.array(ans)[:val,:]
		d = np.array(test_Y) - np.array(ans)
		print('shape of d',d.shape)

		# pickle.dump(open('difference.p','wb')
		dAbs= np.absolute(d)
		g4 = dAbs <4

		print('before',d.max())
		if np.sum(g4)>1:
			d = d[g4]
			ans = ans[g4]
			test_Y = test_Y[g4]
		print('shape of d',d.shape)
		print('after',d.max())
		histd = d.reshape(d.shape[0])
		number = d.shape[0]
		
		y = tuple(np.array(test_Y).reshape(number))
		dT = tuple(d.reshape(number))

		g3 = np.sum(dAbs>3)
		g2l3 = np.sum(np.bitwise_and(dAbs<3 ,dAbs>2))
		g1l2 =  np.sum(np.bitwise_and(dAbs>1 , dAbs <2))
		gpointl1 =  np.sum(np.bitwise_and(dAbs>0.5 , dAbs <1))
		number = float(number)
		g0lpoint =  np.sum(np.bitwise_and(dAbs>0 , dAbs <0.5))
		#print('greater than 3',g3/number,'greater 2 less 3',g2l3/number,'greater 1 less 2',g1l2/number,'greater 0.5 less 1',gpointl1/number,'greater 0 less 0.5',g0lpoint/number)
		table = pd.DataFrame(columns = ['+-0','+-.5','+-1', '+-2','+-3'])
		table.loc[0] =  [g0lpoint,gpointl1,g1l2,g2l3,g3]
		m = table.to_html(border=0)
		# print(histd.max(),'this is the shape')
		
		hist(histd,50)
		plt.savefig('./app/templates/new.png')
		plt.show()
		plt.clf()
		plt.bar(y,dT,width=0.09,align='center')
		plt.xlabel('Val')
		plt.ylabel('Diff')
		# plt.show()
		plt.savefig('./app/templates/plot.png')
		plt.clf()
		plt.close()



		# plt.show()
		# return d
		print('cost is '+ str(cost), 'min is '+ str(d.min()) + 'max is ' + str(d.max()))
		ans =pd.DataFrame(ans)
		ans = pd.concat([ans.reindex(),pd.DataFrame(np.array(test_Y)).reindex(),pd.DataFrame(d).reindex()],axis=1)
		ans.columns = ["Predicted","Actual","Diff"]
		# ans.to_csv('ansval.csv')
		val = ans.to_html(border = 0)
		# plt.show()

		# fig.canvas.draw()
		html = val.split('</thead>')
		headVal  =html[0] +"</thead>"
		bodyVal = html[1]
		return render_template('result.html',head=headVal,body = bodyVal,final = m)

	# fig.canvas.draw()
	# data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
	# data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
	# img = PIL.Image.fromarray(data)
	# img.save("out.jpg")
	# plt.show()
	# sess.close()

@app.route('/loggedin', methods=['POST'])
def logindex():
	print('rendered')
	global user,passw,headVal,bodyVal,m
	if request.method == 'POST':
		print('notworking')
		user = request.form['user']
		passw = request.form['pass']
		print(user,passw)
		if user=="admin" and passw=="helloworld":
			return render_template('index.html')
		# print('this is+ the val',x)
		else:
			return "<h2>Wrong Password</h2>"
		# return 'cool'

	if request.method == 'GET':
		print('is it coming')
		return render_template('result.html',head=headVal,body = bodyVal,final = m)		
	# if request.method == 'GET':	
	# 	return render_template('login.html')

@app.route('/')
def index():
	print('rendered')
	return render_template('login.html')