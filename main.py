
import re
import cv2, pickle
#from flask.wrappers import Request
import numpy as np
import tensorflow as tf
from cnn_tf import cnn_model_fn
import os
import sqlite3, pyttsx3
from keras.models import load_model
from threading import Thread
from flask import Flask, render_template, Response,request
import cv2
from datetime import datetime

app=Flask(__name__)
camera = cv2.VideoCapture(0)







engine = pyttsx3.init()
engine.setProperty('rate', 150)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
model = load_model('cnn_model_keras2.h5')

def markconv(convs):
    with open('templates\conversation.csv','r+') as f:
        myDataList = f.readlines()
        nameList =[]
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if convs not in myDataList:
            now = datetime.now()
            dt_string = now.strftime("%H:%M:%S")
            f.writelines(f'\n{dt_string},{convs}')
			#old_convs=convs
            

        #print(myDataList)    


def get_hand_hist():
	with open("hist", "rb") as f:
		hist = pickle.load(f)
	return hist

def get_image_size():
	img = cv2.imread('gestures/0/100.jpg', 0)
	return img.shape

image_x, image_y = get_image_size()

def keras_process_image(img):
	img = cv2.resize(img, (image_x, image_y))
	img = np.array(img, dtype=np.float32)
	img = np.reshape(img, (1, image_x, image_y, 1))
	return img

def keras_predict(model, image):
	processed = keras_process_image(image)
	pred_probab = model.predict(processed)[0]
	pred_class = list(pred_probab).index(max(pred_probab))
	return max(pred_probab), pred_class

def get_pred_text_from_db(pred_class):
	conn = sqlite3.connect("gesture_db.db")
	cmd = "SELECT g_name FROM gesture WHERE g_id="+str(pred_class)
	cursor = conn.execute(cmd)
	for row in cursor:
		return row[0]

def get_pred_from_contour(contour, thresh):
	x1, y1, w1, h1 = cv2.boundingRect(contour)
	save_img = thresh[y1:y1+h1, x1:x1+w1]
	text = ""
	if w1 > h1:
		save_img = cv2.copyMakeBorder(save_img, int((w1-h1)/2) , int((w1-h1)/2) , 0, 0, cv2.BORDER_CONSTANT, (0, 0, 0))
	elif h1 > w1:
		save_img = cv2.copyMakeBorder(save_img, 0, 0, int((h1-w1)/2) , int((h1-w1)/2) , cv2.BORDER_CONSTANT, (0, 0, 0))
	pred_probab, pred_class = keras_predict(model, save_img)
	if pred_probab*100 > 70:
		text = get_pred_text_from_db(pred_class)
	return text

def get_operator(pred_text):
	try:
		pred_text = int(pred_text)
	except:
		return ""
	operator = ""
	if pred_text == 1:
		operator = "+"
	elif pred_text == 2:
		operator = "-"
	elif pred_text == 3:
		operator = "*"
	elif pred_text == 4:
		operator = "/"
	elif pred_text == 5:
		operator = "%"
	elif pred_text == 6:
		operator = "**"
	elif pred_text == 7:
		operator = ">>"
	elif pred_text == 8:
		operator = "<<"
	elif pred_text == 9:
		operator = "&"
	elif pred_text == 0:
		operator = "|"
	return operator

hist = get_hand_hist()
x, y, w, h = 300, 100, 300, 300
is_voice_on = True

def get_img_contour_thresh(img):
	img = cv2.flip(img, 1)
	imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
	dst = cv2.calcBackProject([imgHSV], [0, 1], hist, [0, 180, 0, 256], 1)
	disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(10,10))
	cv2.filter2D(dst,-1,disc,dst)
	blur = cv2.GaussianBlur(dst, (11,11), 0)
	blur = cv2.medianBlur(blur, 15)
	thresh = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
	thresh = cv2.merge((thresh,thresh,thresh))
	thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
	thresh = thresh[y:y+h, x:x+w]
	contours = cv2.findContours(thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]
	return img, contours, thresh

def say_text(text):
	if not is_voice_on:
		return
	while engine._inLoop:
		pass
	engine.say(text)
	engine.runAndWait()

def calculator_mode(cam):
	global is_voice_on
	flag = {"first": False, "operator": False, "second": False, "clear": False}
	count_same_frames = 0
	first, operator, second = "", "", ""
	pred_text = ""
	calc_text = ""
	info = "Enter first number"
	Thread(target=say_text, args=(info,)).start()
	count_clear_frames = 0
	while True:
		img = cam.read()[1]
		img = cv2.resize(img, (640, 480))
		img, contours, thresh = get_img_contour_thresh(img)
		old_pred_text = pred_text
		if len(contours) > 0:
			contour = max(contours, key = cv2.contourArea)
			if cv2.contourArea(contour) > 10000:
				pred_text = get_pred_from_contour(contour, thresh)
				if old_pred_text == pred_text:
					count_same_frames += 1
				else:
					count_same_frames = 0

				if pred_text == "C":
					if count_same_frames > 5:
						count_same_frames = 0
						first, second, operator, pred_text, calc_text = '', '', '', '', ''
						flag['first'], flag['operator'], flag['second'], flag['clear'] = False, False, False, False
						info = "Enter first number"
						Thread(target=say_text, args=(info,)).start()

				elif pred_text == "Best of Luck " and count_same_frames > 15:
					count_same_frames = 0
					if flag['clear']:
						first, second, operator, pred_text, calc_text = '', '', '', '', ''
						flag['first'], flag['operator'], flag['second'], flag['clear'] = False, False, False, False
						info = "Enter first number"
						Thread(target=say_text, args=(info,)).start()
					elif second != '':
						flag['second'] = True
						info = "Clear screen"
						#Thread(target=say_text, args=(info,)).start()
						second = ''
						flag['clear'] = True
						try:
							calc_text += "= "+str(eval(calc_text))
						except:
							calc_text = "Invalid operation"
						if is_voice_on:
							speech = calc_text
							speech = speech.replace('-', ' minus ')
							speech = speech.replace('/', ' divided by ')
							speech = speech.replace('**', ' raised to the power ')
							speech = speech.replace('*', ' multiplied by ')
							speech = speech.replace('%', ' mod ')
							speech = speech.replace('>>', ' bitwise right shift ')
							speech = speech.replace('<<', ' bitwise leftt shift ')
							speech = speech.replace('&', ' bitwise and ')
							speech = speech.replace('|', ' bitwise or ')
							Thread(target=say_text, args=(speech,)).start()
					elif first != '':
						flag['first'] = True
						info = "Enter operator"
						Thread(target=say_text, args=(info,)).start()
						first = ''

				elif pred_text != "Best of Luck " and pred_text.isnumeric():
					if flag['first'] == False:
						if count_same_frames > 15:
							count_same_frames = 0
							Thread(target=say_text, args=(pred_text,)).start()
							first += pred_text
							calc_text += pred_text
					elif flag['operator'] == False:
						operator = get_operator(pred_text)
						if count_same_frames > 15:
							count_same_frames = 0
							flag['operator'] = True
							calc_text += operator
							info = "Enter second number"
							Thread(target=say_text, args=(info,)).start()
							operator = ''
					elif flag['second'] == False:
						if count_same_frames > 15:
							Thread(target=say_text, args=(pred_text,)).start()
							second += pred_text
							calc_text += pred_text
							count_same_frames = 0	

		if count_clear_frames == 30:
			first, second, operator, pred_text, calc_text = '', '', '', '', ''
			flag['first'], flag['operator'], flag['second'], flag['clear'] = False, False, False, False
			info = "Enter first number"
			Thread(target=say_text, args=(info,)).start()
			count_clear_frames = 0

		blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
		cv2.putText(blackboard, "Calculations", (100, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 135,0))
		cv2.putText(blackboard, "Prediction- " + pred_text, (30, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 0))
		cv2.putText(blackboard, "Operator " + operator, (30, 140), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 127))
		cv2.putText(blackboard, calc_text, (30, 240), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 255))
		cv2.putText(blackboard, info, (30, 440), cv2.FONT_HERSHEY_TRIPLEX, 1, (0, 255, 255) )
		if is_voice_on:
			cv2.putText(blackboard, "Voice on", (450, 440), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 127, 0))
		else:
			cv2.putText(blackboard, "Voice off", (450, 440), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 127, 0))
		cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
		res = np.hstack((img, blackboard))
		cv2.imshow("Gesture", res)
		cv2.imshow("Thresh", thresh)
		keypress = cv2.waitKey(1)
		if keypress == ord('q') or keypress == ord('t'):
			break
		if keypress == ord('v') and is_voice_on:
			is_voice_on = False
		elif keypress == ord('v') and not is_voice_on:
			is_voice_on = True

	if keypress == ord('t'):
		return 1
	else:
		return 0

def text_mode(cam):
	global is_voice_on
	text = ""
	word = ""
	count_same_frame = 0
	while True:
		img = cam.read()[1]
		img = cv2.resize(img, (640, 480))
		img, contours, thresh = get_img_contour_thresh(img)
		old_text = text
		if len(contours) > 0:
			contour = max(contours, key = cv2.contourArea)
			if cv2.contourArea(contour) > 10000:
				text = get_pred_from_contour(contour, thresh)
				if old_text == text:
					count_same_frame += 1
				else:
					count_same_frame = 0

				if count_same_frame > 20:
					if len(text) == 1:
						Thread(target=say_text, args=(text, )).start()
					word = word + text
					#print(word)
					
					if word.startswith('I/Me '):
						word = word.replace('I/Me ', 'I ')
						
					elif word.endswith('I/Me '):
						word = word.replace('I/Me ', 'me ')
						
					count_same_frame = 0
					#print(word)
					markconv(word)

			elif cv2.contourArea(contour) < 1000:
				if word != '':
					markconv(word)
					#print('yolo')
					#say_text(text)
					Thread(target=say_text, args=(word, )).start()
				    
				text = ""
				word = ""
		else:
			if word != '':
				#print('yolo1')
				#say_text(text)
				Thread(target=say_text, args=(word, )).start()
			text = ""
			word = ""
		blackboard = np.zeros((480, 640, 3), dtype=np.uint8)
		cv2.putText(blackboard, "Text", (180, 50), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 167,0))
		cv2.putText(blackboard, "Prediction- " + text, (30, 100), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 0, 0))
		cv2.putText(blackboard, word, (30, 240), cv2.FONT_HERSHEY_TRIPLEX, 2, (255, 255, 0))
		if is_voice_on:
			cv2.putText(blackboard, "Voice ON", (450, 440), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 127, 89))
		else:
			cv2.putText(blackboard, "Voice OFF", (450, 440), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 127, 78))
		cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
		res = np.hstack((img, blackboard))
		cv2.imshow("Gesture", res)
		cv2.imshow("Thresh", thresh)
		keypress = cv2.waitKey(1)
		if keypress == ord('q') or keypress == ord('c'):
			break
		if keypress == ord('v') and is_voice_on:
			is_voice_on = False
		elif keypress == ord('v') and not is_voice_on:
			is_voice_on = True

	if keypress == ord('c'):
		return 2
	else:
		return 0

def recognize():
	cam = cv2.VideoCapture(0)
	if cam.read()[0]==False:
		cam = cv2.VideoCapture(0)
	text = ""
	word = ""
	count_same_frame = 0
	keypress = 1
	while True:
		if keypress == 1:
			keypress = text_mode(cam)
		elif keypress == 2:
			keypress = calculator_mode(cam)
		else:
			break

keras_predict(model, np.zeros((50, 50), dtype = np.uint8))		


def gen_frames():  
    while True:
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
			
            detector=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
            faces=detector.detectMultiScale(frame,1.1,7)
             #Draw the rectangle around each face
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
c='0'                	   

@app.route('/')
def index():
    return render_template('index2.html')

@app.route('/send',methods=['GET','POST'])
def index2():
    c='1'
    if c=='1':
       recognize()
    return c
    if request.method=='POST':
		    c=request.form['name']
		    print('hi')
    

            

    


@app.route('/video_feed')
def video_feed():
	return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')




	

	#return c
# def cvr():
# 	d='0'
# 	if request.method=='POST':
# 		c=request.form['name']
# 		print('hi')
#         d=c
#         return d
    
	
#k=index2()	
	

	#return render_template('index.html')






if __name__=='__main__':
    app.run(debug=True)
