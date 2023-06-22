from django.http import StreamingHttpResponse
from django.shortcuts import render
from django.views import View

#画像認識
import cv2
import numpy as np
from sklearn import svm
from PIL import Image
import pickle
from django.shortcuts import render

#音声認識
from django.http import HttpResponse
import speech_recognition as sr    #音声認識のライブラリ
from googletrans import Translator #翻訳（google翻訳）
from gtts import gTTS              #テキストデータを音声に変換
import pygame                      #pythonでゲームを作るために用意されたライブラリ
import time                        #標準ライブラリ 時間の管理
from mutagen.mp3 import MP3 as mp3 #音声ファイルのメタデータの取り出し



import os
from datetime import datetime
from django.views import View
# Create your views here.


transSrcLang=""
transSrcReco="ja-JP"
transDestLang=""

faceid=0
facelist=[]
for i in range(1000):
    facelist.append("")
facelist[999]="T.Y"


def recofacefunction(request):
    return render(request,'recoface.html')

def registfacefunction(request):
    return render(request, 'registface.html')

#顔検出と機械学習
def regist_face(request):
    global faceid
    global facelist
    facecascade_path="./haarcascade_frontalface_alt.xml"
    facecascade=cv2.CascadeClassifier(facecascade_path)


    num=10        #add
    label=str(faceid).zfill(3)   #add
    dir="faceimages/"   #add


    capture=cv2.VideoCapture(0)


    width=capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    height=capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

    width=width / 2
    height=height / 2

    count=0
    while(True):
        if count>=num:   #add
            break        #add

        ret,frame=capture.read()
        windowsize=(int(width),int(height))

        frame=cv2.resize(frame,windowsize)
        src_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        faces=facecascade.detectMultiScale(src_gray)

        for x,y,w,h in faces:
            #cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),3)
            writeimage=cv2.resize(frame[int(y):int(y)+int(h),int(x):int(x)+int(w)],(150,150),interpolation=cv2.INTER_LINEAR)
            
            strdate=datetime.now().strftime('%Y-%m-%d-%H-%M-%S.%f')
            filename=dir+label+"-"+str(strdate)+".jpg"
            
            ret=cv2.imwrite(filename,writeimage)
            count=count+1

        #cv2.imshow('cam',frame)

        #if cv2.waitKey(1) & 0xFF == ord('q'):
         #   break
        #if cv2.waitKey(1) & 0xFF == ord('w'):
        #    writeimage=cv2.resize(frame[int(y):int(y)+int(h),int(x):int(x)+int(w)],(200,200),interpolation=cv2.INTER_LINEAR)
        #    cv2.imwrite('./myface1.jpg',writeimage)
        #    break   
    facelist[faceid]=request.POST.get("facename")
    faceid=faceid+1


    images=[]
    label=[]

    #path="./recoface/MyFace/"

    for f in os.listdir(dir):
        image_path=os.path.join(dir,f)

        if image_path=="./faceimages/.DS_Store":
            continue
        else:
            gray_image=Image.open(image_path).convert("L")
            img=np.array(gray_image,"uint8")
            img=img.flatten()
            images.append(img)
            label.append(str(f[0:3]))

    images=np.array(images)
    label=np.array(label)

    clf=svm.LinearSVC()
    clf.fit(images,label)

    filename="face_model.sav"
    pickle.dump(clf,open(filename,"wb"))


    capture.release()
    #cv2.destroyAllWindows()

    return HttpResponse()

def video_feed_view():
    return lambda _: StreamingHttpResponse(generate_frame(), content_type='multipart/x-mixed-replace; boundary=frame')




def generate_frame():
    
    capture = cv2.VideoCapture(0) 
    width=capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    height=capture.get(cv2.CAP_PROP_FRAME_HEIGHT)

    width=width / 2
    height=height / 2

    showcount=0
    
    clf=pickle.load(open("./face_model.sav","rb"))
    facecascade_path="./haarcascade_frontalface_alt.xml"
    facecascade=cv2.CascadeClassifier(facecascade_path)

    while True:
        if not capture.isOpened():
            print("Capture is not opened.")
            break
        ret, frame = capture.read()
        if not ret:
            print("Failed to read frame.")
            break

        windowsize=(int(width),int(height))

         
        frame=cv2.resize(frame,windowsize)
        src_gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        faces=facecascade.detectMultiScale(src_gray)
        for x,y,w,h in faces:
            writeimage=cv2.resize(frame[int(y):int(y)+int(h),int(x):int(x)+int(w)],(150,150),interpolation=cv2.INTER_LINEAR)
            testimage=writeimage
            testimage = cv2.resize(testimage,(150,150),interpolation=cv2.INTER_LINEAR)
            testimage =cv2.cvtColor(testimage, cv2.COLOR_BGR2GRAY)
            testimagenp=np.array(testimage,"uint8")
            testimagenp=testimagenp.flatten()
            testarr=[]
            testarr.append(testimagenp)
            teatarr=np.array(testarr)

            pred=clf.predict(teatarr)
            
            if pred =="999":
                namestr="Yamashita"
            elif pred =="401":
                namestr="DEEPANSHU"
            elif pred =="001":
                namestr="PAVEL"
            elif pred =="002":
                namestr="HAI"
            elif pred =="004":
                namestr="HOANG"
            elif pred =="006":
                namestr="THIDA"
            elif pred =="007":
                namestr="PHUC"
            elif pred =="009":
                namestr="NGUYEN"
            elif pred =="010":
                namestr="THU"
            elif pred =="017":
                namestr="ARISA"
            elif pred =="018":
                namestr="HARUNA"
            else:
                namestr="???"

            cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
            cv2.putText(frame,namestr,(int(x),y-int(h/5)),cv2.FONT_HERSHEY_PLAIN, int(w/50),(255,0,0),2,cv2.LINE_AA)

           
        #---------------------------------------------------
        
        ret, jpeg = cv2.imencode('.jpg', frame)
        byte_frame = jpeg.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + byte_frame + b'\r\n\r\n')
    capture.release()


def indexfunction(request):
    return render(request,'index.html')

    #音声認識
def translatefunction(request):
    return render(request, 'translate.html')

def TranslateText(word):
    global transSrcLang
    global transDestLang

    trans=Translator()
    word2=trans.translate(word,dest="en",src=transSrcLang)
    word2=trans.translate(word2.text,dest=transDestLang,src="en")
    return word2.text

def TransTalk(word):
    global filename

    filename="out" + datetime.now().strftime("%d%m%Y%H%M%S")+".mp3"
    
    tts=gTTS(word,lang=transDestLang)
    tts.save(filename)

def playMp3():

    global filename

    pygame.mixer.init()
    pygame.mixer.music.load(filename) 
    mp3_length = mp3(filename).info.length 
    pygame.mixer.music.play(1) 
    time.sleep(mp3_length + 0.25) 
    pygame.mixer.music.stop() 

def test_ajax_response(request):
    global transSrcLang
    global transDestLang

    transSrcLang=request.POST.get("src_text")
    transDestLang=request.POST.get("dest_text")
    #transSrcLang="ja"
    transSrcReco="ja-JP"
    #transDestLang="my"

    r=sr.Recognizer()
    mic=sr.Microphone()

    filename=""
    while True:
        #print('Say something...')
        with mic as source:
            r.adjust_for_ambient_noise(source)
            audio=r.listen(source)

        #print('Now to recognize it...')
        resStr=[]

        try:
            text=r.recognize_google(audio,language=transSrcLang)
            transtext=TranslateText(text)
            if (transtext!=""):
               TransTalk(transtext)
               playMp3()
            
            resStr.append(text)
            resStr.append(transtext)
            
            #if r.recognize_google(audio,language='ja')=="ストップ":
            #    print('end.')
            return HttpResponse(transtext)
            break

        except sr.UnknownValueError:
            print('could not understand')
        except sr.RequestError as e:
            print('could not request')  
