
<p style="color: red; font-weight: bold">>>>>>  gd2md-html alert:  ERRORs: 0; WARNINGs: 0; ALERTS: 2.</p>
<ul style="color: red; font-weight: bold"><li>See top comment block for details on ERRORs and WARNINGs. <li>In the converted Markdown or HTML, search for inline alerts that start with >>>>>  gd2md-html alert:  for specific instances that need correction.</ul>

<p style="color: red; font-weight: bold">Links to alert messages:</p><a href="#gdcalert1">alert1</a>
<a href="#gdcalert2">alert2</a>

<p style="color: red; font-weight: bold">>>>>> PLEASE check and correct alert issues and delete this message and the inline alerts.<hr></p>




### 15장 CNN  이미지 분류

Multilayer Perceptron (MLP)

-입력층, 은익층, 출력층,

           [https://github.com/rickiepark/python-machine-learning-book-2nd-edition](https://github.com/rickiepark/python-machine-learning-book-2nd-edition)


###### tensorflow 1.0 기본



1. Graph를 만든다
    1. placeholder 만들고 입력파라미터  형태 지정. x
    2. Variable w , Variable bias 지정
    3. init > variables  초기화
2. Session을 만들고  Sess.run
    4. sess.run(init)
    5. sess.run (  feed_dict={x:t})(
3. 

	


```
g = tf.Graph()
with g.as_default():
   x= tf.compat.v1.placeholder(dtype= tf.float32, shape=(None),name='x')
   w = tf.Variable(2.0,name='weight')
   b = tf.Variable(0.7,name='bias')

   z = w * x + b
   init = tf.compat.v1.global_variables_initializer()

with tf.compat.v1.Session(graph=g) as sess:
   sess.run(init)
   for t in [1.0,0.6,-1.8]:
           print('x=%4.1f -> %4.1f'%(
               t,sess.run(z,feed_dict={x:t})))
```



###### tensorflow 2.0 기본


```
#tensor flow 2.0
w = tf.Variable(2.0,name='weight')
b = tf.Variable(0.7,name='bias')

for x in [1.0,0.6,-1.8]:
   z = w * x + b
   print('x=%4.1f -> %4.1f'%(x,z))
print(z)

# tensor flow 2.0  for 문 제거
z = w * [1.0,0.6,-1.8] + b
print(z.numpy())
```


tf.keras가 tensorflow 2.0에서는 표준 파이썬 API

[http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)

minst 예제 



*   표준화 

    ```
#(Xi - mean(x)) / (std(x))  =>  mean(x): 평균  std(x): 표준편차
X_train_center = (X_train -mean_vals) / std_val
X_test_center =  (X_test - mean_vals) / std_val
```


*   원-핫 인코딩

    ```
y_train_onehot = tf.keras.utils.to_categorical(y_train)
```


*   모델 만들기 계층 쌓기

    ```
model = tf.keras.models.Sequential()
model.add(
   tf.keras.layers.Dense(
       units=50,
       input_dim = X_train_center.shape[1],
       kernel_initializer ='glorot_uniform',
       bias_initializer= 'zeros',
       activation='tanh'
   ))
```



kernel_initializer='glorot_uniform' 글로럿 초기화,세이비어(Xavier) 심층안정망을 안정적으로 초기화, 절편은 0으로 초기화 , 케라스의 기본값

activation='softmax'  소프트 맥스는 간적적인  argMax함수이다

각 클래스의 확률을 반환함으로 다중 클래스 환경에서 의미있는 클래스 확률을 계산 할 수 있다.



*   옵티마이저

    ```
sgd_optimizer = tf.keras.optimizers.SGD(
               lr=0.001,decay=1e-7,momentum=.9)
model.compile(optimizer=sgd_optimizer,loss='categorical_crossentropy')
```


*   훈련

    ```
history = model.fit(X_train_center,y_train_onehot,
                   batch_size=64,epochs=50,
                   verbose = 1,
                   validation_split=0.1)
```


*   예측

    ```
y_train_pred = model.predict_classes(X_train_center,verbose=00)
```



시그모이드 함수의 경우 : 함수가 0에 가까운 출력을 내면 신경망이 매우 느리게 학습하게된다. 지역 최소값에 갇힐 가능성이 높다 이런 이유로 하이퍼볼릭 탄젠트 함수를 선호

glorot_uniform


### 텐서 



*   랭크 (Rank): 텐서의 차원
*   get_shape() 메서드는  TensorShape객체를 반환
*   TensorShape 객체의 인덱스를 참조하면 Dimension 를 리턴함.
*   텐서플로우는 계산 그래프를 만들지 않고 텐서를 계산 할 수 있다.
*   
*   @tf.function : tf.function decorator
    *   tf.cond, tf.case, tf.while  같은 control flow를 tf.session()을 실행할 필요가 없음
    *   function앞에  tf.function 데코레이터만 붙이면 computational graph를 생성해주어 바로 사용하게 준다 @
    *   autograph: tf.function 안의 연산은 자동으로 텐서플로 그래프에 포함되어 실행
*    v1.0 vs v20
    *   2.0이 나오기 전 소스는 내부적으로는 tf.Session()을 이용하여 수행이 되고,
    *   오른쪽의 tensorflow 2.0소스는 내부적으로도 eager execution으로 실행이 됩니다
*   <code>[tf.function](https://www.tensorflow.org/api_docs/python/tf/function)</code>을 함수에 붙여줄 경우, 여전히 다른 일반 함수들처럼 사용할 수 있습니다. 하지만 그래프 내에서 컴파일 되었을 때는 더 빠르게 실행하고, GPU나 TPU를 사용해서 작동하고, 세이브드모델(SavedModel)로 내보내는 것이 가능해집니다.

    ```
@tf.function
def simple_func():
   a = tf.constant(1)
   b = tf.constant(2)
   c = tf.constant(3)

   z = 2*(a-b)+c
   return z
```


*   
*   tf.Tensor(1, shape=(), dtype=int32)
*   텐서플로우 변수
    *   tf.Variable(<initial-value>,name=optional-name)
    *   tf.Variable  에는 shape나 dtype를 설정할 수 없다.
    *   tf 2.x에서는 파이썬 객체를 공유하듯이 변수 공유
    *   tf 2.x 에서 변수값을 증가시키려면 assign() 메소드를 반복해서 전달하면 됨

        ```
w2 = tf.Variable(np.array([[1,2,3,4],
                            [5,6,7,8]]),name='w2')
w2.assign(w2+1)
w2.assign(w2+1)
```


    *   


##### 모델만드는 법 



*   방법1)

    ```
from tensorflow.keras import Model,Input
model = tf.keras.Sequential()
input = tf.keras.Input(shape=(1,))  # 튜플크기를 지정하여 input객체를 만듦.
output = tf.keras.layers.Dense(1)(input
```


*   방법2

    ```
dense = tf.keras.layers.Dense(1)
output= dense(input)
```


*   방법3

    ```
dense = tf.keras.layers.Dense(1)
output= dense.__call__(input)
model = tf.keras.Model(input,output)
```


*   훈련


```
model.compile(optimizer='sgd',loss='mse')
history = model.fit(x_train,y_train,epochs=500, validation_split=0.3)
```


평가 & 예측


```
model.evaluate(x_test,y_test)
x_arr = np.arange(-2,2,0.1)
y_arr = model.predict(x_arr)
```


\



*   가중치 저장

    ```
model = tf.keras.Model(input,output)
model.summary()
model.compile(optimizer='sgd',loss='mse')
history = model.fit(x_train,y_train,epochs=500, validation_split=0.3)

# 모델 가중치 적용
model.save_weights('simple_weight.h5')
```


*   모델 저장

    ```
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1,input_dim=1))
model.compile(optimizer='sgd',loss='mse')
model.load_weights('simple_weight.h5')
model.evaluate(x_test,y_test)
model.save('simple_model.h5')
```


*   모델 로딩 사용 

    ```
model = tf.keras.models.load_model('simple_model.h5')
model.load_weights('my_model.h5')
model.evaluate(x_test,y_test)
```


*   modelCheckpoint 

modelCheckpoint callback를 사용하여 최고의 가중치를 저장 할 수 있다.

EarlyStopping : 더 이상 성능이 개선되지 않을때


```
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=1,input_dim=1))
model.compile(optimizer='sgd',loss='mse')
callback_list = [tf.keras.callbacks.ModelCheckpoint(filepath='my_model.h5',
                                   monitor='val_loss',save_best_only=True),
                                   tf.keras.callbacks.EarlyStopping(patience=5)]
history= model.fit(x_train,y_train,epochs=500,
                   validation_split=0.2,callbacks=callback_list)
```


[https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.plot.html](https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.pyplot.plot.html)



<p id="gdcalert1" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/-0.png). Store image on your image server and adjust path/filename if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert2">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/-0.png "image_tooltip")



##### 텐서 보드 실행 



*   command 창에서 실행

tensorboard --logdir=logs



*   jupyter notebook에서 실행

    ```
%load_ext tensorboard
%tensorboard --logdir logs --port 6006
```



    ```
To reload it, use:
  %reload_ext tensorboard
Reusing TensorBoard on port 6006 (pid 77859), started 0:00:23 ago. (Use '!kill 77859' to kill it.)
```



케라스 층의 그래프 그리기


```
tf.keras.utils.plot_model(model,show_shapes=True,to_file="mode1.png")
```


show_shapes=True 입력,출력 크기도 함께 나타내줌



### 15장 심증합성곱으로 신경망으로 이미지 분류 



*   구성요소
    *   얀 르쿤(Yann LeGun) : 손글씨 분류하는 새로운 신경망 구조 발표
    *   에지나 동그라미 같은 저수준 특성이 앞 쪽에서 추출되고 이 특성이 결되어 고수준 특성(건물,자동차,강아지) 같은 것 특성을 형성한다.
    *   Feature Map: CNN는 입력이미지에서 특성맵을 만든다. 이 맵의 각 원소는 입력 이미지의 국부적인 픽셀 패치에서 유도됨
        1. 희소 연결 : 특성맵에 있는 하나의 원소는 작은 픽셀 매치 하나에만 연결
        2. 파라미터 공유: 동일한 가중치가 입력 이미지의 모든 패치에 사용됨
        3. 네트워크의 가중치(파라미터 갯수)가 극적으로 감소하고 중요한 특징을 잡아내는 능력이 향상됨.
    *   Conv 층 + Pooling층(subsampling) + FC 층( 다층 퍼셉트론)
        4. 풀링층은 학습파라미터 없고 가중치나 절편 유닛이 없다.
*   합성곱 출력의 크기 계산
    *   

<p id="gdcalert2" ><span style="color: red; font-weight: bold">>>>>>  gd2md-html alert: inline image link here (to images/-1.gif). Store image on your image server and adjust path/filename if necessary. </span><br>(<a href="#">Back to top</a>)(<a href="#gdcalert3">Next alert</a>)<br><span style="color: red; font-weight: bold">>>>>> </span></p>


![alt_text](images/-1.gif "image_tooltip")

    *   n : 입력벡터  p : 페딩  m: 필터의 크기  s:스트라이드 
    *   합성곱 연산자 ⊙
    *   최근에 푸리에 변환을 사용하여 합성곱을 계산.
*   서브 샘플링
    *   maxing pooling
    *   average polling
*   드롭아웃
*   데이타 전처리
    *   training set , validation set  분리
    *   표준편차 구함
    *   28*28 형태로 만듦
    *   결과 training
    *   to_categorical: one-hot encoding 으로 변경 
*   keras.API로 CNN구성

    ```
from tensorflow.keras import layers,models
model = models.Sequential()
model.add(layers.Conv2D(32,(5,5),padding='valid',activation='relu', input_shape=(28,28,1)))
#32 필터의 갯수   (5,5) 필터의 크기
#padding = valid 기본값 :  same 구성요소
#stride: 정수 두개로 이루어진 튜블
#kernel_initializer : 세이비어 초기화 방식 (글로럿) : glorot_uniform
model.add(layers.MaxPool2D((2,2)))
# 두번째 합성곱 층
model.add(layers.Conv2D(64,(5,5),padding='valid',activation='relu'))
model.add(layers.MaxPool2D((2,2)))
model.add(layers.Flatten())
# 1개 이상의 완전 연결층으로 연결됨
model.add(layers.Dense(1024, activation='relu'))  # 기본값 kernel_initialize: gloror_uniform  bias_init: zeros
model.add(layers.Dropout(0.5))  # 유닛을 끌 확률 매개변수
model.add(layers.Dense(10,activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['acc'])
import time
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
callback_list = [ModelCheckpoint(filepath='cnn_checkpoint.h5', monitor='val_loss',save_best_only=True),
                               TensorBoard(log_dir="logs/{}".format(time.asctime()))]
v_epoch = 50                               
history = model.fit(X_train_centered,y_train_onehot,
                   batch_size=64, epochs=v_epoch,
                   validation_data=(X_valid_centered,y_valid_onehot),
                   callbacks= callback_list)
```


*   Conv2D(32,(5,5),padding, activation. input_shape..)
*   MaxPooling2D
*   Conv2D(64,(5,5),padding,activation
*   MaxPooling2D
*   Flatten
*   Dense(1012,activation)
*   Dropout(0.5)
*   Dense(10,activation=softmax
*   compile(loss=’catergorical_crossentropy’,optimizer=’adam’
*   **fit( x_train,y_train,batch_size, epoch, validation_data, callback_list)**
*   체크포인트 콜백:
    *   ModelCheckPoint callback=: vla_loss, 가중치 저장,
    *   Tensorboard callback  
*   


###### loss 함수 시각화


```
import matplotlib.pyplot as plt
epochs = np.arange(1,v_epoch+1)ç
plt.plot(epochs,history.history['loss'])
plt.plot(epochs,history.history['val_loss'])
plt.xlabel('epochs')
plt.ylabel('loss')
plt.show()
```



###### 정확도 시각화


```
history.history['val_acc'] 
```



##### feature map, filter 시각화


<!-- Docs to Markdown version 1.0β17 -->
