# mnist

1. python svr.py 로 실행하거나
    python -m http.server --cgi 8000 으로 실행하면
    webserver 가 8000 포트로 실행된다.
    http://localhost:8000 으로 접속한다.
2. custom.js 에서 cgi 를 호출하는 python 을 수정해서 ANN 과 CNN 을 테스트 할 수 있다.
3. 테스트 방법은
    * web 에서 retrain 을 시키면 train.py 가 실행되고 학습한 Weight 값을 model 폴더에 저장한다.
    * 학습이 끝나면 web 의 canvas 에 원하는 숫자를 넣고 Predict 를 하면 mnist.py 가 실행된다.
    * mnist.py 는 학습해놓은 weight 값을 읽어서 model 에 load 하고 입력한 숫자를 model 을 통해 예측한 후 return한다.
    * return 된 예측이 표시된다.
