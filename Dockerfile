FROM python:3

# 各種セットアップ
WORKDIR /docker-work

# 必要なファイルを移す。不要なファイルは.dockerignoreファイルで定義しておく。
ADD . .

# pip3の実行
RUN pip3 install -r requirement.txt
RUN pip3 install scikit-surprise==1.0.6

# 処理を記述する
CMD echo "FINISH"
