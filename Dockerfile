FROM ufoym/deepo:all-py27-cpu
WORKDIR /working
RUN git clone -b production https://github.com/sw6-aau/LSTnet-demo.git
WORKDIR /working/LSTnet-demo/
RUN git checkout production
RUN mkdir log/ save/ data/
RUN wget -O data/exchange_rate.txt https://sembrik.s3.eu-west-2.amazonaws.com/sw6/exchange_rate.txt
COPY . /working/LSTnet-demo/
RUN pip install -r requirements.txt
ENTRYPOINT ["python"]
CMD ["app.py"]
