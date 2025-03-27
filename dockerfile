FROM freqtradeorg/freqtrade:stable

COPY requirements.txt /freqtrade/
RUN pip install -r /freqtrade/requirements.txt