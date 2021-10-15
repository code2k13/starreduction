FROM tensorflow/tensorflow:latest
WORKDIR /usr/src/app

RUN python3 -m pip install -U pip
RUN pip3 install pillow tqdm
RUN apt-get update 
RUN apt-get install git git-lfs -y
RUN git clone https://github.com/code2k13/starreduction


WORKDIR /usr/src/app/starreduction
RUN git-lfs pull
CMD ["bash"]