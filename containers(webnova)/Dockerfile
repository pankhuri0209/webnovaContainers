FROM ubuntu:focal
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install vim python3-dev python3-pip -y
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && pip install opencv-python-headless opencv-contrib-python-headless matplotlib
# RUN pip install imageio scikit-image
# RUN pip install scikit-learn
# RUN pip install matplotlib
RUN useradd -ms /bin/bash user1
WORKDIR /home/user1
RUN mkdir -p /home/user1/mission_config
RUN mkdir -p /home/user1/entry_point
RUN mkdir -p /home/user1/input_queue
RUN mkdir -p /home/user1/output_queue
ADD run_executable.sh /home/user1/entry_point
ADD container.py /home/user1/entry_point
ADD mission_config1712967621622.json /home/user1/mission_config
WORKDIR /home/user1/entry_point
CMD /home/user1/entry_point/run_executable.sh
