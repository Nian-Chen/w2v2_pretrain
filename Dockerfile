FROM centos:7.9.2009
#RUN yum -y install wget && \
#    wget https://repo.anaconda.com/archive/Anaconda3-2020.07-Linux-x86_64.sh && \
#    sh Anaconda3-2020.07-Linux-x86_64.sh -p anaconda3 -b && \
#    echo 'export PATH="anaconda3/bin:$PATH"' >> ~/.bashrc && \
#    . ~/.bashrc
#COPY Anaconda3-2020.07-Linux-x86_64.sh .
#COPY hf_pack.tar.gz .
COPY . .
RUN yum -y install {vim,wget,sox,make,gcc} && \
    tar zxvf yasm-1.3.0.tar.gz && \
    cd yasm-1.3.0 && \
    ./configure && \
    make && make install && cd .. && \
    tar -zxvf ffmpeg-3.4.4.tar.gz && \
    cd ffmpeg-3.4.4 && \
    ./configure --enable-ffplay --enable-ffserver && \
    make && make install && cd .. && \
    sh Anaconda3-2020.07-Linux-x86_64.sh -p anaconda3 -b && \
    mkdir -p anaconda3/envs/huggingface && \
    tar -zxvf hf_pack.tar.gz -C anaconda3/envs/huggingface
ENV PATH anaconda3/bin:$PATH
ENV LD_LIBRARY_PATH anaconda3/bin:$PATH
CMD ["/bin/bash"]
#echo 'export PATH="anaconda3/bin:$PATH"' >> ~/.bashrc && \
#echo 'export LD_LIBRARY_PATH="/usr/lib64:anaconda3/lib:$LD_LIBRARY_PATH"' >> ~/.bashrc && \
#. ~/.bashrc && \
#conda init bash && \