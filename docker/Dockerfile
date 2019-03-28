FROM nvidia/cuda:10.0-cudnn7-devel-ubuntu16.04
MAINTAINER Soonmin Hwang <jjang9hsm@gmail.com>

RUN apt-get update && apt-get install -y --no-install-recommends \
         build-essential \
         cmake \
         git \
         curl \
         vim \
         ca-certificates \
         libjpeg-dev \
         libpng-dev &&\
     rm -rf /var/lib/apt/lists/*

# 1. Add $USER_NAME
ARG UID
ARG USER_NAME
RUN apt-get update && apt-get install -y sudo && \
    adduser $USER_NAME -u $UID --quiet --gecos "" --disabled-password && \
    echo "$USER_NAME ALL=(root) NOPASSWD:ALL" > /etc/sudoers.d/$USER_NAME && \
    chmod 0440 /etc/sudoers.d/$USER_NAME

# Below commands will be executed as $USER_NAME
USER $USER_NAME

# 2. Install anaconda for python 3.7 & PyTorch 1.0
RUN curl -o /home/$USER_NAME/anaconda.sh -O https://repo.anaconda.com/archive/Anaconda3-2018.12-Linux-x86_64.sh
RUN chmod +x /home/$USER_NAME/anaconda.sh && \
     /home/$USER_NAME/anaconda.sh -b -p /home/$USER_NAME/anaconda3 && \
     /home/$USER_NAME/anaconda3/bin/conda install numpy pyyaml mkl mkl-include setuptools cmake cffi typing ipython && \
     /home/$USER_NAME/anaconda3/bin/conda install pytorch torchvision cudatoolkit=10.0 -c pytorch && \
     /home/$USER_NAME/anaconda3/bin/conda install opencv && \
     /home/$USER_NAME/anaconda3/bin/conda clean -ya

ENV PATH /home/$USER_NAME/anaconda3/bin:${PATH}

RUN echo "cd ~" >> ~/.bashrc

# 3. Install misc
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    unzip && \
    rm -rf /var/lib/apt/lists/*

# 4. Jupyter lab settings
#USER $USER_NAME
#ARG JUPYTER_PASSWORD
# Jupyter notebook settings except port
#RUN jupyter notebook --generate-config

# Enable jupyter lab
#RUN jupyter serverextension enable --py jupyterlab --sys-prefix

# Install plotly for advanced visualization on jupyter lab
#RUN pip install plotly && \
#    conda install -c conda-forge nodejs && \
#    jupyter labextension install @jupyterlab/plotly-extension

#RUN jupyter_sha=$(python -c "from notebook.auth import passwd; print(passwd('${JUPYTER_PASSWORD}'))") && \
#    echo "c.NotebookApp.password=u'$jupyter_sha'" >> ~/.jupyter/jupyter_notebook_config.py && \
#    echo "c.NotebookApp.ip='0.0.0.0'" >> ~/.jupyter/jupyter_notebook_config.py && \
#    echo "c.NotebookApp.open_browser=False" >> ~/.jupyter/jupyter_notebook_config.py && \
#    echo "c.NotebookApp.terminado_settings = { 'shell_command': ['bash'] }" >> ~/.jupyter/jupyter_notebook_config.py

# 5. Install tensorflow, tensorboard, tensorboardX
RUN pip install tensorflow tensorboard tensorboardX
