USER=$(whoami)
sudo nvidia-docker run -it -u ${USER} \
	-v /home/${USER}/workspace:/home/${USER}/workspace \
	-v /raid:/raid \
	-v /usr/share/zoneinfo:/usr/share/zoneinfo \
	--ipc=host \
        --net=host \
	--name ${USER} \
	${USER}/pytorch-1.0:cuda10.0-cudnn7-dev-ubuntu16.04

#	jupyter lab --port=$2 --notebook-dir=~/workspace
#	-e NVIDIA_VISIBLE_DEVICES=$3 \

