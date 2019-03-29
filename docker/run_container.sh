USER=$(whoami)
nvidia-docker run -it -u ${USER} \
	-v /home/${USER}/workspace:/home/${USER}/workspace \
	-v /raid:/raid \
	-v /usr/share/zoneinfo:/usr/share/zoneinfo \
        -p $1:$1 \
	-p $2:$2 \
	-e NVIDIA_VISIBLE_DEVICES=6,7 \
	--shm-size=128G \
	--name $3 \
	${USER}/pytorch-1.0:cuda10.0-cudnn7-dev-ubuntu16.04

#	jupyter lab --port=$2 --notebook-dir=~/workspace
#	-e NVIDIA_VISIBLE_DEVICES=$3 \

