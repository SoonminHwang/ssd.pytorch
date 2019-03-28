USERNAME=$(whoami)
sudo nvidia-docker build -t \
	${USERNAME}/pytorch-1.0:cuda10.0-cudnn7-dev-ubuntu16.04 \
	--build-arg UID=$(id -u) \
	--build-arg USER_NAME=${USERNAME} \
	.

#	--build-arg JUPYTER_PASSWORD=dream1005 \
