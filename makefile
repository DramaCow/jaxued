# NVCC_RESULT := $(shell which nvcc 2> NULL; rm NULL)
# NVCC_TEST := $(notdir $(NVCC_RESULT))
# ifeq ($(NVCC_TEST),nvcc)
GPUS='"device=1"'
# else
# GPUS=
# endif


# Set flag for docker run command
MYUSER=nmonette
WANDB_API_KEY=$(shell cat ./wandb_key)
BASE_FLAGS=--rm -v ${PWD}:/home/duser/uedfomo --shm-size 20G
RUN_FLAGS=--gpus $(GPUS) $(BASE_FLAGS) -e WANDB_API_KEY=$(WANDB_API_KEY)

DOCKER_IMAGE_NAME = $(MYUSER)-ncc-craftax
IMAGE = $(DOCKER_IMAGE_NAME):latest
DOCKER_RUN=docker run $(RUN_FLAGS) $(IMAGE)
USE_CUDA = $(if $(GPUS),true,false)
ID = $(shell id -u)

# make file commands
build:
	DOCKER_BUILDKIT=1 docker build --build-arg USE_CUDA=$(USE_CUDA) --build-arg UID=$(ID) --build-arg GID=1234 --build-arg REQS="$(shell cat ./requirements.txt | tr '\n' ' ')" --tag $(IMAGE) --progress=plain ${PWD}/.

run:
	docker run -it $(RUN_FLAGS) $(IMAGE) /bin/bash

# Start WandB sweep agents
sweep:
	@if [ -z "$(SWEEP_ID)" ]; then \
		echo "Error: SWEEP_ID is required. Usage: make sweep SWEEP_ID=<entity/project/sweep_id>"; \
		exit 1; \
	fi
	@echo "Starting WandB sweep with ID: $(SWEEP_ID)"
	docker run -d $(RUN_FLAGS) $(IMAGE) wandb agent $(SWEEP_ID)