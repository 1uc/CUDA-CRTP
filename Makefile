CUDA_ARCH := -arch=sm_30

all: with-user-cast no-user-cast

with-user-cast: main.cu
	nvcc $(CUDA_ARCH) -DHAS_USER_DEFINED_CAST $< -o $@

no-user-cast: main.cu
	nvcc $(CUDA_ARCH) $< -o $@

clean:
	rm with-user-cast no-user-cast
