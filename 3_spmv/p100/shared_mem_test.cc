#include <Kokkos_Core.hpp>

#define N 256
#ifndef __device__
#define __device__
#define __global__
#endif

typedef double value_type;

typedef Kokkos::View<value_type**, Kokkos::LayoutRight, Kokkos::Cuda> vectortype;
typedef const typename Kokkos::TeamPolicy<Kokkos::Cuda>::member_type team_member;

void _shared_mem_test__kokkos (const int Blocks, const int ThreadsPerBlock, const int VectorLength)
{
	const Kokkos::TeamPolicy<Kokkos::Cuda> policy( Blocks , ThreadsPerBlock, VectorLength );
	const int shm_size = ThreadsPerBlock*N;
	vectortype temp("V", ThreadsPerBlock, N);

	Kokkos::parallel_for(policy.set_scratch_size(0, Kokkos::PerTeam(shm_size*sizeof(value_type))), [=] __device__ (team_member & thread )
	{
		const int blockId = thread.league_rank();
		const int CudaThreadPerBlock = ThreadsPerBlock*VectorLength;


		//shared memory: each thread stores its temporary sums in shared memory
		value_type * R_shared = (value_type*) thread.team_shmem().get_shmem(shm_size*sizeof(value_type));

		Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, ThreadsPerBlock), [=] __device__ (const int& thread_id)
		{
			Kokkos::parallel_for(Kokkos::ThreadVectorRange(thread, VectorLength), [=] __device__ (const int& vector_id)
			{
				//const int gshared_id = thread_id * VectorLength + vector_id;
				const int gshared_id = blockDim.x * threadIdx.y + threadIdx.x;

				for(int i = gshared_id; i<shm_size; i=i+CudaThreadPerBlock)
					R_shared[i] = i;

				__syncthreads();

				for(int i = gshared_id; i<shm_size; i=i+CudaThreadPerBlock)
				{
					//thread_id vector_id warp_id lane_id index bank_id
					//printf("%d %d %d %d %d %d\n",thread_id, vector_id, gshared_id/32, gshared_id%32, i, i%32);
					temp(i/N, i%N) = R_shared[i];
					__syncthreads();
				}

				__syncthreads();

				if(blockId == 0 && gshared_id==0)
					printf("%f\n", temp(0,0));
			});
		});
	});
}

__global__ void _shared_mem_test__cuda()
{
	int ThreadsPerBlock = blockDim.y, VectorLength = blockDim.x;
	const int shm_size = ThreadsPerBlock*N;
	value_type * temp = (value_type *)malloc(shm_size * sizeof(value_type));
	const int CudaThreadPerBlock = ThreadsPerBlock*VectorLength;

	//shared memory: each thread stores its temporary sums in shared memory
	extern __shared__ value_type R_shared[];
	int thread_id = threadIdx.y;
	int vector_id = threadIdx.x;

	if(thread_id==0 && vector_id ==0)
		printf("in cuda blockdim.x: %d, blockDim.y: %d:\n", blockDim.x, blockDim.y);

	//const int gshared_id = thread_id * VectorLength + vector_id;
	const int gshared_id = blockDim.x * threadIdx.y + threadIdx.x;

	for(int i = gshared_id; i<shm_size; i=i+CudaThreadPerBlock)
		R_shared[i] = i;

	__syncthreads();

	for(int i = gshared_id; i<shm_size; i=i+CudaThreadPerBlock)
	{
		//thread_id vector_id warp_id lane_id index bank_id
		//printf("%d %d %d %d %d %d\n",thread_id, vector_id, gshared_id/32, gshared_id%32, i, i%32);
		__syncthreads();
		temp[i] = R_shared[i];

	}

	__syncthreads();

	if(blockIdx.x == 0 && gshared_id==0)
		printf("%f\n", temp[0]);

	free(temp);

}

int main(int narg, char* args[])
{
	std::string filename(args[1]);
	int Blocks = atoi(args[2]);
	int ThreadsPerBlock = atoi(args[3]);
	int VectorLength = atoi(args[4]);

	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

	printf("Blocks %d, ThreadsPerBlock: %d, VectorLength:%d \n",Blocks, ThreadsPerBlock, VectorLength);
	printf("kokkos kernel:\n");
	Kokkos::initialize(narg,args);
	{
		_shared_mem_test__kokkos(Blocks, ThreadsPerBlock, VectorLength);
	}
	Kokkos::finalize();

	const int shm_size = ThreadsPerBlock*N*sizeof(value_type);

	printf("cuda kernel:\n");
	dim3 blocks(1,1), threadsperblock(VectorLength, ThreadsPerBlock);
	_shared_mem_test__cuda<<<blocks, threadsperblock, shm_size>>>();
	return(0);
}
