//notchpeak:
//nvcc_wrapper -O3 --expt-extended-lambda -std=c++11 -I/uufs/chpc.utah.edu/sys/installdir/cuda/9.1.85/include -I$HOME/install/kokkos/cuda/include -L$HOME/install/kokkos/cuda/lib -lkokkos -lineinfo --expt-relaxed-constexpr sparse_matvec_v1_matching_cublas.cc -o sparse_matvec -g -fopenmp -L/usr/local/cuda/lib64 -lcublas -lcusparse -lcudart
//kingspeak
//~/install/kingspeak/kokkos/src/bin/nvcc_wrapper -O3 --expt-extended-lambda -std=c++11 -I/uufs/chpc.utah.edu/sys/installdir/cuda/9.1.85/include -I$HOME/install/kingspeak/kokkos/cuda/include -L$HOME/install/kingspeak/kokkos/cuda/lib -lkokkos -lineinfo --expt-relaxed-constexpr ../sparse_matvec_v1_matching_cublas.cc -o sparse_matvec -g -fopenmp -L/usr/local/cuda/lib64 -lcublas -lcusparse -lcudart

#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <sys/time.h>
#include <unistd.h>


#ifndef __device__
#define __device__
#define __host__
#define __forceinline__
#endif

#include "read_rb.h"
#include "simd.h"

#define N 256

// Array of vectors. (or a 2D array)
//typedef Kokkos::View<value_type**, Kokkos::LayoutRight, Kokkos::Cuda, Kokkos::MemoryTraits<Kokkos::Atomic>> automicvectortype;
typedef Kokkos::View<value_type**, Kokkos::LayoutRight, Kokkos::Cuda, Kokkos::MemoryTraits<Kokkos::RandomAccess>> automicvectortype;
typedef Kokkos::View<value_type**, Kokkos::LayoutRight, Kokkos::Cuda, Kokkos::MemoryTraits<Kokkos::RandomAccess>> vectortype;
typedef Kokkos::View<value_type**, Kokkos::LayoutRight, Kokkos::Cuda>::HostMirror vectortype_mirror;
typedef const typename Kokkos::TeamPolicy<Kokkos::Cuda>::member_type team_member;



template <int PVL, int LVL=N, const int EL=LVL/PVL>
void ___csr_simple_spmv___(const spm &M, const vectortype &V, automicvectortype &R, const int Blocks, const int ThreadsPerBlock)
{
	const Kokkos::TeamPolicy<Kokkos::Cuda> policy( Blocks , ThreadsPerBlock, PVL );

	//rows are distributed among teams. Each team processes 1 row in 1 iteration.
	Kokkos::parallel_for(policy, [=] __device__ (team_member & thread )
	{
		const int numberofthreads = thread.league_size() * thread.team_size();
		Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, ThreadsPerBlock), [=] __device__ (const int& thread_id)
		{
			Kokkos::parallel_for(Kokkos::ThreadVectorRange(thread, PVL), [=] __device__ (const int& vector_id)
			{
				int gthread_id = thread.league_rank() * thread.team_size() + thread_id;

				for(int row=gthread_id; row<M.m_num_of_rows; row=row+numberofthreads)	//iterate over rows. Each team processes 1 row at a time.
				{
					const int start = M.m_ptr_in_col_id_gpu(row);
					const int end = M.m_ptr_in_col_id_gpu(row+1);

					value_type temp[EL];
#pragma unroll(EL)
					for(int i=0; i<EL; i++)
						temp[i]=0.0;
					//main spmv loop
					for(int j= start; j<end; j++)
					{
						const int col = M.m_col_id_gpu(j);	//column
						const value_type a = M.m_csr_data_gpu(j);	//matrix data
#pragma unroll(EL)
						for(int i=0; i<EL; i++)
							temp[i] += V(col, i*PVL+vector_id)*a;
					}
#pragma unroll(EL)
					for(int i=0; i<EL; i++)
						R(row, i*PVL+vector_id) = temp[i];

				}
			});
		});
	});
}

template <int PVL, int LVL=N>	//LVL is not used in this routine. ALWAYS use PVL.
void ___csr_simd_spmv___(const spm &M, const vectortype &V1, automicvectortype &R1, const int Blocks, const int ThreadsPerBlock)
{
	const Kokkos::TeamPolicy<Kokkos::Cuda> policy( Blocks , ThreadsPerBlock, PVL );
	typedef simd<PVL, LVL> simd_type;
	typedef scalar_for_simd<LVL/PVL> Scalar;

	typedef Kokkos::View<simd_type*, Kokkos::LayoutRight, Kokkos::Cuda> simd_vectortype;
	simd_vectortype simd_V = simd_vectortype(reinterpret_cast<simd_type*>(V1.data()), V1.extent(0));
	simd_vectortype simd_R = simd_vectortype(reinterpret_cast<simd_type*>(R1.data()), R1.extent(0));

	//rows are distributed among teams. Each team processes 1 row in 1 iteration.
	Kokkos::parallel_for(policy, [=] __device__ (team_member & thread )
	{
		const int numberofthreads = thread.league_size() * thread.team_size();
		Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, ThreadsPerBlock), [=] __device__ (const int& thread_id)
		{
				int gthread_id = thread.league_rank() * thread.team_size() + thread_id;

				for(int row=gthread_id; row<M.m_num_of_rows; row=row+numberofthreads)	//iterate over rows. Each team processes 1 row at a time.
				{
					const int start = M.m_ptr_in_col_id_gpu(row);
					const int end = M.m_ptr_in_col_id_gpu(row+1);

					//main spmv loop
					Scalar temp = 0.0;
					for(int j= start; j<end; j++)
					{
						const int col = M.m_col_id_gpu(j);	//column
						const value_type a = M.m_csr_data_gpu(j);	//matrix data
						temp += simd_V(col)*a;
					}
					simd_R(row) = temp;
				}

		});
	});
}


inline double calc_error(int rows, vectortype_mirror R, value_type * R_cpu)
{
	double error=0.0;
#pragma omp parallel for schedule (dynamic) reduction(+ : error)
	for(int i=0; i<rows; i++)
		for(int j=0; j<N; j++)
			error += abs(R(i, j) - R_cpu[i*N+j]);
	error 			= error / rows / N;
	return error;
}

inline void resetR(int rows, automicvectortype &R)
{
	Kokkos::parallel_for(rows, [=] __device__ (int row)
	{
		for(int i=0;i<N;i++)
			R(row,i) = 0.0;
	});
}
template<int PVL, int LVL=-1>
void call_spmv(const spm &M, const vectortype &V_gpu, automicvectortype &R_gpu,vectortype_mirror R, value_type * R_cpu, const int Blocks, const int ThreadsPerBlock)
{
	if(PVL*ThreadsPerBlock >= 1024)
		return;

	printf("%d \t %d \t %d \t %d \t ",Blocks, ThreadsPerBlock, PVL, LVL);

	struct timeval  tv1, tv2;
	double simd_spmv_time=1000, simple_spmv_time=1000;
	double simd_spmv_error=0.0, simple_spmv_error=0.0;

	//___csr_tile1_spmv___
	resetR(M.m_num_of_rows, R_gpu);
	Kokkos::Cuda::fence();
	gettimeofday(&tv1, NULL);
	___csr_simple_spmv___<PVL, N>(M, V_gpu, R_gpu, Blocks, ThreadsPerBlock);
	Kokkos::Cuda::fence();
	gettimeofday(&tv2, NULL);
	simple_spmv_time = (double) (tv2.tv_usec - tv1.tv_usec) / 1000 + (double) (tv2.tv_sec - tv1.tv_sec)*1000;
	Kokkos::deep_copy(R, R_gpu);
	Kokkos::Cuda::fence();
	simple_spmv_error = calc_error(M.m_num_of_rows, R, R_cpu);


	resetR(M.m_num_of_rows, R_gpu);
	Kokkos::Cuda::fence();
	gettimeofday(&tv1, NULL);
	___csr_simd_spmv___<PVL, N>(M, V_gpu, R_gpu, Blocks, ThreadsPerBlock);
	Kokkos::Cuda::fence();
	gettimeofday(&tv2, NULL);
	simd_spmv_time = (double) (tv2.tv_usec - tv1.tv_usec) / 1000 + (double) (tv2.tv_sec - tv1.tv_sec)*1000;
	Kokkos::deep_copy(R, R_gpu);
	Kokkos::Cuda::fence();
	simd_spmv_error = calc_error(M.m_num_of_rows, R, R_cpu);

	/*for(int i=0; i<M.m_num_of_rows; i++)
	{
		for(int j=0;j<N;j++)
			printf("%f ",R(i,j));
		printf("\n");
	}*/
	printf("%f \t %f \t %f \t %f \n",
			simple_spmv_error, simd_spmv_error, simple_spmv_time, simd_spmv_time);
	fflush(stdout);

}

//https://docs.nvidia.com/cuda/cusparse/index.html#appendix-b-cusparse-library-c---example. check appendix A for example of cuparse and call spmv.
// M and V should be pointers to GPU memory with data already loaded. zHostPtr should pointer in cpu memory with memory allocated.
double cublas_spmv(const spm& M, value_type * V, value_type * R_cpu )
{

	value_type *zHostPtr    = (value_type *)malloc(N*(M.m_num_of_cols+1)*sizeof(value_type));	//allocate host memory for results
    cusparseHandle_t handle=0;
    cusparseMatDescr_t descr=0;
    cusparseStatus_t status;

    cusparseCreate(&handle);	// initialize cusparse library
    cusparseCreateMatDescr(&descr);	// create and setup matrix descriptor
    cusparseSetMatType(descr,CUSPARSE_MATRIX_TYPE_GENERAL);	//symmetric matrix
    cusparseSetMatIndexBase(descr,CUSPARSE_INDEX_BASE_ZERO);

    value_type dOne=1.0, dZero=0.0;
    value_type * z=0;

    cudaMalloc((void**)&z, N*(M.m_num_of_cols)*sizeof(value_type));
    cudaMemset((void *)z,0, N*(M.m_num_of_cols)*sizeof(value_type));

    cudaDeviceSynchronize();

    struct timeval  tv1, tv2;
	gettimeofday(&tv1, NULL);
	status = cusparseDcsrmm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, M.m_num_of_rows, N, M.m_num_of_cols,
		                               M.m_num_of_entries,
									   &dOne, //alpha. scalar used for multiplication. store 1 into a variable.
									   descr, M.m_csr_data_gpu.data(), M.m_ptr_in_col_id_gpu.data(), M.m_col_id_gpu.data(),
									   V, 	//B array of dimensions (ldb, n)
									   M.m_num_of_cols,	//ldb leading dimension of B. It must be at least max (1, k) if op ( A ) = A and at least max (1, m) otherwise. Try M.m_num_of_rows if this fails
									   &dZero, //beta scalar used for multiplication. If beta is zero, C does not have to be a valid input.
									   z,	// C array of dimensions (ldc, n)
									   M.m_num_of_rows);

    cudaDeviceSynchronize();
    gettimeofday(&tv2, NULL);

	if (status != CUSPARSE_STATUS_SUCCESS)
		printf("Matrix-matrix multiplication failed\n");

	double cublass_time = (double) (tv2.tv_usec - tv1.tv_usec) / 1000 + (double) (tv2.tv_sec - tv1.tv_sec)*1000;

    cudaMemcpy(zHostPtr, z, (size_t)(N*(M.m_num_of_cols)*sizeof(z[0])), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cusparseDestroyMatDescr(descr);	//destroy matrix descriptor
    cusparseDestroy(handle);	//destroy handle

    /*printf("cublas:\n");

	for(int i=0; i<M.m_num_of_rows; i++)
	{
		for(int j=0; j<N; j++)
		{
			//value_type a = i*N + j;
			printf("%f\t",zHostPtr[j*M.m_num_of_rows + i]);
		}
		printf("\n");
	}*/

    double error = 0.0;
#pragma omp parallel for schedule (dynamic) reduction(+ : error)
	for(int i=0; i<M.m_num_of_rows; i++)
		for(int j=0; j<N; j++)
			error += abs(R_cpu[i*N+j]- zHostPtr[j*M.m_num_of_rows + i]);

	error = error / M.m_num_of_rows / N;

	printf("cublas success, cublas_error: %f, cublas time: %f ms\n", error, cublass_time);
	fflush(stdout);

    cudaFree(z);
    free(zHostPtr);
    return cublass_time;
}


int main(int narg, char* args[])
{
	std::string filename(args[1]);
	//int Blocks = atoi(args[2]);
//	int ThreadsPerBlock = atoi(args[3]);
	//int VectorLength = atoi(args[4]);
	//int VectorLength = P_VECTOR_LENGTH;
	printf("input: %s\n",filename.data());
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

	//printf("Blocks %d, ThreadsPerBlock: %d\n",Blocks, ThreadsPerBlock);
	Kokkos::initialize(narg,args);
	{
		//----------------------------------  read matrix from rb file  ----------------------------------
		spm M = read_rb(filename.data());
		//print(M);

		printf("matrix read\n");
		size_t free_t,total_t;
		cudaMemGetInfo(&free_t,&total_t);

		printf("free: %lu, tot: %u\n", free_t/1024/1024/1024, total_t/1024/1024/1024);

		//----------------------------------  generate vectors.  ----------------------------------
		vectortype V_gpu("V", M.m_num_of_cols+1, N);		//N vectors of size num of cols in matrix. layout is compact batched layout
		vectortype_mirror V = Kokkos::create_mirror(V_gpu);
		printf("V allocated \n");
		cudaMemGetInfo(&free_t,&total_t);



#pragma omp parallel for
		for(int i=0; i<M.m_num_of_cols; i++)
		{
			for(int j=0; j<N; j++)
			{
				//V(i, j) = M_PI * a / (a+1.0);
				V(i, j) = i*N + j;
				//printf("%f\t",V(i,j));
			}
			//printf("\n");
		}

		//set last row to 0 hence even if col values are invalid, we will get 0.
		//this is to avoid checking if every time
		for(int j=0; j<N; j++)
			V(M.m_num_of_cols, j) = 0.0;
		Kokkos::deep_copy(V_gpu, V);


		//----------------------------------  cpu calc  ----------------------------------
		value_type R_cpu[M.m_num_of_rows*N];	//result vector for cpu.
#pragma omp parallel for
		for(int i=0; i<M.m_num_of_rows; i++)
			for(int j=0; j<N; j++)
				R_cpu[i*N+j] = 0.0;

#pragma omp parallel for
		for(int i=0; i<M.m_num_of_rows; i++)
		{
			const int row = i;
			for(int j=M.m_ptr_in_col_id(i); j<M.m_ptr_in_col_id(i+1); j++)
			{
				const int col = M.m_col_id(j);
				const value_type a = M.m_csr_data(j);	//matrix data

				for(int k=0; k<N; k++)
				{
					value_type x = V(col, k);
					R_cpu[row*N+k] += a*x;

					//symmetric matrix. hence interchanging row col and multiplying
					//value_type y = V(row, k);
					//R_cpu[col][k] += a*y;
				}
			}
		}



		/*printf("manual:\n");
		for(int i=0; i<M.m_num_of_rows; i++)
		{
			for(int j=0; j<N; j++)
			{
				//value_type a = i*N + j;
				printf("%f\t",R_cpu[i][j]);
			}
			printf("\n");
		}*/



		//----------------------------------  simple_spmv and simd_spmv  ----------------------------------
		{//local scope for function so that R_gpu will be deallocated

			printf("free: %lu, tot: %u\n", free_t/1024/1024/1024, total_t/1024/1024/1024);
			fflush(stdout);
			automicvectortype R_gpu("R", M.m_num_of_rows, N); //result vector for gpu.
			vectortype_mirror R = Kokkos::create_mirror(R_gpu);
			printf("R allocated \n");

			cudaMemGetInfo(&free_t,&total_t);

			printf("free: %lu, tot: %u\n", free_t/1024/1024/1024, total_t/1024/1024/1024);
			fflush(stdout);

			cudaMemset((void *)R_gpu.data(),0, N*(M.m_num_of_rows)*sizeof(value_type));


			//std::vector<int> BlocksVector = {4096, 8192, 16384, 32768};
			//std::vector<int> ThreadsPerBlockVector = {4, 8, 16, 32, 64, 128};
			//BlocksVector.push_back(std::min(2147483647, M.m_num_of_rows/128));
			std::vector<int> BlocksVector = {32768};	//best timings always among these many teams
			std::vector<int> ThreadsPerBlockVector = {8}; //best timings always among these many threads

			printf("Blocks \t ThreadsPerBlock \t PVL \t LVL \t simple_error \t simd_error \t simple_time \t simd_time \n");
			fflush(stdout);
			for(int i=0; i<BlocksVector.size(); i++)
			{
				int Blocks = BlocksVector[i];
				for(int j=0; j<ThreadsPerBlockVector.size(); j++)
				{
					int ThreadsPerBlock = ThreadsPerBlockVector[j];
					//call_spmv<8, N>(M, V_gpu, R_gpu, R, R_cpu, Blocks, ThreadsPerBlock);
					//call_spmv<16, N>(M, V_gpu, R_gpu, R, R_cpu, Blocks, ThreadsPerBlock);
					//call_spmv<32, N>(M, V_gpu, R_gpu, R, R_cpu, Blocks, ThreadsPerBlock);
					call_spmv<64, N>(M, V_gpu, R_gpu, R, R_cpu, Blocks, ThreadsPerBlock);
					//call_spmv<128, N>(M, V_gpu, R_gpu, R, R_cpu, Blocks, ThreadsPerBlock);
				}
			}

		}

		//----------------------------------  cublass  ----------------------------------

		printf("--------------------- calling cublas ------------------------\n");
		fflush(stdout);

		vectortype V1_gpu("V", N , M.m_num_of_cols);	//cublass wont work with compact form. Hence take transpose.

		Kokkos::parallel_for("___V_Transpose___",M.m_num_of_cols , [=] __device__ (int i)
		{
			for(int j=0; j<N; j++)
				V1_gpu(j, i) = V_gpu(i, j);
		});

		Kokkos::Cuda::fence();

		//for(int i=0; i<10; i++)
			cublas_spmv(M, V1_gpu.data(), R_cpu);


	}
	Kokkos::finalize();

	return(0);
}
