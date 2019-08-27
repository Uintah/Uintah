 //nvcc_wrapper -O2 --expt-extended-lambda -std=c++11 -I/uufs/chpc.utah.edu/sys/installdir/cuda/9.1.85/include -I$HOME/install/kokkos/cuda/include -L$HOME/install/kokkos/cuda/lib -lkokkos -lineinfo --expt-relaxed-constexpr sparse_matvec.cc -o sparse_matvec -O2
#include <stdio.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <sys/time.h>


#ifndef __device__
#define __device__
#endif

#include "read_rb.h"

#define N 256


// Array of vectors. (or a 2D array)
typedef Kokkos::View<value_type***, Kokkos::LayoutRight, Kokkos::Cuda> vectortype_simd;
typedef Kokkos::View<value_type**, Kokkos::LayoutRight, Kokkos::Cuda> vectortype;
typedef Kokkos::View<value_type***, Kokkos::LayoutRight, Kokkos::Cuda>::HostMirror vectortype_simd_mirror;
typedef Kokkos::View<value_type**, Kokkos::LayoutRight, Kokkos::Cuda>::HostMirror vectortype_mirror;
typedef const typename Kokkos::TeamPolicy<Kokkos::Cuda>::member_type team_member;


// logic from https://www.nvidia.com/docs/IO/66889/nvr-2008-004.pdf: Efficient Sparse Matrix-Vector Multiplication on CUDA (Nathan Bell and Michael Garland†)
// + using compact batched format
// + saving shared memory to make not limit N size. will not give performance boost

void ___csr_spmv___(const spm &M, const vectortype_simd &V, vectortype_simd &R, const int Blocks, const int ThreadsPerBlock, const int VectorLength)
{
	const Kokkos::TeamPolicy<Kokkos::Cuda> policy( Blocks , ThreadsPerBlock, VectorLength );
	const int shm_size = 32 * 1024;	//for now request all available shared memory. change it later for portability.

	Kokkos::parallel_for("___csc_spmv___", policy.set_scratch_size(0, Kokkos::PerTeam(shm_size)), [=] __device__ (team_member & thread )
	{
		const int blockId = thread.league_rank();
		const int blockSize = thread.league_size();
		const int N_by_simd = N / VectorLength;
		//*********  ASSUMING NNZ * 8 (FOR DATA) + NNZ * 4 (FOR COL) + BLOCK_SIZE * 8 (FOR RESULT) < 46 KB. **********************
		//get all the shared memory available
		unsigned char* all_shared_memory = (unsigned char*) thread.team_shmem().get_shmem(shm_size);

		//shared memory: each thread stores its temporary sums in shared memory. ThreadsPerBlock*VectorLength is block size
		value_type * R_shared = (value_type*) all_shared_memory;
		all_shared_memory = all_shared_memory + blockSize * sizeof(double);	//increment all_shared_memory pointer. block size will be align with 32. so should not cause bank conflicts


		//reset results and copy data into shared memory
		Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, ThreadsPerBlock), [=] __device__ (const int& thread_id)	//iterate over matrix rows
		{
			Kokkos::parallel_for(Kokkos::ThreadVectorRange(thread, VectorLength), [=] __device__ (const int& vector_id)
			{
				for(int row=blockId; row<M.m_num_of_rows; row=row+blockSize)	//iterate over columns. Each partition processes 1 column at a time.
				{
					const int start = M.m_ptr_in_col_id_gpu(row);
					const int end = M.m_ptr_in_col_id_gpu(row+1);
					const int nnz_elements = end - start;

					//allocate shared memory for matrix data and col indexes and copy those values
					int nnz_size_data = nnz_elements * sizeof(value_type);	//pad extra elements to avoid bank conflicts.
					int nnz_size_int  = nnz_elements * sizeof(int);
					nnz_size_data 	= nnz_size_data + (  (nnz_size_data%32==0) ? 0 : (32 - nnz_size_data%32) );
					nnz_size_int 	= nnz_size_int  + (  (nnz_size_int%32==0) ? 0 : (32 - nnz_size_int%32) );

					unsigned char * available_shared_mem = all_shared_memory;	//recycle from last row
					value_type * data_shared = (value_type*) available_shared_mem;
					available_shared_mem = available_shared_mem + nnz_size_data;
					int * col_id_shared = (int*) available_shared_mem;
					available_shared_mem = available_shared_mem + nnz_size_int;

					const int shared_id = thread_id * VectorLength + vector_id;

					R_shared[shared_id] = 0.0;	//reset results
#pragma unroll(4)
					for(int k= start + shared_id, l=shared_id; k<end; k=k+blockSize, l=l+blockSize)	//copy
					{
						data_shared[l] = M.m_csr_data_gpu(k);
						col_id_shared[l] = M.m_col_id_gpu(k);
					}

					__syncthreads();


					for(int i=0; i<N_by_simd; i++)
					{
#pragma unroll(4)
						for(int j= thread_id; j<nnz_elements; j=j+ThreadsPerBlock)
						{
							const int col = col_id_shared[j];	//add back M.m_ptr_in_row_id(i)
							const value_type a = data_shared[j];	//matrix data
							value_type x = V(i, col, vector_id);
							R_shared[shared_id] += a*x;
						}

						__syncthreads();

#pragma unroll(4)
						for(int next_thread = ThreadsPerBlock >> 1; next_thread > 0; next_thread = next_thread >> 1)	//divide next thread by 2 every time
						{
							if ( thread_id < next_thread)
								R_shared[ shared_id ] += R_shared [ shared_id + next_thread * VectorLength];
							__syncthreads();
						}

						if ( thread_id == 0)	// first thread writes the result
							R(i, row, vector_id) = R_shared [ shared_id ];
						R_shared [ shared_id ] = 0.0;
					}
				}
			});
		});
	});
}

//https://docs.nvidia.com/cuda/cusparse/index.html#appendix-b-cusparse-library-c---example. check appendix A for example of cuparse and call spmv.
// M and V should be pointers to GPU memory with data already loaded. zHostPtr should pointer in cpu memory with memory allocated.
double cublas_spmv(const spm& M, value_type * V, value_type *zHostPtr )
{
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
    cudaFree(z);
    return cublass_time;
}


int main(int narg, char* args[])
{
	std::string filename(args[1]);
	const int Blocks = atoi(args[2]);
	const int ThreadsPerBlock = atoi(args[3]);
	const int VectorLength = atoi(args[4]);
	const int N_by_simd = N / VectorLength;

	if(sizeof(value_type)==4)
		cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeFourByte);
	else if(sizeof(value_type)==8)
		cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);

	printf("Blocks %d, ThreadsPerBlock: %d, VectorLength:%d \n",Blocks, ThreadsPerBlock, VectorLength);
	Kokkos::initialize(narg,args);
	{
		//----------------------------------  read matrix from rb file  ----------------------------------
		spm M = read_rb(filename.data());
		//print(M);

		//----------------------------------  generate vectors.  ----------------------------------
		vectortype_simd V_gpu("V", N_by_simd, M.m_num_of_cols, VectorLength);		//N vectors of size num of cols in matrix. layout is compact batched layout
		vectortype_simd_mirror V = Kokkos::create_mirror(V_gpu);
		vectortype_simd R_gpu("R", N_by_simd, M.m_num_of_rows, VectorLength); //result vector for gpu.
		vectortype_simd_mirror R = Kokkos::create_mirror(R_gpu);

#pragma omp parallel for
		for(int h=0; h<N_by_simd; h++)
		{
			for(int i=0; i<M.m_num_of_cols; i++)
			{
				for(int j=0; j<VectorLength; j++)
				{
					int a = h * M.m_num_of_cols * VectorLength + i * VectorLength + j;
					//V(h, i, j) = M_PI * a / (a+1.0);
					V(h, i, j) = a;
					//printf("%f\t",V(h, i, j));
				}
				//printf("\n");
			}
			//printf("-------------------------------\n\n");
		}
		Kokkos::deep_copy(V_gpu, V);

		cudaMemset((void *)R_gpu.data(),0, N*(M.m_num_of_rows)*sizeof(value_type));


		//----------------------------------  simple_spmv  ----------------------------------
		Kokkos::Cuda::fence();

		struct timeval  tv1, tv2;
		gettimeofday(&tv1, NULL);

		___csr_spmv___(M, V_gpu, R_gpu, Blocks, ThreadsPerBlock, VectorLength);
		Kokkos::Cuda::fence();

		gettimeofday(&tv2, NULL);

		Kokkos::deep_copy(R, R_gpu);
		double simple_spmv_time = (double) (tv2.tv_usec - tv1.tv_usec) / 1000 + (double) (tv2.tv_sec - tv1.tv_sec)*1000;


		//----------------------------------  cublass  ----------------------------------

		value_type *zHostPtr    = (value_type *)malloc(N*(M.m_num_of_cols+1)*sizeof(value_type));	//allocate host memory for results
		vectortype V1_gpu("V", N , M.m_num_of_cols);	//cublass wont work with compact form. Hence take transpose.

		Kokkos::parallel_for("___V_Transpose___",N_by_simd , [=] __device__ (int h)
		{
			for(int i=0; i<M.m_num_of_cols; i++)
				for(int j=0; j<VectorLength; j++)
				{
					int k = h*VectorLength + j;
					V1_gpu(k, i) = V_gpu(h, i, j);
				}

		});

		Kokkos::Cuda::fence();

		double cublass_time = cublas_spmv(M, V1_gpu.data(), zHostPtr);



		//----------------------------------  verify  ----------------------------------
		value_type R_cpu[M.m_num_of_rows][N];	//result vector for cpu.
#pragma omp parallel for
		for(int i=0; i<M.m_num_of_rows; i++)
			for(int j=0; j<N; j++)
				R_cpu[i][j] = 0.0;

#pragma omp parallel for
		for(int i=0; i<M.m_num_of_rows; i++)
		{
			const int row = i;
			for(int j=M.m_ptr_in_col_id(i); j<M.m_ptr_in_col_id(i+1); j++)
			{
				const int col = M.m_col_id(j);
				const value_type a = M.m_csr_data(j);	//matrix data

				for(int h=0; h<N_by_simd; h++)	//this access pattern of V is very bad. but ok for verification
					for(int k=0; k<VectorLength; k++)
					{
						value_type x = V(h, col, k);
						R_cpu[row][h*VectorLength + k] += a*x;

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


		double error=0.0, cublas_error=0.0, relative_error=0.0;

#pragma omp parallel for schedule (dynamic) reduction(+ : error) reduction(+ : cublas_error) reduction(+ : relative_error)
		for(int i=0; i<M.m_num_of_rows; i++)
			for(int j=0; j<N; j++)
			{
				error += abs(R(j/VectorLength, i, j%VectorLength) - R_cpu[i][j]);
				cublas_error += abs(R_cpu[i][j]- zHostPtr[j*M.m_num_of_rows + i]);
				relative_error += abs(R(j/VectorLength, i, j%VectorLength) - zHostPtr[j*M.m_num_of_rows + i]);
			}

		relative_error	= relative_error / M.m_num_of_rows / N;
		error 			= error / M.m_num_of_rows / N;
		cublas_error 	= cublas_error / M.m_num_of_rows / N;

		printf("success, custom error: %f, cublas_error: %f, relative_error: %f\n", error, cublas_error, relative_error);
		printf("simple_spmv time: %f ms, cublas time: %f ms\n\n\n", simple_spmv_time, cublass_time);


		free(zHostPtr);

	}
	Kokkos::finalize();

	return(0);
}



