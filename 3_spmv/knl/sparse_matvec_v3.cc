//notchpeak:
//nvcc_wrapper -O3 --expt-extended-lambda -std=c++11 -I/uufs/chpc.utah.edu/sys/installdir/cuda/9.1.85/include -I$HOME/install/kokkos/cuda/include -L$HOME/install/kokkos/cuda/lib -lkokkos -lineinfo --expt-relaxed-constexpr sparse_matvec_v1_matching_cublas.cc -o sparse_matvec -g -fopenmp -L/usr/local/cuda/lib64 -lcublas -lcusparse -lcudart
//kingspeak
//~/install/kingspeak/kokkos/src/bin/nvcc_wrapper -O3 --expt-extended-lambda -std=c++11 -I/uufs/chpc.utah.edu/sys/installdir/cuda/9.1.85/include -I$HOME/install/kingspeak/kokkos/cuda/include -L$HOME/install/kingspeak/kokkos/cuda/lib -lkokkos -lineinfo --expt-relaxed-constexpr ../sparse_matvec_v1_matching_cublas.cc -o sparse_matvec -g -fopenmp -L/usr/local/cuda/lib64 -lcublas -lcusparse -lcudart
//cthulhu (KNL)
//icpc ./sparse_matvec_v3.cc -fopenmp -std=c++11 -xMIC-AVX512 -I$HOME/KNL/uintah/host-O2/TPLs/include -L$HOME/KNL/uintah/host-O2/TPLs/lib -lkokkos -L/home/sci/damodars/installs/hwloc/install/lib -lhwloc -o 3_spmv

#include <stdio.h>
#include <math.h>
#include <sys/time.h>
#include <unistd.h>
#include <omp.h>
#include <ittnotify.h>

#if 0
#include <cuda_runtime.h>
#include <cusparse.h>
#define mirror Kokkos::create_mirror_view
#else
#include<mkl_spblas.h>
#endif

#ifndef __device__
#define __device__
#define __host__
#define __forceinline__
#define mirror
#endif



#include "read_rb.h"
#include "simd.h"


#define N 256
#define REPEAT 1


// Array of vectors. (or a 2D array)

//typedef Kokkos::Cuda execution_space;


typedef Kokkos::View<value_type**, Kokkos::LayoutRight, execution_space, Kokkos::MemoryTraits<Kokkos::RandomAccess>> automicvectortype;
typedef Kokkos::View<value_type**, Kokkos::LayoutRight, execution_space, Kokkos::MemoryTraits<Kokkos::RandomAccess>> vectortype;
typedef Kokkos::View<value_type**, Kokkos::LayoutRight, execution_space>::HostMirror vectortype_mirror;
typedef const typename Kokkos::TeamPolicy<execution_space>::member_type team_member;

/*

template <int PVL, int LVL=N, const int EL=LVL/PVL>
void ___csr_simple_spmv___(const spm &M, const vectortype &V, automicvectortype &R, const int Blocks, const int ThreadsPerBlock)
{
	const Kokkos::TeamPolicy<execution_space> policy( Blocks , ThreadsPerBlock, PVL );

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

*/

template <int PVL, int LVL=N, const int EL=LVL/PVL>
void ___csr_simple_spmv___(const spm &M, const vectortype &V, automicvectortype &R, const int Blocks, const int ThreadsPerBlock)
{
	const Kokkos::TeamPolicy<execution_space> policy( Blocks , ThreadsPerBlock);

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
					for(int j= start; j<end; j++)
					{
						const int col = M.m_col_id_gpu(j);	//column
						const value_type a = M.m_csr_data_gpu(j);	//matrix data
#pragma simd
						for(int i=0; i<N; i++)
							 R(row, i) += V(col, i)*a;
					}

				}

		});
	});
}


template <int PVL, int LVL=N, const int EL=LVL/PVL>
void ___csr_simple1_spmv___(const spm &M, const vectortype &V, automicvectortype &R, const int Blocks, const int ThreadsPerBlock)
{
	const Kokkos::TeamPolicy<execution_space> policy( Blocks , ThreadsPerBlock );

	//rows are distributed among teams. Each team processes 1 row in 1 iteration.
	Kokkos::parallel_for(policy, [=] __device__ (team_member & thread )
	{
		const int numberofthreads = thread.league_size() * thread.team_size();
		Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, ThreadsPerBlock), [=] __device__ (const int& thread_id)
		{

				int gthread_id = thread.league_rank() * thread.team_size() + thread_id;
				int rows_per_thread = M.m_num_of_rows / numberofthreads + 1;
				int rows_start = gthread_id * rows_per_thread;
				int rows_end = std::min(rows_start + rows_per_thread, M.m_num_of_rows);
				//for(int row=gthread_id; row<M.m_num_of_rows; row=row+numberofthreads)	//iterate over rows. Each team processes 1 row at a time.
				for(int row=rows_start; row<rows_end; row=row+1)	//iterate over rows. Each team processes 1 row at a time.
				{
					const int start = M.m_ptr_in_col_id_gpu(row);
					const int end = M.m_ptr_in_col_id_gpu(row+1);


					//main spmv loop
					for(int j= start; j<end; j++)
					{
						const int col = M.m_col_id_gpu(j);	//column
						const value_type a = M.m_csr_data_gpu(j);	//matrix data
#pragma simd
						for(int i=0; i<N; i++)
							 R(row, i) += V(col, i)*a;
					}

				}

		});
	});
}


template <int PVL, int LVL=N>	//LVL is not used in this routine. ALWAYS use PVL.
void ___csr_simd_spmv___(const spm &M, const vectortype &V1, automicvectortype &R1, const int Blocks, const int ThreadsPerBlock)
{
	const Kokkos::TeamPolicy<execution_space> policy( Blocks , ThreadsPerBlock );
	typedef simd<PVL, LVL> simd_type;
	typedef scalar_for_simd<PVL, LVL> Scalar;

	typedef Kokkos::View<simd_type*, Kokkos::LayoutRight, execution_space> simd_vectortype;
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
					//temp += simd_V(col)*a;
					temp.fma_self(simd_V(col),a);
				}
				simd_R(row) = temp;
			}

		});
	});
}


template <int PVL, int LVL=N>	//LVL is not used in this routine. ALWAYS use PVL.
void ___csr_simd1_spmv___(const spm &M, const vectortype &V1, automicvectortype &R1, const int Blocks, const int ThreadsPerBlock)
{
	const Kokkos::TeamPolicy<execution_space> policy( Blocks , ThreadsPerBlock );
	typedef simd<PVL, LVL> simd_type;
	typedef scalar_for_simd<PVL, LVL> Scalar;

	typedef Kokkos::View<simd_type*, Kokkos::LayoutRight, execution_space> simd_vectortype;
	simd_vectortype simd_V = simd_vectortype(reinterpret_cast<simd_type*>(V1.data()), V1.extent(0));
	simd_vectortype simd_R = simd_vectortype(reinterpret_cast<simd_type*>(R1.data()), R1.extent(0));

	//rows are distributed among teams. Each team processes 1 row in 1 iteration.
	Kokkos::parallel_for(policy, [=] __device__ (team_member & thread )
	{
		const int numberofthreads = thread.league_size() * thread.team_size();
		Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, ThreadsPerBlock), [=] __device__ (const int& thread_id)
		{
			int gthread_id = thread.league_rank() * thread.team_size() + thread_id;
			int rows_per_thread = M.m_num_of_rows / numberofthreads + 1;
			int rows_start = gthread_id * rows_per_thread;
			int rows_end = std::min(rows_start + rows_per_thread, M.m_num_of_rows);
			//for(int row=gthread_id; row<M.m_num_of_rows; row=row+numberofthreads)	//iterate over rows. Each team processes 1 row at a time.
			for(int row=rows_start; row<rows_end; row=row+1)	//iterate over rows. Each team processes 1 row at a time.
			{
				const int start = M.m_ptr_in_col_id_gpu(row);
				const int end = M.m_ptr_in_col_id_gpu(row+1);

				//main spmv loop
				Scalar temp = 0.0;
				for(int j= start; j<end; j++)
				{
					const int col = M.m_col_id_gpu(j);	//column
					const value_type a = M.m_csr_data_gpu(j);	//matrix data
					//temp += simd_V(col)*a;
					temp.fma_self(simd_V(col),a);
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
template<int PVL, int LVL=N>
void call_spmv(const spm &M, const vectortype &V_gpu, automicvectortype &R_gpu,vectortype_mirror R, value_type * R_cpu, const int Blocks, const int ThreadsPerBlock)
{
	//if(PVL*ThreadsPerBlock >= 1024)
	//	return;

	printf("%d \t %d \t %d \t %d \t ",Blocks, ThreadsPerBlock, PVL, LVL);
	fflush(stdout);

	struct timeval  tv1, tv2;
	double simd_spmv_time=0.0, simd1_spmv_time=0.0, simple_spmv_time=0.0,simple1_spmv_time=0.0;
	double simd_spmv_error=0.0, simd1_spmv_error=0.0, simple_spmv_error=0.0, simple1_spmv_error=0.0;

	/*
	for(int i=-2;i<REPEAT;i++)
	{
		resetR(M.m_num_of_rows, R_gpu);
		execution_space::fence();
		gettimeofday(&tv1, NULL);
		___csr_simple_spmv___<PVL, N>(M, V_gpu, R_gpu, Blocks, ThreadsPerBlock);
		execution_space::fence();
		gettimeofday(&tv2, NULL);
		if(i>=0)
			simple_spmv_time += (double) (tv2.tv_usec - tv1.tv_usec) / 1000 + (double) (tv2.tv_sec - tv1.tv_sec)*1000;
	}
	simple_spmv_time /= (double)REPEAT;
	Kokkos::deep_copy(R, R_gpu);
	execution_space::fence();
	simple_spmv_error = calc_error(M.m_num_of_rows, R, R_cpu);
	*/



	for(int i=0;i<REPEAT;i++)
	{
		resetR(M.m_num_of_rows, R_gpu);
		execution_space::fence();
		gettimeofday(&tv1, NULL);
		___csr_simple1_spmv___<PVL, N>(M, V_gpu, R_gpu, Blocks, ThreadsPerBlock);
		execution_space::fence();
		gettimeofday(&tv2, NULL);
		if(i>=0)
			simple1_spmv_time += (double) (tv2.tv_usec - tv1.tv_usec) / 1000 + (double) (tv2.tv_sec - tv1.tv_sec)*1000;
	}
	simple1_spmv_time /= (double)REPEAT;
	Kokkos::deep_copy(R, R_gpu);
	execution_space::fence();
	simple1_spmv_error = calc_error(M.m_num_of_rows, R, R_cpu);




	/*
	for(int i=-2;i<REPEAT;i++)
	{
		resetR(M.m_num_of_rows, R_gpu);
		execution_space::fence();
		gettimeofday(&tv1, NULL);
		___csr_simd_spmv___<PVL, N>(M, V_gpu, R_gpu, Blocks, ThreadsPerBlock);
		execution_space::fence();
		gettimeofday(&tv2, NULL);
		if(i>=0)
			simd_spmv_time += (double) (tv2.tv_usec - tv1.tv_usec) / 1000 + (double) (tv2.tv_sec - tv1.tv_sec)*1000;
	}
	simd_spmv_time /= (double)REPEAT;
	Kokkos::deep_copy(R, R_gpu);
	execution_space::fence();
	simd_spmv_error = calc_error(M.m_num_of_rows, R, R_cpu);
	*/

	for(int i=0;i<REPEAT;i++)
	{
		resetR(M.m_num_of_rows, R_gpu);
		execution_space::fence();
		gettimeofday(&tv1, NULL);
		___csr_simd1_spmv___<PVL, N>(M, V_gpu, R_gpu, Blocks, ThreadsPerBlock);
		execution_space::fence();
		gettimeofday(&tv2, NULL);
		if(i>=0)
			simd1_spmv_time += (double) (tv2.tv_usec - tv1.tv_usec) / 1000 + (double) (tv2.tv_sec - tv1.tv_sec)*1000;
	}
	simd1_spmv_time /= (double)REPEAT;
	Kokkos::deep_copy(R, R_gpu);
	execution_space::fence();
	simd1_spmv_error = calc_error(M.m_num_of_rows, R, R_cpu);



	/*for(int i=0; i<M.m_num_of_rows; i++)
	{
		for(int j=0;j<N;j++)
			printf("%f ",R(i,j));
		printf("\n");
	}*/
	printf("%f \t %f \t %f \t %f \t %f \t %f \t %f \t %f\n",
			simple_spmv_error, simple1_spmv_error,  simd_spmv_error, simd1_spmv_error, simple_spmv_time, simple1_spmv_time, simd_spmv_time, simd1_spmv_time);

	fflush(stdout);

}



#if 0
//https://docs.nvidia.com/cuda/cusparse/index.html#appendix-b-cusparse-library-c---example. check appendix A for example of cuparse and call spmv.
// M and V should be pointers to GPU memory with data already loaded. zHostPtr should pointer in cpu memory with memory allocated.
double blas_spmv(const spm& M, value_type * V, value_type * R_cpu )
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
	status = cusparseScsrmm(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, M.m_num_of_rows, N, M.m_num_of_cols,
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
#else

double blas_spmv(const spm& M, value_type * V, value_type * R_cpu )
{

	sparse_matrix_t        A;
	int *rows_end = new int[M.m_num_of_rows];
	value_type *R = new value_type[M.m_num_of_rows*N];

#pragma omp parallel for
	for(int i=0; i<M.m_num_of_rows;i++)
		rows_end[i] = M.m_ptr_in_col_id(i+1);

	sparse_status_t stat = mkl_sparse_d_create_csr( &A,
							 SPARSE_INDEX_BASE_ZERO, //0 based of 1 based
	                         M.m_num_of_rows,	//rows
							 M.m_num_of_cols,	//cols
	                         M.m_ptr_in_col_id.data(),	//rows_start
	                         rows_end,			//rows_end
	                         M.m_col_id.data(),	 	//col_indx,
	                         M.m_csr_data.data()		//values
							 );

	if(stat != SPARSE_STATUS_SUCCESS)
	{
		printf("failed to create spm handle: %d\n", stat);
		exit(1);
	}

	struct timeval  tv1, tv2;
	double mkl_time = 0.0;
	for(int i=-2;i<REPEAT;i++)
	{
		gettimeofday(&tv1, NULL);

		stat = mkl_sparse_d_mm( SPARSE_OPERATION_NON_TRANSPOSE, //    operation,
									 1.0,	                 //alpha,
									 A,			//sparse_matrix_t A
									 {SPARSE_MATRIX_TYPE_GENERAL, SPARSE_FILL_MODE_LOWER, SPARSE_DIAG_NON_UNIT}, //struct matrix_descr    descr,
									 SPARSE_LAYOUT_ROW_MAJOR, 		//	 sparse_layout_t        layout,
									 V,	 // const  double  *x,
									 N, 		//MKL_INT        columns,
									 M.m_num_of_cols,	 //MKL_INT        ldx,
									 0.0,	 //double         beta,
									 R, //double         *y,
									 N );//MKL_INT        ldy );
		execution_space::fence();
		gettimeofday(&tv2, NULL);
	    if(stat != SPARSE_STATUS_SUCCESS)
	    {
	    		printf("spmm failed : %d\n", stat);
	    		exit(1);
	    }

		mkl_time += (double) (tv2.tv_usec - tv1.tv_usec) / 1000 + (double) (tv2.tv_sec - tv1.tv_sec)*1000;
	}
	mkl_time /= (double)REPEAT;

    double error = 0.0;
#pragma omp parallel for schedule (dynamic) reduction(+ : error)
	for(int i=0; i<M.m_num_of_rows; i++)
		for(int j=0; j<N; j++)
			error += abs(R_cpu[i*N+j]- R[i*N + j]);

	error = error / M.m_num_of_rows / N;

	printf("mkl success, mkl_error: %f, mkl time: %f ms\n", error, mkl_time);
	fflush(stdout);


    delete []R;
	delete []rows_end;
	return 0;
}




#endif

int main(int narg, char* args[])
{

	__itt_pause();
	std::string filename(args[1]);
	//int Blocks = atoi(args[2]);
//	int ThreadsPerBlock = atoi(args[3]);
	//int VectorLength = atoi(args[4]);
	//int VectorLength = P_VECTOR_LENGTH;
	printf("input: %s\n",filename.data());
	int omp_threads = omp_get_max_threads();

#if 0
	cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte);
#endif
	//printf("Blocks %d, ThreadsPerBlock: %d\n",Blocks, ThreadsPerBlock);
	Kokkos::initialize(narg,args);
	{
		//----------------------------------  read matrix from rb file  ----------------------------------
		spm M = read_rb(filename.data());
		//print(M);
		printf("%s %d\n", __FILE__, __LINE__);

		//----------------------------------  generate vectors.  ----------------------------------
		vectortype V_gpu("V", M.m_num_of_cols+1, N);		//N vectors of size num of cols in matrix. layout is compact batched layout
		vectortype_mirror V = mirror(V_gpu);

		printf("%s %d\n", __FILE__, __LINE__);

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
		printf("%s %d\n", __FILE__, __LINE__);
		//set last row to 0 hence even if col values are invalid, we will get 0.
		//this is to avoid checking if every time
		for(int j=0; j<N; j++)
			V(M.m_num_of_cols, j) = 0.0;
		Kokkos::deep_copy(V_gpu, V);

		printf("%s %d\n", __FILE__, __LINE__);
		//----------------------------------  cpu calc  ----------------------------------
		value_type *R_cpu = new value_type[M.m_num_of_rows*N];	//result vector for cpu.
		printf("%s %d\n", __FILE__, __LINE__);
#pragma omp parallel for
		for(int i=0; i<M.m_num_of_rows; i++)
			for(int j=0; j<N; j++)
				R_cpu[i*N+j] = 0.0;
		printf("%s %d\n", __FILE__, __LINE__);
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
		printf("%s %d\n", __FILE__, __LINE__);


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

			fflush(stdout);
			automicvectortype R_gpu("R", M.m_num_of_rows, N); //result vector for gpu.
			vectortype_mirror R = mirror(R_gpu);
			printf("R allocated \n");

			resetR(M.m_num_of_rows, R_gpu);

			printf("omp_threads: %d \n",omp_threads);

			printf("Teams \t Threads \t PVL \t LVL \t simple_error \t simple1_error \t simd_error \t simd1_error \t simple_time \t simple1_time \t simd_time \t simd1_time \n");


			//for(int Blocks=4; Blocks<=omp_threads; Blocks=Blocks*2)
			const int Blocks = 64;
			__itt_resume();
			call_spmv<KNL_PVL, N>(M, V_gpu, R_gpu, R, R_cpu, Blocks, omp_threads/Blocks);
			__itt_detach();

			printf("%s %d\n", __FILE__, __LINE__);
			/*std::vector<int> BlocksVector = {4096, 8192, 16384, 32768};
			std::vector<int> ThreadsPerBlockVector = {4, 8, 16, 32, 64, 128};
			//BlocksVector.push_back(std::min(2147483647, M.m_num_of_rows/128));
			//std::vector<int> BlocksVector = {512};	//best timings always among these many teams
			//std::vector<int> ThreadsPerBlockVector = {16}; //best timings always among these many threads


			fflush(stdout);
			for(int i=0; i<BlocksVector.size(); i++)
			{
				int Blocks = BlocksVector[i];
				for(int j=0; j<ThreadsPerBlockVector.size(); j++)
				{
					int ThreadsPerBlock = ThreadsPerBlockVector[j];
					call_spmv<8, N>(M, V_gpu, R_gpu, R, R_cpu, Blocks, ThreadsPerBlock);
					call_spmv<16, N>(M, V_gpu, R_gpu, R, R_cpu, Blocks, ThreadsPerBlock);
					call_spmv<32, N>(M, V_gpu, R_gpu, R, R_cpu, Blocks, ThreadsPerBlock);
					call_spmv<64, N>(M, V_gpu, R_gpu, R, R_cpu, Blocks, ThreadsPerBlock);
					call_spmv<128, N>(M, V_gpu, R_gpu, R, R_cpu, Blocks, ThreadsPerBlock);
				}
			}
			*/

		}

		//----------------------------------  cublass  ----------------------------------

		printf("--------------------- calling cublas ------------------------\n");
		fflush(stdout);


#if 0
		vectortype V1_gpu("V", N , M.m_num_of_cols);	//cublass wont work with compact form. Hence take transpose.

		Kokkos::parallel_for("___V_Transpose___",M.m_num_of_cols , [=] __device__ (int i)
		{
			for(int j=0; j<N; j++)
				V1_gpu(j, i) = V_gpu(i, j);
		});
		execution_space::fence();
#endif


		//blas_spmv(M, V_gpu.data(), R_cpu);

		delete []R_cpu;

	}
	Kokkos::finalize();

	return(0);
}
