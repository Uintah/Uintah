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
#define TILE 4
// Array of vectors. (or a 2D array)
//typedef Kokkos::View<value_type**, Kokkos::LayoutRight, Kokkos::Cuda, Kokkos::MemoryTraits<Kokkos::Atomic>> automicvectortype;
typedef Kokkos::View<value_type**, Kokkos::LayoutRight, Kokkos::Cuda, Kokkos::MemoryTraits<Kokkos::RandomAccess>> automicvectortype;
typedef Kokkos::View<value_type**, Kokkos::LayoutRight, Kokkos::Cuda, Kokkos::MemoryTraits<Kokkos::RandomAccess>> vectortype;
typedef Kokkos::View<value_type**, Kokkos::LayoutRight, Kokkos::Cuda>::HostMirror vectortype_mirror;
typedef const typename Kokkos::TeamPolicy<Kokkos::Cuda>::member_type team_member;


// logic from https://www.nvidia.com/docs/IO/66889/nvr-2008-004.pdf: Efficient Sparse Matrix-Vector Multiplication on CUDA (Nathan Bell and Michael Garland†)
// + using compact batched format
//PVL and LVL: Physical and logical vector lengths
template <int PVL, int LVL=-1>	//LVL is not used in this routine. ALWAYS use PVL.
void ___csr_spmv___(const spm &M, const vectortype &V, automicvectortype &R, const int Blocks, const int ThreadsPerBlock)
{

	const Kokkos::TeamPolicy<Kokkos::Cuda> policy( Blocks , ThreadsPerBlock, PVL );
	const int shm_size = ThreadsPerBlock*N;

	//rows are distributed among teams. Each team processes 1 row in 1 iteration.
	Kokkos::parallel_for(policy.set_scratch_size(0, Kokkos::PerTeam(shm_size*sizeof(value_type))), [=] __device__ (team_member & thread )
	{
		const int blockId = thread.league_rank();
		const int blockSize = thread.league_size();
		const int CudaThreadPerBlock = ThreadsPerBlock*PVL;

		//shared memory: each thread stores its local sums in shared memory for every vector in matrix V. Sums are finally reduced across threads for every vector
		//Does not have generalized implementation yet. Crashes if N*ThreadsPerBlock*sizeof(value_type) > size of shared memory

		value_type * R_shared = (value_type*) thread.team_shmem().get_shmem(shm_size*sizeof(value_type));

		Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, ThreadsPerBlock), [=] __device__ (const int& thread_id)
		{
			Kokkos::parallel_for(Kokkos::ThreadVectorRange(thread, PVL), [=] __device__ (const int& vector_id)
			{
				//init rows to 0 only for first row. later on can be done after result is written to main memory
				const int gshared_id = thread_id * PVL + vector_id;
				for(int i = gshared_id; i<shm_size; i=i+CudaThreadPerBlock)	//advancing by block size..
					R_shared[gshared_id] = 0.0;

				// no need of sync thread because each thread uses locations reset by itself only

				for(int row=blockId; row<M.m_num_of_rows; row=row+blockSize)	//iterate over rows. Each team processes 1 row at a time.
				{
					const int start = M.m_ptr_in_col_id_gpu(row);
					const int end = M.m_ptr_in_col_id_gpu(row+1);

					//main spmv loop
					for(int j= start + thread_id; j<end; j=j+ThreadsPerBlock)
					{
						const int col = M.m_col_id_gpu(j);	//column
						const value_type a = M.m_csr_data_gpu(j);	//matrix data

						//for(int k=vector_id, l=gshared_id; k<N; k=k+VectorLength, l=l+CudaThreadPerBlock)	//alternate iteration to avoid bank conflicts.. but not working
						for(int k=vector_id; k<N; k=k+PVL)	//iterate over simd dimension - number of vectors
						{
							value_type x = V(col, k);
							const int shared_id = thread_id * N + k;
							R_shared[shared_id] += x*a;
							//R_shared[l] += a*x;
						}
					}
					__syncthreads();	//__syncthreads because reduction can not start untill all threads complete their local sums

					//reduce values to shared memory, write to main memory and reset shared memory
#pragma unroll(4)
					for(int next_thread = ThreadsPerBlock >> 1; next_thread > 0; next_thread = next_thread >> 1)	//divide next thread by 2 every time
					{
						for(int k=vector_id; k<N; k=k+PVL)
						{
							const int shared_id = thread_id * N + k;
							if ( thread_id < next_thread)
								R_shared[ shared_id ] += R_shared [ shared_id + next_thread * N];
						}
						__syncthreads();	//__syncthreads because reduction can not proceed to next level untill all threads complete their reductions at current level
					}

					for(int k=vector_id; k<N; k=k+PVL)
					{
						const int shared_id = thread_id * N + k;

						if ( thread_id == 0)	// first thread writes the result
							R(row,k) = R_shared [ shared_id ];

						R_shared [ shared_id ] = 0.0;	//reset shared memory
						// no need of sync thread because each thread uses locations reset by itself only
					}
				}
			});
		});
	});
}

template <int PVL, int LVL=N>	//LVL is not used in this routine. ALWAYS use PVL.
void ___csr_simple_spmv___(const spm &M, const vectortype &V1, automicvectortype &R1, const int Blocks, const int ThreadsPerBlock)
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

#define read_M(j, end, col, a)	\
		if(j<end)	\
		{			\
			col = M.m_col_id_gpu(j);	\
			a = M.m_csr_data_gpu(j);	\
		}


template <int PVL, int LVL=-1>	//LVL is not used in this routine. ALWAYS use PVL.
void ___csr_tile_spmv___(const spm &M, const vectortype &V, automicvectortype &R, const int Blocks, const int ThreadsPerBlock)
{

	const Kokkos::TeamPolicy<Kokkos::Cuda> policy( Blocks , ThreadsPerBlock, PVL );
	const int shm_size = ThreadsPerBlock*N*TILE;

	//rows are distributed among teams. Each team processes 1 row in 1 iteration.
	Kokkos::parallel_for(policy.set_scratch_size(0, Kokkos::PerTeam(shm_size*sizeof(value_type))), [=] __device__ (team_member & thread )
	{
		const int blockId = thread.league_rank();
		const int blockSize = thread.league_size();
		const int CudaThreadPerBlock = ThreadsPerBlock*PVL;

		//shared memory: each thread stores its local sums in shared memory for every vector in matrix V. Sums are finally reduced across threads for every vector
		//Does not have generalized implementation yet. Crashes if N*ThreadsPerBlock*sizeof(value_type) > size of shared memory

		value_type * R_shared = (value_type*) thread.team_shmem().get_shmem(shm_size*sizeof(value_type));

		Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, ThreadsPerBlock), [=] __device__ (const int& thread_id)
		{
			Kokkos::parallel_for(Kokkos::ThreadVectorRange(thread, PVL), [=] __device__ (const int& vector_id)
			{
				//init rows to 0 only for first row. later on can be done after result is written to main memory
				const int gshared_id = thread_id * PVL + vector_id;
				for(int i = gshared_id; i<shm_size; i=i+CudaThreadPerBlock)	//advancing by block size..
					R_shared[gshared_id] = 0.0;
				__syncthreads();
				// no need of sync thread because each thread uses locations reset by itself only
				//const int row_counter = M.m_num_of_rows + M.m_num_of_rows%TILE;	//make loop count even.

				for(int row=blockId*TILE; row<M.m_num_of_rows; row=row+blockSize*TILE)	//iterate over rows. Each team processes TILE rows at a time.
				{
					const int row0 = row;
					const int start0 = M.m_ptr_in_col_id_gpu(row0) + thread_id*TILE;
					const int end0 = M.m_ptr_in_col_id_gpu(row0+1);

					int row1 = -1, start1=-1, end1=-1;
					int row2 = -1, start2=-1, end2=-1;
					int row3 = -1, start3=-1, end3=-1;

					if(row0+1 < M.m_num_of_rows)
					{
						row1 = row0+1;
						start1 = M.m_ptr_in_col_id_gpu(row1) + thread_id*TILE;
						end1 = M.m_ptr_in_col_id_gpu(row1+1);
					}

					if(row0+2 < M.m_num_of_rows)
					{
						row2 = row0+2;
						start2 = M.m_ptr_in_col_id_gpu(row2) + thread_id*TILE;
						end2 = M.m_ptr_in_col_id_gpu(row2+1);
					}

					if(row0+3 < M.m_num_of_rows)
					{
						row3 = row0+3;
						start3 = M.m_ptr_in_col_id_gpu(row3) + thread_id*TILE;
						end3 = M.m_ptr_in_col_id_gpu(row3+1);
					}
					//main spmv loop
					for(int j0=start0, j1=start1, j2=start2, j3=start3;
							(j0<end0) || (j1<end1) || (j2<end2) || (j3<end3);
							j0=j0+ThreadsPerBlock*TILE, j1=j1+ThreadsPerBlock*TILE, j2=j2+ThreadsPerBlock*TILE, j3=j3+ThreadsPerBlock*TILE)
					{
						/*int col00=-1, col01=-1, col02=-1, col03=-1,
							col10=-1, col11=-1, col12=-1, col13=-1,
							col20=-1, col21=-1, col22=-1, col23=-1,
							col30=-1, col31=-1, col32=-1, col33=-1;	*/

						int col00=M.m_num_of_cols, col01=col00, col02=col00, col03=col00,
							col10=col00, col11=col00, col12=col00, col13=col00,
							col20=col00, col21=col00, col22=col00, col23=col00,
							col30=col00, col31=col00, col32=col00, col33=col00;


						value_type 	a00=-1, a01=-1, a02=-1, a03=-1,
									a10=-1, a11=-1, a12=-1, a13=-1,
									a20=-1, a21=-1, a22=-1, a23=-1,
									a30=-1, a31=-1, a32=-1, a33=-1;


						read_M(j0+0, end0, col00, a00); read_M(j0+1, end0, col01, a01); read_M(j0+2, end0, col02, a02); read_M(j0+3, end0, col03, a03);
						read_M(j1+0, end1, col10, a10); read_M(j1+1, end1, col11, a11); read_M(j1+2, end1, col12, a12); read_M(j1+3, end1, col13, a13);
						read_M(j2+0, end2, col20, a20); read_M(j2+1, end2, col21, a21); read_M(j2+2, end2, col22, a22); read_M(j2+3, end2, col23, a23);
						read_M(j3+0, end3, col30, a30); read_M(j3+1, end3, col31, a31); read_M(j3+2, end3, col32, a32); read_M(j3+3, end3, col33, a33);


						//for(int k=vector_id, l=gshared_id; k<N; k=k+VectorLength, l=l+CudaThreadPerBlock)	//alternate iteration to avoid bank conflicts.. but not working
						for(int k=vector_id*TILE; k<N; k=k+PVL*TILE)	//iterate over simd dimension - number of vectors. Assuming N to be even ALWAYS
						{
							value_type  r0x00=0, r0x01=0, r0x02=0, r0x03=0,
										r0x10=0, r0x11=0, r0x12=0, r0x13=0,
										r0x20=0, r0x21=0, r0x22=0, r0x23=0,
										r0x30=0, r0x31=0, r0x32=0, r0x33=0;

							value_type  r1x00=0, r1x01=0, r1x02=0, r1x03=0,
										r1x10=0, r1x11=0, r1x12=0, r1x13=0,
										r1x20=0, r1x21=0, r1x22=0, r1x23=0,
										r1x30=0, r1x31=0, r1x32=0, r1x33=0;

							value_type  r2x00=0, r2x01=0, r2x02=0, r2x03=0,
										r2x10=0, r2x11=0, r2x12=0, r2x13=0,
										r2x20=0, r2x21=0, r2x22=0, r2x23=0,
										r2x30=0, r2x31=0, r2x32=0, r2x33=0;

							value_type  r3x00=0, r3x01=0, r3x02=0, r3x03=0,
										r3x10=0, r3x11=0, r3x12=0, r3x13=0,
										r3x20=0, r3x21=0, r3x22=0, r3x23=0,
										r3x30=0, r3x31=0, r3x32=0, r3x33=0;


							//last row of V is set to 0. hence even if col values are invalid, we will get 0.
							//this is to avoid checking if every time
							r0x00 = V(col00, k); r0x01 = V(col00, k+1); r0x02 = V(col00, k+2); r0x03 = V(col00, k+3);
							r0x10 = V(col01, k); r0x11 = V(col01, k+1); r0x12 = V(col01, k+2); r0x13 = V(col01, k+3);
							r0x20 = V(col02, k); r0x21 = V(col02, k+1); r0x22 = V(col02, k+2); r0x23 = V(col02, k+3);
							r0x30 = V(col03, k); r0x31 = V(col03, k+1); r0x32 = V(col03, k+2); r0x33 = V(col03, k+3);

							r1x00 = V(col10, k); r1x01 = V(col10, k+1); r1x02 = V(col10, k+2); r1x03 = V(col10, k+3);
							r1x10 = V(col11, k); r1x11 = V(col11, k+1); r1x12 = V(col11, k+2); r1x13 = V(col11, k+3);
							r1x20 = V(col12, k); r1x21 = V(col12, k+1); r1x22 = V(col12, k+2); r1x23 = V(col12, k+3);
							r1x30 = V(col13, k); r1x31 = V(col13, k+1); r1x32 = V(col13, k+2); r1x33 = V(col13, k+3);

							r2x00 = V(col20, k); r2x01 = V(col20, k+1); r2x02 = V(col20, k+2); r2x03 = V(col20, k+3);
							r2x10 = V(col21, k); r2x11 = V(col21, k+1); r2x12 = V(col21, k+2); r2x13 = V(col21, k+3);
							r2x20 = V(col22, k); r2x21 = V(col22, k+1); r2x22 = V(col22, k+2); r2x23 = V(col22, k+3);
							r2x30 = V(col23, k); r2x31 = V(col23, k+1); r2x32 = V(col23, k+2); r2x33 = V(col23, k+3);

							r3x00 = V(col30, k); r3x01 = V(col30, k+1); r3x02 = V(col30, k+2); r3x03 = V(col30, k+3);
							r3x10 = V(col31, k); r3x11 = V(col31, k+1); r3x12 = V(col31, k+2); r3x13 = V(col31, k+3);
							r3x20 = V(col32, k); r3x21 = V(col32, k+1); r3x22 = V(col32, k+2); r3x23 = V(col32, k+3);
							r3x30 = V(col33, k); r3x31 = V(col33, k+1); r3x32 = V(col33, k+2); r3x33 = V(col33, k+3);

							const int shared_id0 = thread_id * N *TILE + k;
							const int shared_id1 = shared_id0 + N;
							const int shared_id2 = shared_id0 + 2*N;
							const int shared_id3 = shared_id0 + 3*N;

							R_shared[shared_id0+0] += a00*r0x00 + a01*r0x10 + a02*r0x20 + a03*r0x30;
							R_shared[shared_id0+1] += a00*r0x01 + a01*r0x11 + a02*r0x21 + a03*r0x31;
							R_shared[shared_id0+2] += a00*r0x02 + a01*r0x12 + a02*r0x22 + a03*r0x32;
							R_shared[shared_id0+3] += a00*r0x03 + a01*r0x13 + a02*r0x23 + a03*r0x33;

							R_shared[shared_id1+0] += a10*r1x00 + a11*r1x10 + a12*r1x20 + a13*r1x30;
							R_shared[shared_id1+1] += a10*r1x01 + a11*r1x11 + a12*r1x21 + a13*r1x31;
							R_shared[shared_id1+2] += a10*r1x02 + a11*r1x12 + a12*r1x22 + a13*r1x32;
							R_shared[shared_id1+3] += a10*r1x03 + a11*r1x13 + a12*r1x23 + a13*r1x33;

							R_shared[shared_id2+0] += a20*r2x00 + a21*r2x10 + a22*r2x20 + a23*r2x30;
							R_shared[shared_id2+1] += a20*r2x01 + a21*r2x11 + a22*r2x21 + a23*r2x31;
							R_shared[shared_id2+2] += a20*r2x02 + a21*r2x12 + a22*r2x22 + a23*r2x32;
							R_shared[shared_id2+3] += a20*r2x03 + a21*r2x13 + a22*r2x23 + a23*r2x33;

							R_shared[shared_id3+0] += a30*r3x00 + a31*r3x10 + a32*r3x20 + a33*r3x30;
							R_shared[shared_id3+1] += a30*r3x01 + a31*r3x11 + a32*r3x21 + a33*r3x31;
							R_shared[shared_id3+2] += a30*r3x02 + a31*r3x12 + a32*r3x22 + a33*r3x32;
							R_shared[shared_id3+3] += a30*r3x03 + a31*r3x13 + a32*r3x23 + a33*r3x33;

						}
					}
					__syncthreads();	//__syncthreads because reduction can not start untill all threads complete their local sums

					//reduce values to shared memory, write to main memory and reset shared memory
#pragma unroll(4)
					for(int next_thread = ThreadsPerBlock >> 1; next_thread > 0; next_thread = next_thread >> 1)	//divide next thread by 2 every time
					{
						for(int k=vector_id; k<N; k=k+PVL)
						{
							const int shared_id0 = thread_id * N *TILE + k;
							const int shared_id1 = shared_id0 + N;
							const int shared_id2 = shared_id0 + 2*N;
							const int shared_id3 = shared_id0 + 3*N;

							if ( thread_id < next_thread)
							{
								R_shared[ shared_id0 ] += R_shared [ shared_id0 + next_thread * N*TILE];
								R_shared[ shared_id1 ] += R_shared [ shared_id1 + next_thread * N*TILE];
								R_shared[ shared_id2 ] += R_shared [ shared_id2 + next_thread * N*TILE];
								R_shared[ shared_id3 ] += R_shared [ shared_id3 + next_thread * N*TILE];
							}
						}
						__syncthreads();	//__syncthreads because reduction can not proceed to next level untill all threads complete their reductions at current level
					}

					for(int k=vector_id; k<N; k=k+PVL)
					{
						const int shared_id = thread_id * N *TILE + k;

						if ( thread_id == 0)	// first thread writes the result
						{
							R(row0,k) = R_shared [ shared_id ];
							if(row1>-1) R(row1,k) = R_shared [ shared_id + N];
							if(row2>-1) R(row2,k) = R_shared [ shared_id + 2*N];
							if(row3>-1) R(row3,k) = R_shared [ shared_id + 3*N];
						}
						//__syncthreads();
						R_shared [ shared_id ] = 0.0;	//reset shared memory
						R_shared [ shared_id + N ] = 0.0;
						R_shared [ shared_id + 2*N ] = 0.0;
						R_shared [ shared_id + 3*N ] = 0.0;
						// no need of sync thread because each thread uses locations reset by itself only
					}
				}
			});
		});
	});
}



template <int PVL, int LVL>
void ___simd_spmv___(const spm &M, const vectortype &V1, automicvectortype &R1, const int Blocks, const int ThreadsPerBlock)
{
	//cast V and R to simd views
	typedef simd<PVL, LVL> simd_type;
	typedef Kokkos::View<simd_type**, Kokkos::LayoutRight, Kokkos::Cuda> simd_vectortype;
	simd_vectortype simd_V = simd_vectortype(reinterpret_cast<simd_type*>(V1.data()), V1.extent(0), V1.extent(1) / LVL);
	simd_vectortype simd_R = simd_vectortype(reinterpret_cast<simd_type*>(R1.data()), R1.extent(0), R1.extent(1) / LVL);
	const int N_by_simd = N / LVL;

	const Kokkos::TeamPolicy<Kokkos::Cuda> policy( Blocks , ThreadsPerBlock, PVL );
	const int shm_size = ThreadsPerBlock*N;

	Kokkos::parallel_for(policy.set_scratch_size(0, Kokkos::PerTeam(shm_size*sizeof(value_type))), [=] __device__ (team_member & thread )
	{
		const int blockId = thread.league_rank();
		const int blockSize = thread.league_size();

		//shared memory: each thread stores its temporary sums in shared memory
		simd_type* simd_R_shared = (simd_type*) thread.team_shmem().get_shmem(shm_size*sizeof(value_type));

		Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, ThreadsPerBlock), [=] __device__ (const int& thread_id)
		{
			//no need of ThreadVectorRange loop here. Those iterations will be inside overloaded operators.
			//However need loop to iterate of simd dimension with iterations = N / LVL.

			//init rows to 0 only for first row. later on can be done after result is written to main memory
			for(int k=0; k<N_by_simd; k++)
			{
				int shared_id = thread_id*N_by_simd + k;
				simd_R_shared[shared_id] = 0.0;
			}


			for(int row=blockId; row<M.m_num_of_rows; row=row+blockSize)	//iterate over columns. Each partition processes 1 column at a time.
			{
				const int start = M.m_ptr_in_col_id_gpu(row);
				const int end = M.m_ptr_in_col_id_gpu(row+1);

				//spmv
				for(int j= start + thread_id; j<end; j=j+ThreadsPerBlock)
				{
					const int col = M.m_col_id_gpu(j);	//add back M.m_ptr_in_row_id(i)
					const value_type a = M.m_csr_data_gpu(j);	//matrix data

					//for(int k=vector_id, l=gshared_id; k<N; k=k+VectorLength, l=l+CudaThreadPerBlock)
					for(int k=0; k<N_by_simd; k=k+1)
					{
						const int shared_id = thread_id * N_by_simd + k;
						simd_R_shared[shared_id] += simd_V(col, k)*a;
					}
				}
				__syncthreads();

#pragma unroll(4)
				for(int next_thread = ThreadsPerBlock >> 1; next_thread > 0; next_thread = next_thread >> 1)	//divide next thread by 2 every time
				{
					for(int k=0; k<N_by_simd; k=k+1)
					{
						const int shared_id = thread_id * N_by_simd + k;
						if ( thread_id < next_thread)
							simd_R_shared[ shared_id ] += simd_R_shared [ shared_id + next_thread * N_by_simd];
					}
					__syncthreads();
				}

				for(int k=0; k<N_by_simd; k=k+1)
				{
					const int shared_id = thread_id * N_by_simd + k;
					if ( thread_id == 0)	// first thread writes the result
						simd_R(row,k) = simd_R_shared [ shared_id ];
				}

				__syncthreads();
				for(int k=0; k<N_by_simd; k=k+1)
				{
					const int shared_id = thread_id * N_by_simd + k;
					simd_R_shared [ shared_id ] = 0.0;	//reset shared memory
				}
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
	if(PVL*ThreadsPerBlock > 1024)
		return;


	struct timeval  tv1, tv2;
	double spmv_time=1000, tile_time=1000, simple_spmv_time=1000;
	double error=0.0 , tile_error=0.0, simple_spmv_error=0.0;


	if(ThreadsPerBlock*N*8 < 46*1024)
	{
		resetR(M.m_num_of_rows, R_gpu);
		//simple spmv
		Kokkos::Cuda::fence();
		gettimeofday(&tv1, NULL);
		___csr_spmv___<PVL>(M, V_gpu, R_gpu, Blocks, ThreadsPerBlock);
		Kokkos::Cuda::fence();
		gettimeofday(&tv2, NULL);
		spmv_time = (double) (tv2.tv_usec - tv1.tv_usec) / 1000 + (double) (tv2.tv_sec - tv1.tv_sec)*1000;
		Kokkos::deep_copy(R, R_gpu);
		Kokkos::Cuda::fence();
		error = calc_error(M.m_num_of_rows, R, R_cpu);
	}
	if(ThreadsPerBlock*N*8*TILE < 46*1024)
	{
		tile_error=0.0;
		//___csr_tile_spmv___ spmv
		resetR(M.m_num_of_rows, R_gpu);
		Kokkos::Cuda::fence();
		gettimeofday(&tv1, NULL);
		___csr_tile_spmv___<PVL>(M, V_gpu, R_gpu, Blocks, ThreadsPerBlock);
		Kokkos::Cuda::fence();
		gettimeofday(&tv2, NULL);
		tile_time = (double) (tv2.tv_usec - tv1.tv_usec) / 1000 + (double) (tv2.tv_sec - tv1.tv_sec)*1000;
		Kokkos::deep_copy(R, R_gpu);
		Kokkos::Cuda::fence();
		tile_error = calc_error(M.m_num_of_rows, R, R_cpu);
	}

	//___csr_tile1_spmv___ spmv
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


	/*for(int i=0; i<M.m_num_of_rows; i++)
	{
		for(int j=0;j<N;j++)
			printf("%f ",R(i,j));
		printf("\n");
	}*/
	printf("%d \t %d \t %d \t %d \t %f \t %f \t %f \t %f \t %f \t %f\n",
			Blocks, ThreadsPerBlock, PVL, LVL, error, tile_error, simple_spmv_error, spmv_time, tile_time, simple_spmv_time);
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


			std::vector<int> BlocksVector = {512, 2048, 8192, 16384};
			std::vector<int> ThreadsPerBlockVector = {8, 16, 32};
			//std::vector<int> BlocksVector = {512};	//best timings always among these many teams
			//std::vector<int> ThreadsPerBlockVector = {16}; //best timings always among these many threads

			printf("Blocks \t ThreadsPerBlock \t PVL \t LVL \t spmv_error \t simd_error \t simple_spmv_error \t spmv_time \t simd_spmv_time \t simple_spmv_time \n");
			fflush(stdout);
			for(int i=0; i<BlocksVector.size(); i++)
			{
				int Blocks = BlocksVector[i];
				for(int j=0; j<ThreadsPerBlockVector.size(); j++)
				{
					int ThreadsPerBlock = ThreadsPerBlockVector[j];

					//call_spmv<2, 64>(M, V_gpu, R_gpu, R, R_cpu, Blocks, ThreadsPerBlock);
					//call_spmv<4, 64>(M, V_gpu, R_gpu, R, R_cpu, Blocks, ThreadsPerBlock);
					call_spmv<8>(M, V_gpu, R_gpu, R, R_cpu, Blocks, ThreadsPerBlock);
					call_spmv<16>(M, V_gpu, R_gpu, R, R_cpu, Blocks, ThreadsPerBlock);
					call_spmv<32>(M, V_gpu, R_gpu, R, R_cpu, Blocks, ThreadsPerBlock);
					call_spmv<64>(M, V_gpu, R_gpu, R, R_cpu, Blocks, ThreadsPerBlock);
					call_spmv<128>(M, V_gpu, R_gpu, R, R_cpu, Blocks, ThreadsPerBlock);
					call_spmv<256>(M, V_gpu, R_gpu, R, R_cpu, Blocks, ThreadsPerBlock);

					/*call_spmv<32, 64>(M, V_gpu, R_gpu, R, R_cpu, Blocks, ThreadsPerBlock);
					call_spmv<64, 64>(M, V_gpu, R_gpu, R, R_cpu, Blocks, ThreadsPerBlock);

					call_spmv<32, 128>(M, V_gpu, R_gpu, R, R_cpu, Blocks, ThreadsPerBlock);
					call_spmv<64, 128>(M, V_gpu, R_gpu, R, R_cpu, Blocks, ThreadsPerBlock);

					call_spmv<32, 256>(M, V_gpu, R_gpu, R, R_cpu, Blocks, ThreadsPerBlock);
					call_spmv<64, 256>(M, V_gpu, R_gpu, R, R_cpu, Blocks, ThreadsPerBlock);*/

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

		for(int i=0; i<10; i++)
			cublas_spmv(M, V1_gpu.data(), R_cpu);


	}
	Kokkos::finalize();

	return(0);
}



