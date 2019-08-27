/*void ___csc_spmv___(const spm &M, const vectortype &V, automicvectortype &R, int Blocks=2048, int ThreadsPerBlock=32, int VectorLength=16)
{

	const Kokkos::TeamPolicy<Kokkos::Cuda> policy( Blocks , ThreadsPerBlock, VectorLength );

	Kokkos::parallel_for("___csc_spmv___", policy, [=] __device__ (team_member & thread )
	{
		int blockId = thread.league_rank();
		int blockSize = thread.league_size();

		for(int col=blockId; col<M.m_num_of_cols; col=col+blockSize)	//iterate over columns. Each partition processes 1 column at a time.
		{
			const int nnz_rows_in_col = M.m_ptr_in_row_id_gpu(col+1) - M.m_ptr_in_row_id_gpu(col);

			Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, nnz_rows_in_col), [=] __device__ (const int& it)	//iterate over matrix rows
			{
				const int j =  M.m_ptr_in_row_id_gpu(col) + it;
				const int row = M.m_row_id_gpu(j);	//add back M.m_ptr_in_row_id(i)
				const value_type a = M.m_data_gpu(j);	//matrix data

				Kokkos::parallel_for(Kokkos::ThreadVectorRange(thread, N), [=] __device__ (const int& k)
				{
					value_type x = V(col, k);
					R(row,k) += a*x;

					//symmetric matrix. hence interchanging row col and multiplying
					//value_type y = V(row, k);
					//R(col,k) += a*y;
				});
			});
		}
	});
}*/

/*https://www.nvidia.com/docs/IO/66889/nvr-2008-004.pdf: Efficient Sparse Matrix-Vector Multiplication on CUDA (Nathan Bell and Michael Garland†)
__global__ void spmv_csr_vector_kernel( const int num_rows ,
						const int * ptr ,
						const int * indices ,
						const double * data ,
						const double * x ,
						double * y)
{
	__shared__ double vals [32];
	int thread_id = blockDim.x * blockIdx.x + threadIdx.x ; // global thread index
	int warp_id = thread_id / 32; // global warp index
	int lane = thread_id & (32 - 1); // thread index within the warp
	// one warp per row
	int row = warp_id ;
	if ( row < num_rows )
	{
		int row_start = ptr [ row ];
		int row_end = ptr [ row +1];
		// compute running sum per thread
		vals [ threadIdx.x ] = 0;
		for ( int jj = row_start + lane ; jj < row_end ; jj += 32)
			vals [ threadIdx.x ] += data [ jj ] * x [ indices [ jj ]];
		// parallel reduction in shared memory
		if ( lane < 16) vals [ threadIdx.x ] += vals [ threadIdx.x + 16];
		if ( lane < 8) vals [ threadIdx.x ] += vals [ threadIdx.x + 8];
		if ( lane < 4) vals [ threadIdx.x ] += vals [ threadIdx.x + 4];
		if ( lane < 2) vals [ threadIdx.x ] += vals [ threadIdx.x + 2];
		if ( lane < 1) vals [ threadIdx.x ] += vals [ threadIdx.x + 1];
		// first thread writes the result
		if ( lane == 0)
		y[ row ] += vals [ threadIdx.x ];
	}
}*/









		/*for(int i=0; i<M.m_num_of_cols; i++)
		{
			const int col = i;
			for(int j=M.m_ptr_in_row_id(i); j<M.m_ptr_in_row_id(i+1); j++)
			{
				const int row = M.m_row_id(j);
				const value_type a = M.m_data(j);	//matrix data

				for(int k=0; k<N; k++)
				{
					value_type x = V(col, k);
					R_cpu[row][k] += a*x;

					//symmetric matrix. hence interchanging row col and multiplying
					//value_type y = V(row, k);
					//R_cpu[col][k] += a*y;
				}
			}
		}*/





/*for(int i=0; i<M.m_num_of_cols; i++)
{
	for(int j=0; j<N; j++)
		printf("%f ",V(i, j));
	printf("\n\n ");
}*/


/*printf("result\n\n");
for(int i=0; i<M.m_num_of_rows; i++)
{
	for(int j=0; j<N; j++)
		printf("%f ",R_cpu[i][j]);
	printf("\n\n ");
}*/






/*
void ___csr_spmv___(const spm &M, const vectortype_simd &V, vectortype_simd &R, const int Blocks, const int ThreadsPerBlock, const int VectorLength)
{
	const Kokkos::TeamPolicy<Kokkos::Cuda> policy( Blocks , ThreadsPerBlock, VectorLength );
	const int shm_size = 32 * 1024;	//for now request all available shared memory. change it later for portability.

	Kokkos::parallel_for("___csc_spmv___", policy.set_scratch_size(0, Kokkos::PerTeam(shm_size)), [=] __device__ (team_member & thread )
	{
		const int blockId = thread.league_rank();
		const int blockSize = thread.league_size();

		//*********  ASSUMING NNZ * 8 (FOR DATA) + NNZ * 4 (FOR COL) + BLOCK_SIZE * 8 (FOR RESULT) < 46 KB. **********************
		//get all the shared memory available
		unsigned char* all_shared_memory = (unsigned char*) thread.team_shmem().get_shmem(shm_size);

		//shared memory: each thread stores its temporary sums in shared memory. ThreadsPerBlock*VectorLength is block size
		value_type * R_shared = (value_type*) all_shared_memory;
		all_shared_memory = all_shared_memory + blockSize * sizeof(double);	//increment all_shared_memory pointer. block size will be align with 32. so should not cause bank conflicts

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

			//reset results and copy data into shared memory
			Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, ThreadsPerBlock), [=] __device__ (const int& i)	//iterate over matrix rows
			{
				Kokkos::parallel_for(Kokkos::ThreadVectorRange(thread, VectorLength), [=] __device__ (const int& j)
				{
					int thread_id = i*VectorLength + j;
					R_shared[thread_id] = 0.0;	//reset results
					for(int k= start + thread_id, l=thread_id; k<end; k=k+blockSize, l=l+blockSize)	//copy
					{
						data_shared[l] = M.m_csr_data_gpu(k);
						col_id_shared[l] = M.m_col_id_gpu(k);
					}
				});
			});

			__syncthreads();

			const int N_by_simd = N / VectorLength;
			for(int i=0; i<N_by_simd; i++)
			{
				//main spmv
				Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, ThreadsPerBlock), [=] __device__ (const int& thread_id)	//iterate over matrix rows
				{
					Kokkos::parallel_for(Kokkos::ThreadVectorRange(thread, VectorLength), [=] __device__ (const int& k)
					{

						for(int j= thread_id; j<nnz_elements; j=j+ThreadsPerBlock)
						{
							const int col = col_id_shared[j];	//add back M.m_ptr_in_row_id(i)
							const value_type a = data_shared[j];	//matrix data
							value_type x = V(i, col, k);
							const int shared_id = thread_id * VectorLength + k;
							R_shared[shared_id] += a*x;
						}
					});
				});

				__syncthreads();

				//reduce results in shared memory and copy to main memory
				Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, ThreadsPerBlock), [=] __device__ (const int& thread_id)	//iterate over matrix rows
				{
					Kokkos::parallel_for(Kokkos::ThreadVectorRange(thread, VectorLength), [=] __device__ (const int& k)
					{
						const int shared_id = thread_id * VectorLength + k;
	#pragma unroll(4)
						for(int next_thread = ThreadsPerBlock >> 1; next_thread > 0; next_thread = next_thread >> 1)	//divide next thread by 2 every time
						{
							if ( thread_id < next_thread)
								R_shared[ shared_id ] += R_shared [ shared_id + next_thread * VectorLength];
							__syncthreads();
						}

						if ( thread_id == 0)	// first thread writes the result
							R(i, row, k) = R_shared [ shared_id ];
						R_shared [ shared_id ] = 0.0;
					});
				});
				__syncthreads();
			}

		}
	});
}*/





























void call_simdn_spmv(const spm &M, const vectortype &V_gpu, vectortype &R_gpu,vectortype_mirror R, value_type * R_cpu, const int Blocks, const int ThreadsPerBlock)
{
	struct timeval  tv1, tv2;
	Kokkos::Cuda::fence();
	gettimeofday(&tv1, NULL);
	___simdn_spmv___<simd2, 2>(M, V_gpu, R_gpu, Blocks, ThreadsPerBlock);
	Kokkos::Cuda::fence();
	gettimeofday(&tv2, NULL);
	double simd2_time = (double) (tv2.tv_usec - tv1.tv_usec) / 1000 + (double) (tv2.tv_sec - tv1.tv_sec)*1000;
	Kokkos::deep_copy(R, R_gpu);
	Kokkos::Cuda::fence();
	double simd2_error = calc_error(M.m_num_of_rows, R, R_cpu);

	gettimeofday(&tv1, NULL);
	___simdn_spmv___<simd4, 4>(M, V_gpu, R_gpu, Blocks, ThreadsPerBlock);
	Kokkos::Cuda::fence();
	gettimeofday(&tv2, NULL);
	double simd4_time = (double) (tv2.tv_usec - tv1.tv_usec) / 1000 + (double) (tv2.tv_sec - tv1.tv_sec)*1000;
	Kokkos::deep_copy(R, R_gpu);
	Kokkos::Cuda::fence();
	double simd4_error = calc_error(M.m_num_of_rows, R, R_cpu);

	gettimeofday(&tv1, NULL);
	___simdn_spmv___<simd8, 8>(M, V_gpu, R_gpu, Blocks, ThreadsPerBlock);
	Kokkos::Cuda::fence();
	gettimeofday(&tv2, NULL);
	double simd8_time = (double) (tv2.tv_usec - tv1.tv_usec) / 1000 + (double) (tv2.tv_sec - tv1.tv_sec)*1000;
	Kokkos::deep_copy(R, R_gpu);
	Kokkos::Cuda::fence();
	double simd8_error = calc_error(M.m_num_of_rows, R, R_cpu);

	printf("%d \t %d \t %f \t %f \t %f \t %f \t %f\t %f\n",Blocks, ThreadsPerBlock, simd2_error, simd4_error, simd8_error, simd2_time, simd4_time, simd8_time );
}






			std::vector<int> BlocksVector1 = {512, 1024, 2048, 4096, 8192};	//best timings always among these many teams
			std::vector<int> ThreadsPerBlockVector1 = {16, 32, 64, 128, 256, 512, 1024}; //best timings always among these many threads


			printf("Blocks \t ThreadsPerBlock \t PVL \t simd2_error \t simd4_error \t simd8_error \t simd2_time \t simd4_time \t simd8_time\n");








template <int PVL, int LVL=-1>	//LVL is not used in this routine. ALWAYS use PVL.
void ___csr_tile_spmv___(const spm &M, const vectortype &V, automicvectortype &R, const int Blocks, const int ThreadsPerBlock)
{

	const Kokkos::TeamPolicy<Kokkos::Cuda> policy( Blocks , ThreadsPerBlock, PVL );
	const int shm_size = ThreadsPerBlock*N*2;

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

				for(int row=blockId; row<M.m_num_of_rows; row=row+blockSize*TILE)	//iterate over rows. Each team processes TILE rows at a time.
				{
					const int row0 = row;
					const int start0 = M.m_ptr_in_col_id_gpu(row0) + thread_id;
					const int end0 = M.m_ptr_in_col_id_gpu(row0+1);

					int row1 = -1, start1=-1, end1=-1;
					if(row0+blockSize < M.m_num_of_rows)
					{
						row1 = row0+blockSize;
						start1 = M.m_ptr_in_col_id_gpu(row1) + thread_id;
						end1 = M.m_ptr_in_col_id_gpu(row1+1);
					}
					//main spmv loop
					for(int j0=start0, j1=start1; (j0<end0) || (j1<end1); j0=j0+ThreadsPerBlock*TILE, j1=j1+ThreadsPerBlock*TILE)
					{
						int col00=-1, col01=-1, col10=-1, col11=-1;	//for dense matrix col00 = col10 and col01 = col11
						value_type a00=0, a01=0, a10=0, a11=0;

						if(j0<end0)	//row0 will always be valid. so no need to check it
						{
							col00 = M.m_col_id_gpu(j0);	//column
							a00 = M.m_csr_data_gpu(j0);	//matrix data
						}
						if(j0+ThreadsPerBlock<end0)
						{
							col01 = M.m_col_id_gpu(j0+ThreadsPerBlock);	//column
							a01 = M.m_csr_data_gpu(j0+ThreadsPerBlock);	//matrix data
						}
						if(j1<end1)
						{
							col10 = M.m_col_id_gpu(j1);
							a10 = M.m_csr_data_gpu(j1);	//matrix data
						}
						if(j1+ThreadsPerBlock<end1)
						{
							col11 = M.m_col_id_gpu(j1+ThreadsPerBlock);
							a11 = M.m_csr_data_gpu(j1+ThreadsPerBlock);	//matrix data
						}

						//for(int k=vector_id, l=gshared_id; k<N; k=k+VectorLength, l=l+CudaThreadPerBlock)	//alternate iteration to avoid bank conflicts.. but not working
						for(int k=vector_id; k<N; k=k+PVL*TILE)	//iterate over simd dimension - number of vectors. Assuming N to be even ALWAYS
						{
							value_type r0x00=0, r0x01=0, r0x10=0, r0x11=0;
							value_type r1x00=0, r1x01=0, r1x10=0, r1x11=0;

							if(col00>-1) r0x00 = V(col00, k);
							if(col00>-1) r0x01 = V(col00, k+PVL);
							if(col01>-1) r0x10 = V(col01, k);
							if(col01>-1) r0x11 = V(col01, k+PVL);

							if(col10>-1) r1x00 = V(col10, k);
							if(col10>-1) r1x01 = V(col10, k+PVL);
							if(col11>-1) r1x10 = V(col11, k);
							if(col11>-1) r1x11 = V(col11, k+PVL);

							/*R(row0,k)+= a00*r0x00 + a01*r0x10;
							R(row0,k+1)+= a00*r0x01 + a01*r0x11;
							if(row1>-1) R(row1,k) += a10*r1x00 + a11*r1x10;
							if(row1>-1) R(row1,k+1) += a10*r1x01 + a11*r1x11;*/

							const int shared_id0 = thread_id * N *TILE + k;
							const int shared_id1 = shared_id0 + N;

							R_shared[shared_id0] += a00*r0x00 + a01*r0x10;
							R_shared[shared_id0+PVL] += a00*r0x01 + a01*r0x11;

							R_shared[shared_id1] += a10*r1x00 + a11*r1x10;
							R_shared[shared_id1+PVL] += a10*r1x01 + a11*r1x11;

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
							const int shared_id0 = thread_id * N *TILE + k;
							const int shared_id1 = shared_id0 + N;
							if ( thread_id < next_thread)
							{
								R_shared[ shared_id0 ] += R_shared [ shared_id0 + next_thread * N*TILE];
								R_shared[ shared_id1 ] += R_shared [ shared_id1 + next_thread * N*TILE];
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
						}
						//__syncthreads();
						R_shared [ shared_id ] = 0.0;	//reset shared memory
						R_shared [ shared_id + N ] = 0.0;
						// no need of sync thread because each thread uses locations reset by itself only
					}
				}
			});
		});
	});
}









/*
template <int PVL, int LVL>
void ___simd_spmv___(const spm &M, const vectortype &V1, vectortype &R1, const int Blocks, const int ThreadsPerBlock)
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
*/











template <int PVL, int LVL=-1>	//LVL is not used in this routine. ALWAYS use PVL.
void ___warp_spmv___(const spm &M, const vectortype &V, vectortype &R, const int Blocks, const int ThreadsPerBlock)
{

	const Kokkos::TeamPolicy<Kokkos::Cuda> policy( Blocks , ThreadsPerBlock, PVL );
	const int shm_size = N;

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
calc			{
				//init rows to 0 only for first row. later on can be done after result is written to main memory
				const int gshared_id = thread_id * PVL + vector_id;
				const int warp_id = gshared_id/32;
				const int lane = gshared_id%32;
				const int num_of_warp = CudaThreadPerBlock/32;

				for(int i = gshared_id; i<shm_size; i=i+CudaThreadPerBlock)	//advancing by block size..
					R_shared[gshared_id] = 0.0;
				__syncthreads();
				// no need of sync thread because each thread uses locations reset by itself only

				for(int row=blockId; row<M.m_num_of_rows; row=row+blockSize)	//iterate over rows. Each team processes 1 row at a time.
				{
					const int start = M.m_ptr_in_col_id_gpu(row);
					const int end = M.m_ptr_in_col_id_gpu(row+1);

					//main spmv loop
					for(int j= start + warp_id; j<end; j=j+num_of_warp)	//each warp processes 1 column at a time
					{
						const int col = M.m_col_id_gpu(j);	//column
						const value_type a = M.m_csr_data_gpu(j);	//matrix data

#pragma unroll(8)
						for(int k=lane; k<N; k=k+32)	//iterate over simd dimension - number of vectors
						{
							value_type x = V(col, k);
							R_shared[k] += x*a;
						}
					}
					__syncthreads();	//__syncthreads because reduction can not start untill all threads complete their local sums

					//reduce values to shared memory, write to main memory and reset shared memory
#pragma unroll(8)

					for(int k=lane; k<N; k=k+32)
					{
						if ( thread_id == 0)	// first thread writes the result
							R(row,k) = R_shared [ k ];
						__syncthreads();
						R_shared [ k ] = 0.0;	//reset shared memory
						// no need of sync thread because each thread uses locations reset by itself only
					}
					__syncthreads();
				}
			});
		});
	});
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
						int col00=-1, col01=-1, col02=-1, col03=-1,
							col10=-1, col11=-1, col12=-1, col13=-1,
							col20=-1, col21=-1, col22=-1, col23=-1,
							col30=-1, col31=-1, col32=-1, col33=-1;	//for dense matrix col00 = col10 and col01 = col11
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

							if(col00>-1) {r0x00 = V(col00, k); r0x01 = V(col00, k+1); r0x02 = V(col00, k+2); r0x03 = V(col00, k+3); }
							if(col01>-1) {r0x10 = V(col01, k); r0x11 = V(col01, k+1); r0x12 = V(col01, k+2); r0x13 = V(col01, k+3); }
							if(col02>-1) {r0x20 = V(col02, k); r0x21 = V(col02, k+1); r0x22 = V(col02, k+2); r0x23 = V(col02, k+3); }
							if(col03>-1) {r0x30 = V(col03, k); r0x31 = V(col03, k+1); r0x32 = V(col03, k+2); r0x33 = V(col03, k+3); }

							if(col10>-1) {r1x00 = V(col10, k); r1x01 = V(col10, k+1); r1x02 = V(col10, k+2); r1x03 = V(col10, k+3); }
							if(col11>-1) {r1x10 = V(col11, k); r1x11 = V(col11, k+1); r1x12 = V(col11, k+2); r1x13 = V(col11, k+3); }
							if(col12>-1) {r1x20 = V(col12, k); r1x21 = V(col12, k+1); r1x22 = V(col12, k+2); r1x23 = V(col12, k+3); }
							if(col13>-1) {r1x30 = V(col13, k); r1x31 = V(col13, k+1); r1x32 = V(col13, k+2); r1x33 = V(col13, k+3); }

							if(col20>-1) {r2x00 = V(col20, k); r2x01 = V(col20, k+1); r2x02 = V(col20, k+2); r2x03 = V(col20, k+3); }
							if(col21>-1) {r2x10 = V(col21, k); r2x11 = V(col21, k+1); r2x12 = V(col21, k+2); r2x13 = V(col21, k+3); }
							if(col22>-1) {r2x20 = V(col22, k); r2x21 = V(col22, k+1); r2x22 = V(col22, k+2); r2x23 = V(col22, k+3); }
							if(col23>-1) {r2x30 = V(col23, k); r2x31 = V(col23, k+1); r2x32 = V(col23, k+2); r2x33 = V(col23, k+3); }

							if(col30>-1) {r3x00 = V(col30, k); r3x01 = V(col30, k+1); r3x02 = V(col30, k+2); r3x03 = V(col30, k+3); }
							if(col31>-1) {r3x10 = V(col31, k); r3x11 = V(col31, k+1); r3x12 = V(col31, k+2); r3x13 = V(col31, k+3); }
							if(col32>-1) {r3x20 = V(col32, k); r3x21 = V(col32, k+1); r3x22 = V(col32, k+2); r3x23 = V(col32, k+3); }
							if(col33>-1) {r3x30 = V(col33, k); r3x31 = V(col33, k+1); r3x32 = V(col33, k+2); r3x33 = V(col33, k+3); }

							/*if(col01>-1) r0x10 = V(col01, k);
							if(col01>-1) r0x11 = V(col01, k+1);

							if(col10>-1) r1x00 = V(col10, k);
							if(col10>-1) r1x01 = V(col10, k+1);
							if(col11>-1) r1x10 = V(col11, k);
							if(col11>-1) r1x11 = V(col11, k+1);*/

							/*R(row0,k)+= a00*r0x00 + a01*r0x10;
							R(row0,k+1)+= a00*r0x01 + a01*r0x11;
							if(row1>-1) R(row1,k) += a10*r1x00 + a11*r1x10;
							if(row1>-1) R(row1,k+1) += a10*r1x01 + a11*r1x11;*/

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

							/*R_shared[shared_id0+2] += a00*r0x00 + a01*r0x10;
							R_shared[shared_id0+3] += a00*r0x01 + a01*r0x11;

							R_shared[shared_id1] += a10*r1x00 + a11*r1x10;
							R_shared[shared_id1+1] += a10*r1x01 + a11*r1x11;*/

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












%
Consider the following
pseudo code for the matrix multiplication using Kokkos \textit{parallel\_for}
assuming a row major order:

\begin{lstlisting}[caption={Return type challenge},captionpos=b] 
//assuming vector length 8 for KNL
#define VEC_LEN 8

//Vector<double> is a SIMD primitive for double
Vector<double> A[N][N][M/VEC_LEN], B[N][N][M/VEC_LEN], C[N][N][M/VEC_LEN];

//populate A and B.

Kokkos::parallel_for(N*N, [&](int id) {	//parall_for across rows and columns
  int i = id/N, j=id%N; //row and column ids.
  for(int m=0; m<M/VEC_LEN; m++) //iterate over all matrices
  { 
    Vector<double> temp;
    for(int k=0; k<N; k++)
	  temp += A[i][k][m] * B[k][j][m]; 
	C[i][j][m] = temp;
  } 
});
\end{lstlisting}

The main reason for using Kokkos \textit{parallel\_for} is to make it portable
and of course to extract parallelism. A, B and C are arrays of M matrices each
of size N x N. These are declared using SIMD primitive - Vector\textless double \textgreater and are
laid out with matrix (M) dimension as the smallest dimension so that the
vectorization can take place across matrices. m th dimension is divided by
\textit{VEC\_LEN} to get M/\textit{VEC\_LEN} chunks, each containing \textit{VEC\_LEN} elements.
The loop will perform a standard multiplication of 'i'th row of A and 'j'th column
of B for 'm'th chunk of matrices and accumulate its result in temp. Note that each
element returned by A, B and C is SIMD element and contains 8 doubles. The same is
true for the temp variable. 

Now when this code is ported to a cpu, with Kokkos, 'temp' and matrix element both
have same length of \textit{VEC\_LEN}. 
%
The code works smoothly because the declaration of
'temp' is outside the SIMD context and hence each element of temp gets mapped to the
vector lane of VPU. 
%
However when the same code is ported to a GPU, the entire code is
executed in the SIMD context due to execution model of CUDA. 
%
Thus each SIMD thread declares 'temp' to be of size \textit{VEC\_LEN}. 
%
Although each thread operates on its own SIMD lane of temp, thereby making code work technically, 
the rest of the elements of temp always remain unused and this approach thus wastes a  large 
amount of memory.
%

The second challenge is the return type of operation A[i][k][m] * B[k][j][m].
CPU intrinsics return the intrinsic packed element. 
%











  \item \textbf{Dummy SIMD Type}: In this approach, the output CUDA SIMD type was written
  with only one element rather than using an array of \textit{VEC\_LEN}
  elements. This solved the problem as each SIMD element contains
  only one data element, it exactly maps this value to the value returned by a single CUDA
  thread. Further declaring it inside SIMD context also works well as it does
  not waste extra memory space as explained earlier. Furthermore  this approach
  gave the best performance as each element returned a single double. This value could
  well be stored in registers rather than in memory. However a problem arises when
  THIS IS IS REALLY CONFUSING!!!!! 
  SIMD primitive is included as an element of another class / structure. Being ``SIMD
  primitive'', it is expected to be of length VEC\_LEN and all index calculations
  of array access would be based on VEC\_LEN. However this implementation
  occupies space of only one scalar element thus disturbing all the offset
  calculations within a structure.
  




















template <int PVL, int LVL=-1>	//LVL is not used in this routine. ALWAYS use PVL.
void ___csr_tile1_spmv___(const spm &M, const vectortype &V, automicvectortype &R, const int Blocks, const int ThreadsPerBlock)
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

				for(int row=blockId; row<M.m_num_of_rows; row=row+blockSize*TILE)	//iterate over rows. Each team processes TILE rows at a time.
				{
					const int row0 = row;
					const int start0 = M.m_ptr_in_col_id_gpu(row0) + thread_id;
					const int end0 = M.m_ptr_in_col_id_gpu(row0+1);

					int row1 = -1, start1=-1, end1=-1;
					int row2 = -1, start2=-1, end2=-1;
					int row3 = -1, start3=-1, end3=-1;

					if(row+1*blockSize < M.m_num_of_rows)
					{
						row1 = row+1*blockSize;
						start1 = M.m_ptr_in_col_id_gpu(row1) + thread_id;
						end1 = M.m_ptr_in_col_id_gpu(row1+1);
					}

					if(row+2*blockSize < M.m_num_of_rows)
					{
						row2 = row+2*blockSize;
						start2 = M.m_ptr_in_col_id_gpu(row2) + thread_id;
						end2 = M.m_ptr_in_col_id_gpu(row2+1);
					}

					if(row+3*blockSize < M.m_num_of_rows)
					{
						row3 = row+3*blockSize;
						start3 = M.m_ptr_in_col_id_gpu(row3) + thread_id;
						end3 = M.m_ptr_in_col_id_gpu(row3+1);
					}
					//main spmv loop
					for(int j0=start0, j1=start1, j2=start2, j3=start3;
							(j0<end0) || (j1<end1) || (j2<end2) || (j3<end3);
							j0=j0+ThreadsPerBlock*TILE, j1=j1+ThreadsPerBlock*TILE, j2=j2+ThreadsPerBlock*TILE, j3=j3+ThreadsPerBlock*TILE)
					{
						int col00=M.m_num_of_cols, col01=col00, col02=col00, col03=col00,
							col10=col00, col11=col00, col12=col00, col13=col00,
							col20=col00, col21=col00, col22=col00, col23=col00,
							col30=col00, col31=col00, col32=col00, col33=col00;


						value_type 	a00=-1, a01=-1, a02=-1, a03=-1,
									a10=-1, a11=-1, a12=-1, a13=-1,
									a20=-1, a21=-1, a22=-1, a23=-1,
									a30=-1, a31=-1, a32=-1, a33=-1;


						read_M1(j0+0, end0, col00, a00); read_M1(j0+1, end0, col01, a01); read_M1(j0+2, end0, col02, a02); read_M1(j0+3, end0, col03, a03);
						read_M1(j1+0, end1, col10, a10); read_M1(j1+1, end1, col11, a11); read_M1(j1+2, end1, col12, a12); read_M1(j1+3, end1, col13, a13);
						read_M1(j2+0, end2, col20, a20); read_M1(j2+1, end2, col21, a21); read_M1(j2+2, end2, col22, a22); read_M1(j2+3, end2, col23, a23);
						read_M1(j3+0, end3, col30, a30); read_M1(j3+1, end3, col31, a31); read_M1(j3+2, end3, col32, a32); read_M1(j3+3, end3, col33, a33);


						//for(int k=vector_id, l=gshared_id; k<N; k=k+VectorLength, l=l+CudaThreadPerBlock)	//alternate iteration to avoid bank conflicts.. but not working
						for(int k=vector_id; k<N; k=k+PVL*TILE)	//iterate over simd dimension - number of vectors. Assuming N to be even ALWAYS
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
							r0x00 = V(col00, k); r0x01 = V(col00, k+1*PVL); r0x02 = V(col00, k+2*PVL); r0x03 = V(col00, k+3*PVL);
							r0x10 = V(col01, k); r0x11 = V(col01, k+1*PVL); r0x12 = V(col01, k+2*PVL); r0x13 = V(col01, k+3*PVL);
							r0x20 = V(col02, k); r0x21 = V(col02, k+1*PVL); r0x22 = V(col02, k+2*PVL); r0x23 = V(col02, k+3*PVL);
							r0x30 = V(col03, k); r0x31 = V(col03, k+1*PVL); r0x32 = V(col03, k+2*PVL); r0x33 = V(col03, k+3*PVL);

							r1x00 = V(col10, k); r1x01 = V(col10, k+1*PVL); r1x02 = V(col10, k+2*PVL); r1x03 = V(col10, k+3*PVL);
							r1x10 = V(col11, k); r1x11 = V(col11, k+1*PVL); r1x12 = V(col11, k+2*PVL); r1x13 = V(col11, k+3*PVL);
							r1x20 = V(col12, k); r1x21 = V(col12, k+1*PVL); r1x22 = V(col12, k+2*PVL); r1x23 = V(col12, k+3*PVL);
							r1x30 = V(col13, k); r1x31 = V(col13, k+1*PVL); r1x32 = V(col13, k+2*PVL); r1x33 = V(col13, k+3*PVL);

							r2x00 = V(col20, k); r2x01 = V(col20, k+1*PVL); r2x02 = V(col20, k+2*PVL); r2x03 = V(col20, k+3*PVL);
							r2x10 = V(col21, k); r2x11 = V(col21, k+1*PVL); r2x12 = V(col21, k+2*PVL); r2x13 = V(col21, k+3*PVL);
							r2x20 = V(col22, k); r2x21 = V(col22, k+1*PVL); r2x22 = V(col22, k+2*PVL); r2x23 = V(col22, k+3*PVL);
							r2x30 = V(col23, k); r2x31 = V(col23, k+1*PVL); r2x32 = V(col23, k+2*PVL); r2x33 = V(col23, k+3*PVL);

							r3x00 = V(col30, k); r3x01 = V(col30, k+1*PVL); r3x02 = V(col30, k+2*PVL); r3x03 = V(col30, k+3*PVL);
							r3x10 = V(col31, k); r3x11 = V(col31, k+1*PVL); r3x12 = V(col31, k+2*PVL); r3x13 = V(col31, k+3*PVL);
							r3x20 = V(col32, k); r3x21 = V(col32, k+1*PVL); r3x22 = V(col32, k+2*PVL); r3x23 = V(col32, k+3*PVL);
							r3x30 = V(col33, k); r3x31 = V(col33, k+1*PVL); r3x32 = V(col33, k+2*PVL); r3x33 = V(col33, k+3*PVL);

							const int shared_id0 = thread_id * N *TILE + k;
							const int shared_id1 = shared_id0 + N;
							const int shared_id2 = shared_id0 + 2*N;
							const int shared_id3 = shared_id0 + 3*N;

							R_shared[shared_id0+0*PVL] += a00*r0x00 + a01*r0x10 + a02*r0x20 + a03*r0x30;
							R_shared[shared_id0+1*PVL] += a00*r0x01 + a01*r0x11 + a02*r0x21 + a03*r0x31;
							R_shared[shared_id0+2*PVL] += a00*r0x02 + a01*r0x12 + a02*r0x22 + a03*r0x32;
							R_shared[shared_id0+3*PVL] += a00*r0x03 + a01*r0x13 + a02*r0x23 + a03*r0x33;

							R_shared[shared_id1+0*PVL] += a10*r1x00 + a11*r1x10 + a12*r1x20 + a13*r1x30;
							R_shared[shared_id1+1*PVL] += a10*r1x01 + a11*r1x11 + a12*r1x21 + a13*r1x31;
							R_shared[shared_id1+2*PVL] += a10*r1x02 + a11*r1x12 + a12*r1x22 + a13*r1x32;
							R_shared[shared_id1+3*PVL] += a10*r1x03 + a11*r1x13 + a12*r1x23 + a13*r1x33;

							R_shared[shared_id2+0*PVL] += a20*r2x00 + a21*r2x10 + a22*r2x20 + a23*r2x30;
							R_shared[shared_id2+1*PVL] += a20*r2x01 + a21*r2x11 + a22*r2x21 + a23*r2x31;
							R_shared[shared_id2+2*PVL] += a20*r2x02 + a21*r2x12 + a22*r2x22 + a23*r2x32;
							R_shared[shared_id2+3*PVL] += a20*r2x03 + a21*r2x13 + a22*r2x23 + a23*r2x33;

							R_shared[shared_id3+0*PVL] += a30*r3x00 + a31*r3x10 + a32*r3x20 + a33*r3x30;
							R_shared[shared_id3+1*PVL] += a30*r3x01 + a31*r3x11 + a32*r3x21 + a33*r3x31;
							R_shared[shared_id3+2*PVL] += a30*r3x02 + a31*r3x12 + a32*r3x22 + a33*r3x32;
							R_shared[shared_id3+3*PVL] += a30*r3x03 + a31*r3x13 + a32*r3x23 + a33*r3x33;

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





#define read_M1(j, end, col, a)	\
		if(j*ThreadsPerBlock<end)	\
		{			\
			col = M.m_col_id_gpu(j*ThreadsPerBlock);	\
			a = M.m_csr_data_gpu(j*ThreadsPerBlock);	\
		}













template <int PVL, int LVL=N>	//LVL is not used in this routine. ALWAYS use PVL.
void ___csr_spmv_tile2___(const spm &M, const vectortype &V1, automicvectortype &R1, const int Blocks, const int ThreadsPerBlock)
{
	const int TILE=2;
	const Kokkos::TeamPolicy<Kokkos::Cuda> policy( Blocks , ThreadsPerBlock, PVL );
	typedef simd<PVL, LVL> simd_type;
	typedef scalar_for_simd<LVL/PVL> Scalar;

	typedef Kokkos::View<simd_type*, Kokkos::LayoutRight, Kokkos::Cuda> simd_vectortype;
	simd_vectortype simd_V = simd_vectortype(reinterpret_cast<simd_type*>(V1.data()), V1.extent(0));
	simd_vectortype simd_R = simd_vectortype(reinterpret_cast<simd_type*>(R1.data()), R1.extent(0));

	//rows are distributed among teams. Each team processes 1 row in 1 iteration.
	Kokkos::parallel_for(policy, [=] __device__ (team_member & thread )
	{
		const int numberofthreads = thread.league_size() * thread.team_size() * TILE;
		Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, ThreadsPerBlock), [=] __device__ (const int& thread_id)
		{
				const int gthread_id = thread.league_rank() * thread.team_size() + thread_id;
				const int max_rows = M.m_num_of_rows;

				for(int row=gthread_id*TILE; row<max_rows; row=row+numberofthreads)	//iterate over rows. Each team processes 1 row at a time.
				{
					const int row0 = row;
					const int start0 = M.m_ptr_in_col_id_gpu(row);
					const int end0 = M.m_ptr_in_col_id_gpu(row+1);

					int row1=-1, start1=-1, end1=-1;

					if(row+1 < max_rows)
					{
						row1 = row + 1;
						start1 = M.m_ptr_in_col_id_gpu(row1);
						end1 = M.m_ptr_in_col_id_gpu(row1+1);
					}

					//main spmv loop
					Scalar temp0 = 0.0,temp1 = 0.0;
					for(int j0=start0, j1=start1; (j0<end0) || (j1<end1); j0++,j1++)
					{
						int col0=M.m_num_of_cols, col1=M.m_num_of_cols;
						value_type a0=0.0, a1=0.0;

						if(j0<end0) 	{ col0 = M.m_col_id_gpu(j0);	a0=M.m_csr_data_gpu(j0); 	}//column
						if(j1<end1)		{ col1 = M.m_col_id_gpu(j1);	a1=M.m_csr_data_gpu(j1); 	};	//column

						temp0 += simd_V(col0)*a0;
						temp1 += simd_V(col1)*a1;

					}
					simd_R(row0) = temp0;
					if(row1<max_rows) simd_R(row1) = temp1;
				}

		});
	});
}

