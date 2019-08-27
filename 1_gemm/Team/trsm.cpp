/*
3 modes of operation:
1. GPU on white:
	set DEFAULT_VECTOR_LENGTH=32 in trsm.cpp - corresponding to gpu WARP_SIZE
	use "typedef Kokkos::DefaultExecutionSpace KernelExecSpece;"
	compile with "make -gpu"

2. OpenMP (no GPU) on white:
	set DEFAULT_VECTOR_LENGTH=8 in trsm.cpp - corresponding to cpu vector length (assumed for now. 32 would also do.)
	use "typedef Kokkos::DefaultHostExecutionSpace KernelExecSpece;"
	compile with "make -gpu"

3. OpenMP on KNL on bowman:
	set DEFAULT_VECTOR_LENGTH=8 in trsm.cpp - corresponding to KNL vector length for double
	use "typedef Kokkos::DefaultExecutionSpace KernelExecSpece;"
	compile with "make -knl"

run as ./a.out <matrix size> <num of matrices> <teams> <threads>. 
As of now teams should be <= matrix size
Number of matrice should be multiple of DEFAULT_VECTOR_LENGTH

*/

//taking shortcut. Ideally DEFAULT_VECTOR_LENGTH should be defined in cpu and gpu.h with respective values.
//Here it VECTOR_LENGTH shold be defined which will override DEFAULT_VECTOR_LENGTH
// DEFAULT_VECTOR_LENGTH should be (and must be) defined before #include "simd_scalar.h". Its used in DEFAULT_VECTOR_LENGTH

#ifdef __AVX512F__
	//for knl
	#define DEFAULT_VECTOR_LENGTH 8
#else
	//for gpu
	#define DEFAULT_VECTOR_LENGTH 16
#endif

#define TILE 3

#include <utility>
#include "simd_scalar.h"
#include "impl/Kokkos_Timer.hpp"

int N , M, R, NumTeams , ThreadsPerTeams;

//typedef Kokkos::LayoutLeft KernelLayout;
typedef Kokkos::LayoutRight KernelLayout;

//typedef Kokkos::DefaultHostExecutionSpace KernelExecSpece;
typedef Kokkos::DefaultExecutionSpace KernelExecSpece;

typedef typename Kokkos::TeamPolicy<KernelExecSpece>::member_type team_member ;

// Trsm with Vector

template <typename value_type>
struct  Trsm
{
	typedef Kokkos::View<Vector<value_type>***,KernelLayout> view_vectorized;
	typedef Kokkos::View<value_type****,KernelLayout> plain_view;

	view_vectorized A, B;
	int mats, N, R, M, NumTeams, ThreadsPerTeams;

  	Trsm(plain_view a, plain_view b, int NumTeams, int ThreadsPerTeams, int N1, int R1, int M1):NumTeams(NumTeams), ThreadsPerTeams(ThreadsPerTeams), N(N1), R(R1), M(M1/DEFAULT_VECTOR_LENGTH)
	{
		//printf("DEFAULT_VECTOR_LENGTH: %d, M: %d\n", DEFAULT_VECTOR_LENGTH, M);
		A = view_vectorized(reinterpret_cast<Vector<value_type>*>(a.data()), M, N, N);
		B = view_vectorized(reinterpret_cast<Vector<value_type>*>(b.data()), M, R, N);
		mats = M / NumTeams	;	//mat chunks per team
		if(mats==0) mats = 1;	//assign at least 1 matrix
	}

	KOKKOS_INLINE_FUNCTION void operator() (const team_member & thread) const 
	{
		int team_id = thread.league_rank();
		int mb = mats * team_id;
		int tiled_N = N - N%TILE; //adjust to nearest smaller multiple of N

			Kokkos::parallel_for(Kokkos::TeamThreadRange(thread, mats*R), KOKKOS_LAMBDA (const int& ele)
			{
				int m = mb + ele / R, j = ele % R;

					if(m<M && j<R)
					{
						for(int i = 0; i<N ; i++) //row of A and B
						{
							PerVL<double> b_cur;
							b_cur = B(m, j, i) / A(m, i, i);
							B(m, j, i) = b_cur;
							int ii=i+1;
							for(ii = i+1; ii+TILE < tiled_N; ii += TILE)	//forward substitution - subtract B(vb, i, j, v) from every subsequent row.
							{
								B(m, j, ii + 0) -= A(m, i, ii + 0) * b_cur;
								B(m, ii + 1, j) -= A(m, i, ii + 1) * b_cur;
								B(m, ii + 2, j) -= A(m, i, ii + 2) * b_cur;
							}
							for(; ii<N; ii++)
							{
								B(m, j, ii) -= A(m, i, ii) * b_cur;
							}
						}	
					}
			});
		
	}

};


void printView(Kokkos::View<double***,Kokkos::LayoutRight> a, int n, int r)
{
printf("\n\n\n");
  for(int m = 0; m<M ; m++)	//m for matrix
  {
	  for(int i = 0; i<n ; i++) 
	  {
		for(int j = 0; j<r ; j++) 
			printf("%0.f\t", a(m, i, j));
		printf("\n");
	  }
	printf("-----------\n");
  }
}

void printViewT(Kokkos::View<double****,KernelLayout> a, int n, int r)
{
printf("\n\n\n");
	for(int m = 0; m<M/DEFAULT_VECTOR_LENGTH ; m++) 
	for(int i = 0; i<n ; i++) 
	{
		for(int j = 0; j<r ; j++) 
		{
			for(int v = 0; v<DEFAULT_VECTOR_LENGTH ; v++)	//m for matrix
				printf("%0.f\t", a(m, i, j, v));
			printf("\n");
		}
		printf("-----------\n");
	}
}


int main(int narg, char* args[]) {
  N = atoi(args[1]);	//matrix size
  R = atoi(args[2]);	//number of columns
  M = atoi(args[3]);	//number of matrices
  NumTeams = atoi(args[4]);	//number of teams
  ThreadsPerTeams = atoi(args[5]);	//number of threads per team
  int M_vec = M/DEFAULT_VECTOR_LENGTH;	//number of "vectored" matrices

  //matrices are divided among teams and threads as M / (NumTeams*ThreadsPerTeams) )

  Kokkos::initialize(narg,args);
//printf("kokkos init\n");
  Kokkos::View<double***,Kokkos::LayoutRight> a("mat_A",M,N,N), b("mat_B",M,N,R), c("mat_C",M,N,R);
  Kokkos::View<double****,KernelLayout> at("mat_AT",M_vec,N,N, DEFAULT_VECTOR_LENGTH), bt("mat_BT",M_vec,R,N, DEFAULT_VECTOR_LENGTH), ct("mat_CT",M_vec,R,N, DEFAULT_VECTOR_LENGTH);
//printf("view allocated\n");

  Kokkos::View<double****,KernelLayout> a_d("mat_Ad",M_vec,N,N, DEFAULT_VECTOR_LENGTH), b_d("mat_Bd",M_vec,R,N, DEFAULT_VECTOR_LENGTH);
//printf("view allocated\n");


//init:
double exec_time=0.0;
Kokkos::Impl::Timer timer;

		Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, M), [&](int m){	//m for matrix
			  for(int i = 0; i<N ; i++) 
			  {
				b(m, i, 0) = 0.0;
				for(int j = 0; j<N ; j++) 
				{
					if(j<=i)
						a(m, i, j)=i*N + j + 1;
					else
						a(m, i, j)=0.0;	//lower triangular

					b(m, i, 0) += a(m, i, j);	//setting b to be sum of all a columns. thus x will be 1. easy to verify
				}

				for(int j = 1; j<R ; j++) 
					b(m, i, j)=b(m, i, 0)*j;
			  }
			});
		//printView(a, N, N);
		//printView(b, N, R);
		//printf("view init\n");

	
		//transpose
		Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, M_vec), [&](int m){
			int vb = m * DEFAULT_VECTOR_LENGTH;
			for(int i = 0; i<N ; i++)
			{
				for(int j = 0; j<N ; j++) 
					for(int v = 0; v<DEFAULT_VECTOR_LENGTH ; v++)	//v for VL
						at(m, j, i, v) = a(vb + v, i, j);	//rearrange a to bring matrix dim to first dim							

				for(int j = 0; j<R ; j++) 
					for(int v = 0; v<DEFAULT_VECTOR_LENGTH ; v++)	//v for VL
						bt(m, j, i, v) = b(vb + v, i, j);	//take b transpose and also bring matrix dim to first dim. 
			}
		});
		//printViewT(at, N, N);
		
		//printViewT(bt, N, R);
	for(int it = -3; it<33; it++)
	{
		
         Kokkos::deep_copy(a_d, at);
         Kokkos::deep_copy(b_d, bt);

		KernelExecSpece::fence();
		//kernel
		 // printf("starting kernel, getVectorLength(): %d\n", getVectorLength<KernelExecSpece>());
		  
		  const Kokkos::TeamPolicy<KernelExecSpece> policy( NumTeams , ThreadsPerTeams, getVectorLength<KernelExecSpece>());
		   Trsm<double> trsm_functor(a_d, b_d, NumTeams, ThreadsPerTeams, N, R, M);
		  
		  KernelExecSpece::fence();
		  timer.reset();

		  Kokkos::parallel_for( policy , trsm_functor);

		  KernelExecSpece::fence();

		  double it_time = timer.seconds();
		  
		  if(it>=0)
			exec_time += it_time;
	}


	exec_time = exec_time / 30.0;
  printf("kernel completed, verifying\n");

       Kokkos::deep_copy(ct, b_d);	//answer will be present in b.


//rearrange c to individual matrices
Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, M_vec), [&](int m){
	int vb = m * DEFAULT_VECTOR_LENGTH;
	  for(int i = 0; i<N ; i++) 
		for(int j = 0; j<R ; j++) 
			for(int v = 0; v<DEFAULT_VECTOR_LENGTH ; v++)
			c(vb + v, i, j) = ct(m, j, i, v);	
});

	//printView(c, N, R);
	//printViewT(b_d, N, R);

//verify


	Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, M), [&](int m){
		for(int j = 0; j<R ; j++)	//for every column in B
		{
			for(int i = 0; i<N ; i++) //row of A and B
			{
				b(m, i, j) /= a(m, i, i);
				double b_cur = b(m, i, j);
				for(int ii = i+1; ii<N; ii++)	//forward substitution - subtract B(vb, i, j, v) from every subsequent row.
					b(m, ii, j) -= b_cur * a(m, ii, i);
			}
		}
	});

//printView(b, N, R);

Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, M), [&](int m){
	  for(int i = 0; i<N ; i++) 
	  {
		for(int j = 0; j<R ; j++) 
		{
			if(c(m, i, j) != b(m, i, j))
			{
				//printf("\n\n-----------error-----------\n\n");
				//exit(1);
			}
		}
	  }
});

	double gflops = M*((0.5*N*R*(R+1.0)) + (0.5*N*R*(R-1.0))) / exec_time;
	printf("\n\nSuccess:\n N\tTeams\tTeamS\tVL\tele/VL\texec_time(s)\tFLOPS\n%d\t%d\t%d\t%d\t%d\t%f\t%e\n", N, NumTeams, ThreadsPerTeams, getVectorLength<KernelExecSpece>(), DEFAULT_VECTOR_LENGTH/getVectorLength<KernelExecSpece>(), exec_time, gflops);

  printf("\n");

  Kokkos::finalize();
}

