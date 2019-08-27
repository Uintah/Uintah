/*
3 modes of operation:
1. GPU on white:
	set DEFAULT_VECTOR_LENGTH=32 in gemm.cpp - corresponding to gpu WARP_SIZE
	use "typedef Kokkos::DefaultExecutionSpace KernelExecSpece;"
	compile with "make -gpu"

2. OpenMP (no GPU) on white:
	set DEFAULT_VECTOR_LENGTH=8 in gemm.cpp - corresponding to cpu vector length (assumed for now. 32 would also do.)
	use "typedef Kokkos::DefaultHostExecutionSpace KernelExecSpece;"
	compile with "make -gpu"

3. OpenMP on KNL on bowman:
	set DEFAULT_VECTOR_LENGTH=8 in gemm.cpp - corresponding to KNL vector length for double
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

#define c_mm(i,j)\
	c##i##j += a##i##0 * b##j##0 + a##i##1*b##j##1 + a##i##2*b##j##2;// + a##i##3*b##j##3;


#include <utility>
#include "simd_scalar.h"
#include "impl/Kokkos_Timer.hpp"

int N , M, NumTeams , ThreadsPerTeams;

//typedef Kokkos::LayoutLeft KernelLayout;
typedef Kokkos::LayoutRight KernelLayout;

//typedef Kokkos::DefaultHostExecutionSpace KernelExecSpece;
typedef Kokkos::DefaultExecutionSpace KernelExecSpece;

typedef typename Kokkos::TeamPolicy<KernelExecSpece>::member_type team_member ;

//Gemm with Vector

template <typename value_type>
struct Gemm
{
	typedef Kokkos::View<Vector<value_type>***,KernelLayout> view_vectorized;
	typedef Kokkos::View<value_type****,KernelLayout> plain_view;

	view_vectorized A, B, C;
	int N, M, NumTeams, ThreadsPerTeams, NN;

  	Gemm(plain_view a, plain_view b, plain_view c, int NumTeams , int ThreadsPerTeams, int N1, int M1): NumTeams(NumTeams), ThreadsPerTeams(ThreadsPerTeams), N(N1), M(M1/DEFAULT_VECTOR_LENGTH)
	{
		//printf("DEFAULT_VECTOR_LENGTH: %d, M: %d\n", DEFAULT_VECTOR_LENGTH, M);
		A = view_vectorized(reinterpret_cast<Vector<value_type>*>(a.data()), M, N, N);
		B = view_vectorized(reinterpret_cast<Vector<value_type>*>(b.data()), M, N, N);
		C = view_vectorized(reinterpret_cast<Vector<value_type>*>(c.data()), M, N, N);
		NN = N*N / TILE / TILE;	//NN per thread
		if(NN==0) NN = 1;	//assign at least 1 row
	}

	KOKKOS_INLINE_FUNCTION void operator() (const int& m) const 
	{
		for(int i = 0; i < N; i=i+TILE) 	//NN loop
		{
			for(int j = 0; j<N ; j=j+TILE) 	//columns loop
			{
				PerVL<double> c00(0.0), c01(0.0), c02(0.0),
						    c10(0.0), c11(0.0), c12(0.0),
						    c20(0.0), c21(0.0), c22(0.0);

				PerVL<double> a00(0.0), a01(0.0), a02(0.0),
						    a10(0.0), a11(0.0), a12(0.0),
						    a20(0.0), a21(0.0), a22(0.0);
				
				PerVL<double> b00(0.0), b01(0.0), b02(0.0),
						    b10(0.0), b11(0.0), b12(0.0),
						    b20(0.0), b21(0.0), b22(0.0);

				for(int k = 0; k<N ; k=k+TILE) 	
				{
					a00 = A(m, i, k); a01 = A(m, i, k+1); a02 = A(m, i, k+2); 
					a10 = A(m, i+1, k); a11 = A(m, i+1, k+1); a12 = A(m, i+1, k+2); 
					a20 = A(m, i+2, k); a21 = A(m, i+2, k+1); a22 = A(m, i+2, k+2); 

					b00 = B(m, j, k); b01 = B(m, j, k+1); b02 = B(m, j, k+2);
					b10 = B(m, j+1, k); b11 = B(m, j+1, k+1); b12 = B(m, j+1, k+2); 
					b20 = B(m, j+2, k); b21 = B(m, j+2, k+1); b22 = B(m, j+2, k+2); 

					c_mm(0,0); c_mm(0,1); c_mm(0,2);
					c_mm(1,0); c_mm(1,1); c_mm(1,2);
					c_mm(2,0); c_mm(2,1); c_mm(2,2);
				}
				C(m, i, j) = c00; C(m, i, j+1) = c01; C(m, i, j+2) = c02;
				C(m, i+1, j) = c10; C(m, i+1, j+1) = c11; C(m, i+1, j+2) = c12;
				C(m, i+2, j) = c20; C(m, i+2, j+1) = c21; C(m, i+2, j+2) = c22;
			}
		}

	}

};

void printView(Kokkos::View<double***,Kokkos::LayoutRight> a)
{
printf("\n\n\n");
  for(int m = 0; m<M ; m++)	//m for matrix
  {
	  for(int i = 0; i<N ; i++) 
	  {
		for(int j = 0; j<N ; j++) 
			printf("%0.f\t", a(m, i, j));
		printf("\n");
	  }
	printf("-----------\n");
  }
}

void printViewT(Kokkos::View<double***,KernelLayout> a)
{
printf("\n\n\n");
	for(int i = 0; i<N ; i++) 
	{
		for(int j = 0; j<N ; j++) 
		{
			for(int m = 0; m<M ; m++)	//m for matrix
				printf("%0.f\t", a(i, j, m));
			printf("\n");
		}
		printf("-----------\n");
	}
}


int main(int narg, char* args[]) {
  N = atoi(args[1]);	//matrix size
  M = atoi(args[2]);	//number of matrices
  //NumTeams = atoi(args[3]);	//number of teams
  //ThreadsPerTeams = atoi(args[4]);	//number of threads per team
  int M_vec = M/DEFAULT_VECTOR_LENGTH;	//number of "vectored" matrices

  //matrices are divided among teams and threads as M / (NumTeams*ThreadsPerTeams) )

  Kokkos::initialize(narg,args);
//printf("kokkos init\n");
  Kokkos::View<double***,Kokkos::LayoutRight> a("mat_A",M,N,N), b("mat_B",M,N,N), c("mat_C",M,N,N);
  Kokkos::View<double****,KernelLayout> at("mat_AT",M_vec,N,N, DEFAULT_VECTOR_LENGTH), bt("mat_BT",M_vec,N,N, DEFAULT_VECTOR_LENGTH), ct("mat_CT",M_vec,N,N, DEFAULT_VECTOR_LENGTH);
//printf("view allocated\n");

  Kokkos::View<double****,KernelLayout> a_d("mat_d",M_vec,N,N, DEFAULT_VECTOR_LENGTH), b_d("mat_d",M_vec,N,N, DEFAULT_VECTOR_LENGTH), c_d("mat_d",M_vec,N,N, DEFAULT_VECTOR_LENGTH);
//printf("view allocated\n");


//init:
double exec_time=0.0;
Kokkos::Impl::Timer timer;
	for(int it = -3; it<30; it++)
	{
		Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, M), [&](int m){	//m for matrix
			  for(int i = 0; i<N ; i++) 
				for(int j = 0; j<N ; j++) 
				{
					a(m, i, j)=i*N + j;
					b(m, i, j)=1.0;			
				}
			});
		//printView(a);
		//printf("view init\n");

	
		//transpose
		Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, M_vec), [&](int m){
			int vb = m * DEFAULT_VECTOR_LENGTH;
			for(int i = 0; i<N ; i++)
			for(int j = 0; j<N ; j++) 
				for(int v = 0; v<DEFAULT_VECTOR_LENGTH ; v++)	//v for VL
				{
					at(m, i, j, v) = a(vb + v, i, j);	//rearrange a to bring matrix dim to first dim	
		 			bt(m, j, i, v) = b(vb + v, i, j);	//take b transpose and also bring matrix dim to first dim. 
					ct(m, i, j, v)=0.0;
				}
		});
		  //printViewT(at);

         Kokkos::deep_copy(a_d, at);
         Kokkos::deep_copy(b_d, bt);
         Kokkos::deep_copy(c_d, 0);


		//kernel
		  //printf("starting kernel, getVectorLength(): %d\n", getVectorLength<KernelExecSpece>());
		  
		  //const Kokkos::TeamPolicy<KernelExecSpece> policy( NumTeams , ThreadsPerTeams, getVectorLength<KernelExecSpece>());
		  Gemm<double> gemm(a_d, b_d, c_d, NumTeams, ThreadsPerTeams, N, M);
		  KernelExecSpece::fence();
		  
		  timer.reset();

		  Kokkos::parallel_for( Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, M_vec), gemm );
		  
		  KernelExecSpece::fence();

		  double it_time = timer.seconds();
		  
		  if(it>=0)
			exec_time += it_time;
	}
	exec_time = exec_time / 30.0;
  printf("kernel completed, verifying\n");

       Kokkos::deep_copy(ct, c_d);


//rearrange c to individual matrices
Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, M_vec), [&](int m){
	int vb = m * DEFAULT_VECTOR_LENGTH;
	  for(int i = 0; i<N ; i++) 
		for(int j = 0; j<N ; j++) 
			for(int v = 0; v<DEFAULT_VECTOR_LENGTH ; v++)
			c(vb + v, i, j) = ct(m, i, j, v);	
});

	//printView(c);
	//printViewT(ct);

//verify
Kokkos::parallel_for(Kokkos::RangePolicy<Kokkos::DefaultHostExecutionSpace>(0, M), [&](int m){
	  for(int i = 0; i<N ; i++) 
	  {
		for(int j = 0; j<N ; j++) 
		{
			double temp = 0.0;
			for(int k = 0; k<N ; k++) 
			{
				temp += a(m, i, k)*b(m, k, j);
			}
			//printf("%0.f\t", temp);
			if(c(m, i, j) != temp)
			{
				printf("\n\n-----------error-----------\n\n");
				exit(1);
			}
		}
		//printf("\n");
	  }
});

	double gflops = 2*N*N*N*M/exec_time; ///1024/1024/1024;
	printf("\n\nSuccess:\n N\tTeams\tTeamS\tVL\tele/VL\texec_time(s)\tFLOPS\n%d\t%d\t%d\t%d\t%d\t%f\t%e\n", N, NumTeams, ThreadsPerTeams, getVectorLength<KernelExecSpece>(), DEFAULT_VECTOR_LENGTH/getVectorLength<KernelExecSpece>(), exec_time, gflops);

  printf("\n");

  Kokkos::finalize();
}

