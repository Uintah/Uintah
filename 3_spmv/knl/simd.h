#ifndef SPARSEMATVEC_SIMD_H_
#define SPARSEMATVEC_SIMD_H_

#include <type_traits>
#include <stdlib.h>
#include <Kokkos_Core.hpp>

#if !defined( KOKKOS_ENABLE_CUDA )  //This define comes from Kokkos itself.
  //Kokkos GPU is not included in this build.  Create some stub types so these types at least exist.
  namespace Kokkos {
    class Cuda {};
    class CudaSpace {};
  }
#define __forceinline__
#define __host__
#define __device__
struct blockDim_struct{
	int x{1};
}blockDim;
struct threadIdx_struct
{
	int x{0};
}threadIdx;

#elif !defined( KOKKOS_ENABLE_OPENMP )
  // For the Unified Scheduler + Kokkos GPU but not Kokkos OpenMP.
  // This logic may be temporary if all GPU functionality is merged into the Kokkos scheduler
  // and the Unified Scheduler is no longer used for GPU logic.  Brad P Jun 2018
  namespace Kokkos {
    class OpenMP {};
  }
#endif


#define Double
//#define Float

#ifdef Double
typedef double value_type;
typedef __m512d _simd_type;
#define _simd_set1 _mm512_set1_pd
#define _simd_mul _mm512_mul_pd
#define _simd_add _mm512_add_pd
#define _simd_fmadd _mm512_fmadd_pd
#endif

#ifdef Float
typedef float value_type;
typedef __m512 _simd_type;
#define _simd_set1 _mm512_set1_ps
#define _simd_mul _mm512_mul_ps
#define _simd_add _mm512_add_ps
#define _simd_fmadd _mm512_fmadd_ps
#endif

typedef Kokkos::OpenMP execution_space;



#define KNL_PVL 512/(8*sizeof(value_type))

//added only bare minimum operators needed for spmv.



//cpu defs
template <int PVL=KNL_PVL, int LVL=KNL_PVL, int EL=LVL/PVL>
struct simd_cpu
{
	_simd_type d[EL];


	simd_cpu(){};
	simd_cpu(const simd_cpu &a)=default;

	inline simd_cpu(const value_type &a) {
		_simd_type a_simd = _simd_set1(a);
	#pragma unroll(EL)
		  for(int i=0; i<EL; i++)
			  d[i] = a_simd;
	}

	inline simd_cpu& operator= (const value_type &a) {
	  _simd_type a_simd = _simd_set1(a);
#pragma unroll(EL)
	  for(int i=0; i<EL; i++)
		  d[i] = a_simd;
	  return *this;
	}

	simd_cpu operator*(const value_type &a)
	{
		_simd_type a_simd = _simd_set1(a);
		simd_cpu temp;
#pragma unroll(EL)
		for(int i=0; i<EL; i++)
			temp.d[i] = _simd_mul(d[i], a_simd);
		return temp;
	}

	simd_cpu& operator+=(const simd_cpu &a)
	{
#pragma unroll(EL)
		for(int i=0; i<EL; i++)
			d[i] = _simd_add(d[i], a.d[i]);
		return *this;
	}

	simd_cpu& fma_self(const simd_cpu &a, const value_type &b)
	{
		_simd_type b_simd = _simd_set1(b);
#pragma unroll(EL)
		for(int i=0; i<EL; i++)
			d[i] = _simd_fmadd(a.d[i], b_simd, d[i]);
		return *this;
	}

};




//gpu defs

template <int EL=1>	//PVL and LVL: Physical and logical vector lengths, EL: elements per vector lane
struct scalar_for_simd_gpu;




template <int PVL=32, int LVL=32, int EL=LVL/PVL>	//PVL and LVL: Physical and logical vector lengths, EL: elements per vector lane
struct simd_gpu
{
	value_type d[LVL];

	__forceinline__ __host__ __device__ simd_gpu(){};
	simd_gpu(const simd_gpu &a)=default;
	__forceinline__ __host__ __device__    simd_gpu& operator= (const value_type &a) {
#pragma unroll(EL)
	  for(int i=0; i<EL; i++)
		  d[i*blockDim.x + threadIdx.x] = a;
	  return *this;
	}

	__forceinline__ __host__ __device__    simd_gpu& operator= (const scalar_for_simd_gpu<EL> &a) {
#pragma unroll(EL)
	  for(int i=0; i<EL; i++)
		  d[i*blockDim.x + threadIdx.x] = a.d[i];
	  return *this;
	}

	__forceinline__ __host__ __device__ scalar_for_simd_gpu<EL> operator*(const value_type &a)	//copy
	{
		scalar_for_simd_gpu<EL> temp;
#pragma unroll(EL)
		for(int i=0; i<EL; i++)
			temp.d[i] = d[i * blockDim.x + threadIdx.x] * a;

		return temp;
	}

	__forceinline__ __host__ __device__ simd_gpu& operator+=(const simd_gpu &a)
	{
#pragma unroll(EL)
		for(int i=0; i<EL; i++)
			d[i * blockDim.x + threadIdx.x] += a.d[i * blockDim.x + threadIdx.x];

		return *this;
	}

	__forceinline__ __host__ __device__ simd_gpu& operator+=(const scalar_for_simd_gpu<EL> &a)
	{
#pragma unroll(EL)
		for(int i=0; i<EL; i++)
			d[i * blockDim.x + threadIdx.x] += a.d[i];

		return *this;
	}
};

template <int EL>	//PVL and LVL: Physical and logical vector lengths, EL: elements per vector lane
struct scalar_for_simd_gpu
{
	value_type d[EL];

	__forceinline__ __host__ __device__ scalar_for_simd_gpu(){};
		scalar_for_simd_gpu(const scalar_for_simd_gpu &a)=default;

	__forceinline__ __host__ __device__ scalar_for_simd_gpu(const value_type &a){
#pragma unroll(EL)
	  for(int i=0; i<EL; i++)
		  d[i] = a;
	}

	__forceinline__ __host__ __device__ scalar_for_simd_gpu& operator=(const value_type &a){
#pragma unroll(EL)
	  for(int i=0; i<EL; i++)
		  d[i] = a;
	  return *this;
	}

	__forceinline__ __host__ __device__ scalar_for_simd_gpu& operator+=(const scalar_for_simd_gpu &a)
	{
#pragma unroll(EL)
		for(int i=0; i<EL; i++)
			d[i] += a.d[i];

		return *this;
	}


};


// use defs as per namespace


template <int PVL, int LVL, int EL=LVL/PVL, typename exe_space=execution_space>
using simd =	typename std::conditional< std::is_same<exe_space, Kokkos::OpenMP>::value,
										   simd_cpu<PVL, LVL, EL>,
										   simd_gpu<PVL, LVL, EL>
										 >::type;




template <int PVL, int LVL, int EL=LVL/PVL, typename exe_space=execution_space>
using scalar_for_simd =	typename std::conditional< std::is_same<exe_space, Kokkos::OpenMP>::value,
										   simd_cpu<PVL, LVL, EL>,
										   scalar_for_simd_gpu<EL>
										 >::type;





#endif






