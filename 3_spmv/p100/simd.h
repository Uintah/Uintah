#ifndef SPARSEMATVEC_SIMD_H_
#define SPARSEMATVEC_SIMD_H_

#include <stdlib.h>
#include <Kokkos_Core.hpp>

typedef double value_type;

//added only bare minimum operators needed for spmv.

template <int EL=1>	//PVL and LVL: Physical and logical vector lengths, EL: elements per vector lane
struct scalar_for_simd;


template <int PVL=32, int LVL=32, int EL=LVL/PVL>	//PVL and LVL: Physical and logical vector lengths, EL: elements per vector lane
struct simd
{
	value_type d[LVL];

	__forceinline__ __host__ __device__ simd(){};
	simd(const simd &a)=default;
	__forceinline__ __host__ __device__    simd& operator= (const value_type &a) {
#pragma unroll(EL)
	  for(int i=0; i<EL; i++)
		  d[i*blockDim.x + threadIdx.x] = a;
	  return *this;
	}

	__forceinline__ __host__ __device__    simd& operator= (const scalar_for_simd<EL> &a) {
#pragma unroll(EL)
	  for(int i=0; i<EL; i++)
		  d[i*blockDim.x + threadIdx.x] = a.d[i];
	  return *this;
	}

	__forceinline__ __host__ __device__ scalar_for_simd<EL> operator*(const value_type &a)	//copy
	{
		scalar_for_simd<EL> temp;
#pragma unroll(EL)
		for(int i=0; i<EL; i++)
			temp.d[i] = d[i * blockDim.x + threadIdx.x] * a;

		return temp;
	}

	__forceinline__ __host__ __device__ simd& operator+=(const simd &a)
	{
#pragma unroll(EL)
		for(int i=0; i<EL; i++)
			d[i * blockDim.x + threadIdx.x] += a.d[i * blockDim.x + threadIdx.x];

		return *this;
	}

	__forceinline__ __host__ __device__ simd& operator+=(const scalar_for_simd<EL> &a)
	{
#pragma unroll(EL)
		for(int i=0; i<EL; i++)
			d[i * blockDim.x + threadIdx.x] += a.d[i];

		return *this;
	}
};

template <int EL>	//PVL and LVL: Physical and logical vector lengths, EL: elements per vector lane
struct scalar_for_simd
{
	value_type d[EL];

	__forceinline__ __host__ __device__ scalar_for_simd(){};
		scalar_for_simd(const scalar_for_simd &a)=default;

	__forceinline__ __host__ __device__ scalar_for_simd(const value_type &a){
#pragma unroll(EL)
	  for(int i=0; i<EL; i++)
		  d[i] = a;
	}

	__forceinline__ __host__ __device__ scalar_for_simd& operator=(const value_type &a){
#pragma unroll(EL)
	  for(int i=0; i<EL; i++)
		  d[i] = a;
	  return *this;
	}

	__forceinline__ __host__ __device__ scalar_for_simd& operator+=(const scalar_for_simd &a)
	{
#pragma unroll(EL)
		for(int i=0; i<EL; i++)
			d[i] += a.d[i];

		return *this;
	}


};
#endif
