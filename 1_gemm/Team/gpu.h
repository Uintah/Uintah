#include <sys/time.h>
#include <Kokkos_Core.hpp>
#include <cstdio>


template <typename value_type, int vector_length=DEFAULT_VECTOR_LENGTH, int ele_per_vector_lane = vector_length/WARP_SIZE>
struct Vector;

template <typename value_type, int ele_per_vector_lane=DEFAULT_VECTOR_LENGTH/WARP_SIZE>
struct PerVL
{
	value_type data[ele_per_vector_lane];

	KOKKOS_INLINE_FUNCTION PerVL(){}

KOKKOS_INLINE_FUNCTION value_type operator[](int i){return data[0];}

	KOKKOS_INLINE_FUNCTION PerVL(const value_type &a){	
#pragma unroll(ele_per_vector_lane)
	for(int i=0; i<ele_per_vector_lane; i++)
	  data[i] = a;
	}
	//------------------------------- operator = ---------------------------------------------
	KOKKOS_INLINE_FUNCTION PerVL operator= (const Vector<value_type, ele_per_vector_lane*WARP_SIZE>& a)	{
#pragma unroll(ele_per_vector_lane)
	for(int i=0; i<ele_per_vector_lane; i++)
	  data[i] = a.m_thread_val[i*blockDim.x + threadIdx.x];
	  return *this;
	}

	KOKKOS_INLINE_FUNCTION PerVL operator= (const PerVL& a)	{
#pragma unroll(ele_per_vector_lane)
	for(int i=0; i<ele_per_vector_lane; i++)
	  data[i] = a.data[i];
	  return *this;
	}
	
	KOKKOS_INLINE_FUNCTION PerVL operator= (const value_type &a)	{
#pragma unroll(ele_per_vector_lane)
	for(int i=0; i<ele_per_vector_lane; i++)
	  data[i] = a;
	  return *this;
	}
	
	//------------------------------- operator + and += ---------------------------------------------

	KOKKOS_INLINE_FUNCTION PerVL operator+= (const PerVL &a)	{
#pragma unroll(ele_per_vector_lane)
	for(int i=0; i<ele_per_vector_lane; i++)
	  data[i] += a.data[i];
	  return *this;
	}

	KOKKOS_INLINE_FUNCTION PerVL operator+ (const Vector<value_type, ele_per_vector_lane*WARP_SIZE> &a)	{ return a + *this;}

	KOKKOS_INLINE_FUNCTION PerVL operator+ (const PerVL &a)	{
	PerVL temp;
#pragma unroll(ele_per_vector_lane)
	for(int i=0; i<ele_per_vector_lane; i++)
	  temp.data[i] = data[i] + a.data[i];
	  return temp;
	}

	//------------------------------- operator - and -= ---------------------------------------------

	KOKKOS_INLINE_FUNCTION PerVL operator-= (const PerVL &a)	{
#pragma unroll(ele_per_vector_lane)
	for(int i=0; i<ele_per_vector_lane; i++)
	  data[i] -= a.data[i];
	  return *this;
	}

	KOKKOS_INLINE_FUNCTION PerVL operator- (const Vector<value_type, ele_per_vector_lane*WARP_SIZE> &a)	{ return a - *this;}

	KOKKOS_INLINE_FUNCTION PerVL operator- (const PerVL &a)	{
	PerVL temp;
#pragma unroll(ele_per_vector_lane)
	for(int i=0; i<ele_per_vector_lane; i++)
	  temp.data[i] = data[i] - a.data[i];
	  return temp;
	}

	//------------------------------- operator * and *= ---------------------------------------------

	KOKKOS_INLINE_FUNCTION PerVL operator*= (const PerVL &a)	{
#pragma unroll(ele_per_vector_lane)
	for(int i=0; i<ele_per_vector_lane; i++)
	  data[i] *= a.data[i];
	  return *this;
	}

	KOKKOS_INLINE_FUNCTION PerVL operator* (const Vector<value_type, ele_per_vector_lane*WARP_SIZE> &a)	{ return a * (*this);}

	KOKKOS_INLINE_FUNCTION PerVL operator* (const PerVL &a)	{
	PerVL temp;
#pragma unroll(ele_per_vector_lane)
	for(int i=0; i<ele_per_vector_lane; i++)
	  temp.data[i] = data[i] * a.data[i];
	  return temp;
	}

	//------------------------------- operator / and /= ---------------------------------------------

	KOKKOS_INLINE_FUNCTION PerVL operator/= (const PerVL &a)	{
#pragma unroll(ele_per_vector_lane)
	for(int i=0; i<ele_per_vector_lane; i++)
	  data[i] /= a.data[i];
	  return *this;
	}

	KOKKOS_INLINE_FUNCTION PerVL operator/ (const Vector<value_type, ele_per_vector_lane*WARP_SIZE> &a)	{ return a / *this;}

	KOKKOS_INLINE_FUNCTION PerVL operator/ (const PerVL &a)	{
	PerVL temp;
#pragma unroll(ele_per_vector_lane)
	for(int i=0; i<ele_per_vector_lane; i++)
	  temp.data[i] = data[i] / a.data[i];
	  return temp;
	}

};


template <typename value_type, int vector_length, int ele_per_vector_lane>
struct Vector	//struct to be instantiated inside parallel region. 
{
public:
	typedef PerVL<value_type, vector_length/WARP_SIZE> PerVLType;

	KOKKOS_INLINE_FUNCTION Vector(){}

	KOKKOS_INLINE_FUNCTION Vector(const value_type &a) {
#pragma unroll(ele_per_vector_lane)
  	  for(int i=0; i<ele_per_vector_lane; i++)
	   m_thread_val[i*blockDim.x + threadIdx.x]=a;
	}

	//-------------------------------------- operator = --------------------------------------------------------

	KOKKOS_INLINE_FUNCTION Vector operator= (const Vector& a)	{
#pragma unroll(ele_per_vector_lane)
	  for(int i=0; i<ele_per_vector_lane; i++)
	    m_thread_val[i*blockDim.x + threadIdx.x] = a.m_thread_val[i*blockDim.x + threadIdx.x];
	  return *this;
	}

	KOKKOS_INLINE_FUNCTION Vector operator= (const PerVLType &a)	{
 #pragma unroll(ele_per_vector_lane)
	  for(int i=0; i<ele_per_vector_lane; i++)
		m_thread_val[i*blockDim.x + threadIdx.x] = a.data[i];
	  return *this;
	}

	KOKKOS_INLINE_FUNCTION Vector operator= (const value_type &a)	{
#pragma unroll(ele_per_vector_lane)
	  for(int i=0; i<ele_per_vector_lane; i++)
	  m_thread_val[i*blockDim.x + threadIdx.x] = a;
	  return *this;
	}

	//-------------------------------------- operator += and + --------------------------------------------------------

	KOKKOS_INLINE_FUNCTION Vector operator+= (const Vector &a)	{
#pragma unroll(ele_per_vector_lane)
	  for(int i=0; i<ele_per_vector_lane; i++)
		m_thread_val[i*blockDim.x + threadIdx.x] += a.m_thread_val[i*blockDim.x + threadIdx.x];
	  return *this;
	}

	KOKKOS_INLINE_FUNCTION Vector operator+= (const PerVLType &a){
 #pragma unroll(ele_per_vector_lane)
	  for(int i=0; i<ele_per_vector_lane; i++)
		m_thread_val[i*blockDim.x + threadIdx.x] += a.data[i];
	  return *this;
	}

	KOKKOS_INLINE_FUNCTION PerVLType operator+ (const Vector &a) const	{
 	  PerVLType temp;
#pragma unroll(ele_per_vector_lane)
	  for(int i=0; i<ele_per_vector_lane; i++)
		temp.data[i] = m_thread_val[i*blockDim.x + threadIdx.x] + a.m_thread_val[i*blockDim.x + threadIdx.x];
	  return temp;
	}

	KOKKOS_INLINE_FUNCTION PerVLType operator+ (const PerVLType &a)	{
 	  PerVLType temp;
#pragma unroll(ele_per_vector_lane)
	  for(int i=0; i<ele_per_vector_lane; i++)
		temp.data[i] = m_thread_val[i*blockDim.x + threadIdx.x] + a.data[i];
	  return temp;
	}

	//-------------------------------------- operator -= and - --------------------------------------------------------

	KOKKOS_INLINE_FUNCTION Vector operator-= (const Vector &a)	{
#pragma unroll(ele_per_vector_lane)
	  for(int i=0; i<ele_per_vector_lane; i++)
		m_thread_val[i*blockDim.x + threadIdx.x] -= a.m_thread_val[i*blockDim.x + threadIdx.x];
	  return *this;
	}

	KOKKOS_INLINE_FUNCTION Vector operator-= (const PerVLType &a) {
 #pragma unroll(ele_per_vector_lane)
	  for(int i=0; i<ele_per_vector_lane; i++)
		m_thread_val[i*blockDim.x + threadIdx.x] -= a.data[i];
	  return *this;
	}

	KOKKOS_INLINE_FUNCTION PerVLType operator- (const Vector &a) const	{
 	  PerVLType temp;
#pragma unroll(ele_per_vector_lane)
	  for(int i=0; i<ele_per_vector_lane; i++)
		temp.data[i] = m_thread_val[i*blockDim.x + threadIdx.x] - a.m_thread_val[i*blockDim.x + threadIdx.x];
	  return temp;
	}

	KOKKOS_INLINE_FUNCTION PerVLType operator- (const PerVLType &a)	{
 	  PerVLType temp;
#pragma unroll(ele_per_vector_lane)
	  for(int i=0; i<ele_per_vector_lane; i++)
		temp.data[i] = m_thread_val[i*blockDim.x + threadIdx.x] - a.data[i];
	  return temp;
	}

	//-------------------------------------- operator *= and * --------------------------------------------------------

	KOKKOS_INLINE_FUNCTION Vector operator*= (const Vector &a)	{
#pragma unroll(ele_per_vector_lane)
	  for(int i=0; i<ele_per_vector_lane; i++)
		m_thread_val[index] *= a.m_thread_val[i*blockDim.x + threadIdx.x];
	  return *this;
	}

	KOKKOS_INLINE_FUNCTION Vector operator*= (const PerVLType &a)	{
 #pragma unroll(ele_per_vector_lane)
	  for(int i=0; i<ele_per_vector_lane; i++)
		m_thread_val[i*blockDim.x + threadIdx.x] *= a.data[i];
	  return *this;
	}

	KOKKOS_INLINE_FUNCTION PerVLType operator* (const PerVLType &a) const	{
 	  PerVLType temp;
#pragma unroll(ele_per_vector_lane)
	  for(int i=0; i<ele_per_vector_lane; i++)
		temp.data[i] = m_thread_val[i*blockDim.x + threadIdx.x] * a.data[i];
	  return temp;
	}

	KOKKOS_INLINE_FUNCTION PerVLType operator* (const Vector &a)	{
 	  PerVLType temp;
#pragma unroll(ele_per_vector_lane)
	  for(int i=0; i<ele_per_vector_lane; i++)
		temp.data[i] = m_thread_val[i*blockDim.x + threadIdx.x] * a.m_thread_val[i*blockDim.x + threadIdx.x];
	  return temp;
	}


	//-------------------------------------- operator /= and / --------------------------------------------------------

	KOKKOS_INLINE_FUNCTION Vector operator/= (const Vector &a)	{
#pragma unroll(ele_per_vector_lane)
	  for(int i=0; i<ele_per_vector_lane; i++)
		m_thread_val[i*blockDim.x + threadIdx.x] /= a.m_thread_val[i*blockDim.x + threadIdx.x];
	  return *this;
	}

	KOKKOS_INLINE_FUNCTION Vector operator/= (const PerVLType &a){
 #pragma unroll(ele_per_vector_lane)
	  for(int i=0; i<ele_per_vector_lane; i++)
		m_thread_val[i*blockDim.x + threadIdx.x] /= a.data[i];
	  return *this;
	}

	KOKKOS_INLINE_FUNCTION PerVLType operator/ (const Vector &a) const	{
 	  PerVLType temp;
#pragma unroll(ele_per_vector_lane)
	  for(int i=0; i<ele_per_vector_lane; i++)
		temp.data[i] = m_thread_val[i*blockDim.x + threadIdx.x] / a.m_thread_val[i*blockDim.x + threadIdx.x];
	  return temp;
	}

	KOKKOS_INLINE_FUNCTION PerVLType operator/ (const PerVLType &a)	{
 	  PerVLType temp;
#pragma unroll(ele_per_vector_lane)
	  for(int i=0; i<ele_per_vector_lane; i++)
		temp.data[i] = m_thread_val[i*blockDim.x + threadIdx.x] / a.data[i];
	  return temp;
	}

//private: 
	value_type m_thread_val[vector_length]; 	//value_typehere will be WARP_SIZE instances of this struct. Each instance stores a single value corresponding to it's vector lane i.e. threadIdx.x. 
};


