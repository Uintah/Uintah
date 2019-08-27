#include <sys/time.h>
#include <Kokkos_Core.hpp>
#include <cstdio>


template <typename value_type, int vector_length=DEFAULT_VECTOR_LENGTH>
struct Vector;

//for cpu, PerVL and Vector are same
template <typename value_type, int vector_length=DEFAULT_VECTOR_LENGTH>
using PerVL = Vector<value_type, vector_length>;

template <typename value_type, int vector_length>
struct Vector	//struct to be instantiated inside parallel region. 
{
public:

	KOKKOS_INLINE_FUNCTION Vector(){}

	KOKKOS_INLINE_FUNCTION Vector(double a) {
	  for(int i=0; i<vector_length; i++)
	   m_thread_val[i]=a;
	}
	//-------------------------------------- operator = --------------------------------------------------------

	KOKKOS_INLINE_FUNCTION Vector operator= (const Vector &a)	{
	  for(int i=0; i<vector_length; i++)
	    m_thread_val[i] = a.m_thread_val[i];
	  return *this;
	}

	KOKKOS_INLINE_FUNCTION Vector operator= (const value_type &a)	{
	  for(int i=0; i<vector_length; i++)
	  m_thread_val[i] = a;
	  return *this;
	}

	//-------------------------------------- operator += and + --------------------------------------------------------

	KOKKOS_INLINE_FUNCTION Vector operator+= (const Vector &a)	{
	  for(int i=0; i<vector_length; i++)
		m_thread_val[i] += a.m_thread_val[i];
	  return *this;
	}


	KOKKOS_INLINE_FUNCTION Vector operator+ (const Vector &a)	{
 	  Vector temp;
	  for(int i=0; i<vector_length; i++)
		temp.m_thread_val[i] = m_thread_val[i] + a.m_thread_val[i];
	  return temp;
	}

	//-------------------------------------- operator -= and - --------------------------------------------------------

	KOKKOS_INLINE_FUNCTION Vector operator-= (const Vector &a)	{
	  for(int i=0; i<vector_length; i++)
		m_thread_val[i] -= a.m_thread_val[i];
	  return *this;
	}


	KOKKOS_INLINE_FUNCTION Vector operator- (const Vector &a)	{
 	  Vector temp;
	  for(int i=0; i<vector_length; i++)
		temp.m_thread_val[i] = m_thread_val[i] - a.m_thread_val[i];
	  return temp;
	}

	//-------------------------------------- operator *= and * --------------------------------------------------------

	KOKKOS_INLINE_FUNCTION Vector operator*= (const Vector &a)	{
	  for(int i=0; i<vector_length; i++)
		m_thread_val[index] *= a.m_thread_val[i];
	  return *this;
	}


	KOKKOS_INLINE_FUNCTION Vector operator* (const Vector &a)	{
 	  Vector temp;
	  for(int i=0; i<vector_length; i++)
		temp.m_thread_val[i] = m_thread_val[i] * a.m_thread_val[i];
	  return temp;
	}

	//-------------------------------------- operator /= and / --------------------------------------------------------

	KOKKOS_INLINE_FUNCTION Vector operator/= (const Vector &a)	{
	  for(int i=0; i<vector_length; i++)
		m_thread_val[i] /= a.m_thread_val[i];
	  return *this;
	}


	KOKKOS_INLINE_FUNCTION Vector operator/ (const Vector &a)	{
 	  Vector temp;
	  for(int i=0; i<vector_length; i++)
		temp.m_thread_val[i] = m_thread_val[i] / a.m_thread_val[i];
	  return temp;
	}

KOKKOS_INLINE_FUNCTION value_type operator[](int i){return m_thread_val[0];}

//private: 
	value_type m_thread_val[vector_length]; 	//value_typehere will be CPU_VECTOR_LENGTH instances of this struct. Each instance stores a single value corresponding to it's vector lane i.e. threadIdx.x. 
};




