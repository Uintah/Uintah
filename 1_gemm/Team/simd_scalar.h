#include <sys/time.h>
#include <Kokkos_Core.hpp>
#include <typeinfo>

#define WARP_SIZE 16
#define CPU_VECTOR_LENGTH 8

template <typename T>
inline int getVectorLength()
{
	if(typeid(T) == typeid(Kokkos::DefaultHostExecutionSpace))
		return CPU_VECTOR_LENGTH;
	else
		return WARP_SIZE;
}





#ifdef __CUDA_ARCH__

	#include "gpu.h"

#else

	
	#ifdef __PPC__
		// if ibm power pc, include cpu.h which has generic (non simd definitions)
		#include "cpu.h"

	#else
		//for intel architectures, overload std::simd. Need to add specialization for all datatypes. May be fix vector length

		#include "stk_simd/Simd.hpp"
		template <typename value_type, int vector_length=DEFAULT_VECTOR_LENGTH>
		struct Vector;
		
		template <int vector_length>
		struct Vector<double, vector_length> : public stk::simd::Double
		{
			Vector() : stk::simd::Double() {}
			Vector(double a) : stk::simd::Double(a) {}
			Vector(const Vector &a) : stk::simd::Double(a) {}
			Vector(const stk::simd::Double &a) : stk::simd::Double(a) {}

			KOKKOS_INLINE_FUNCTION Vector operator= (double a)	{
			  stk::simd::Double::operator=(a);
			  return *this;
			}	

			KOKKOS_INLINE_FUNCTION Vector operator= (stk::simd::Double a)	{
			  stk::simd::Double::operator=(a);
			  return *this;
			}

			KOKKOS_INLINE_FUNCTION Vector operator= (Vector a)	{
			  stk::simd::Double::operator=(a);
			  return *this;
			}

		};
		template <typename value_type, int vector_length=DEFAULT_VECTOR_LENGTH> using PerVL = stk::simd::Double;

		namespace stk {
		  namespace simd {
		    using stk::math::exp;
		    using stk::math::log;
		    using stk::math::sqrt;
		    using stk::math::min;
		    using stk::math::max;
		    using stk::math::cbrt;
		    using stk::math::tanh;
		  }
		}


	#endif

#endif

