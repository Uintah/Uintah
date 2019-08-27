#ifndef __KOKKOSBATCHED_VECTOR_CUDA_HPP__
#define __KOKKOSBATCHED_VECTOR_CUDA_HPP__

//File contains definitions of KokkosBatched::Experimental::Vector<SIMD<T>,l> for cuda
//similar to KokkosBatched_Vector_SIMD.hpp
//author: Damodar Sahasrabudhe (damodars@sci.utah.edu)

#include <Kokkos_Complex.hpp>
#include <KokkosBatched_Vector.hpp>
#define __CUDA_ARCH__

#if defined(__CUDA_ARCH__)



namespace KokkosBatched {
  namespace Experimental {

#define KOKKOSKERNELS_SIMD_ARITH_RETURN_TYPE(T,l) Vector< SIMD< T >, l >
#define KOKKOSKERNELS_SIMD_ARITH_RETURN_REFERENCE_TYPE(T,l) Vector< SIMD < T >, l> &

    template<typename T, int v = 0>
    struct TypeTraits {
      typedef T thread_private_type;
      typedef T team_private_type;
    };

    template<typename T, int l, int v>
    struct TypeTraits<Vector<SIMD<T>,l>, v> {
      typedef typename std::conditional<std::is_same<Kokkos::Impl::ActiveExecutionMemorySpace,Kokkos::HostSpace>::value,
                                        Vector<SIMD<T>,l>, T>::type thread_private_type;
      typedef typename std::conditional<std::is_same<Kokkos::Impl::ActiveExecutionMemorySpace,Kokkos::HostSpace>::value,
                                        Vector<SIMD<T>,l>, Vector<SIMD<T>,(l/v)+(l%v>0)> > team_private_type;
    };

    //using VectorTemp
    template<typename T, int l>
    class VectorTemp;

    template<typename T, int l>
    class Vector<SIMD<T>,l> {
    public:
      using type = Vector<SIMD<T>,l>;
      using value_type = T;
      using mag_type = typename Kokkos::Details::ArithTraits<T>::mag_type;

      enum : int { vector_length = l };
      enum : int { ele_per_thread = vector_length/vector_length};
      typedef value_type data_type[vector_length];

      KOKKOS_INLINE_FUNCTION
      static const char* label() { return "SIMD"; }

      template<typename,int>
      friend class Vector;

    //private:
      mutable data_type _data;

    public:
      KOKKOS_INLINE_FUNCTION Vector() {
#pragma unroll(ele_per_thread)
        for (int i=0;i<ele_per_thread;i=i+1)
          _data[i*blockDim.x + threadIdx.x] = 0;
      }
      template<typename ArgValueType>
      KOKKOS_INLINE_FUNCTION Vector(const ArgValueType &val) {
#pragma unroll(ele_per_thread)
        for (int i=0;i<ele_per_thread;i=i+1)
          _data[i*blockDim.x + threadIdx.x] = val;
      }
      template<typename ArgValueType>
      KOKKOS_INLINE_FUNCTION Vector(const Vector<SIMD<ArgValueType>,vector_length> &b) {
        static_assert(std::is_convertible<value_type,ArgValueType>::value, "input type is not convertible");
#pragma unroll(ele_per_thread)
        for (int i=0;i<ele_per_thread;i=i+1)
          _data[i*blockDim.x + threadIdx.x] = b[i*blockDim.x + threadIdx.x];
      }
      template<typename ArgValueType>
      KOKKOS_INLINE_FUNCTION Vector(const VectorTemp<ArgValueType,vector_length / vector_length> &b) {
#pragma unroll(ele_per_thread)
        for (int i=0; i<ele_per_thread; i++)
          _data[i*blockDim.x + threadIdx.x] = b[i];
      }

      KOKKOS_INLINE_FUNCTION
      value_type& operator[](const int &i) const {
        return _data[i*blockDim.x + threadIdx.x];
      }
    };

    template<typename T, int l>
    class VectorTemp
	{
	 public:
    	enum : int { ele_per_thread = l };
    	T _data[ele_per_thread];

        KOKKOS_INLINE_FUNCTION VectorTemp() {
#pragma unroll(ele_per_thread)
          for (int i=0; i<ele_per_thread;i=i+1)
            _data[i] = 0;
        }
        template<typename ArgValueType>
        KOKKOS_INLINE_FUNCTION VectorTemp(const ArgValueType &val) {
#pragma unroll(ele_per_thread)
        	for (int i=0; i<ele_per_thread;i=i+1)
            _data[i] = val;
        }
        template<typename ArgValueType, int vector_length>
        KOKKOS_INLINE_FUNCTION VectorTemp(const Vector<SIMD<ArgValueType>,vector_length> &b) {
#pragma unroll(ele_per_thread)
          for (int i=0; i<ele_per_thread;i=i+1)
            _data[i] = b._data[i*blockDim.x + threadIdx.x];
        }
        template<typename ArgValueType, int vector_length>
		KOKKOS_INLINE_FUNCTION VectorTemp(const VectorTemp<ArgValueType, vector_length> &b) {
#pragma unroll(ele_per_thread)
		  for (int i=0; i<ele_per_thread;i=i+1)
			_data[i] = b._data[i];
		}
	};


    template<typename T, int l, int ele_per_thread=l/l>
    KOKKOS_FORCEINLINE_FUNCTION
    static
    KOKKOSKERNELS_SIMD_ARITH_RETURN_REFERENCE_TYPE(T,l)
    operator += (Vector<SIMD<T>,l> &a, const VectorTemp<T,ele_per_thread> &b) {
#pragma unroll(ele_per_thread)
    	for (int i=0; i<ele_per_thread; i++)
    		a._data[i*blockDim.x + threadIdx.x]+=b._data[i];
      return a;
    }

    template<typename T, int l>	//here l = ele_per_thread
    KOKKOS_FORCEINLINE_FUNCTION
    static
	VectorTemp<T,l>&
    operator += (VectorTemp<T,l> &a, const VectorTemp<T,l> &b) {
#pragma unroll(l)
      for (int j=0; j<l; j++)
      	  	  a._data[j] = a._data[j] + b._data[j];
      return a;
    }


    template<typename T, int l, int ele_per_thread=l/l>
    KOKKOS_FORCEINLINE_FUNCTION
    static
	VectorTemp<T,ele_per_thread>
    operator * (const Vector<SIMD<T>,l> &a,  const Vector<SIMD<T>,l> &b) {
    	VectorTemp<T,ele_per_thread> r_val;
#pragma unroll(ele_per_thread)
      for (int i=0; i<ele_per_thread; i++)
        r_val._data[i] = a._data[i*blockDim.x + threadIdx.x] * b._data[i*blockDim.x + threadIdx.x];
      return r_val;
    }

  }
}


#endif //__CUDA_ARCH__

#endif //__KOKKOSBATCHED_VECTOR_CUDA_HPP__
