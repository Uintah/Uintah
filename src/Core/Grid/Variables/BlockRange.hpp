/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef UINTAH_HOMEBREW_BLOCK_RANGE_HPP
#define UINTAH_HOMEBREW_BLOCK_RANGE_HPP

#include <Core/Parallel/Parallel.h>
#include <sci_defs/cuda_defs.h>
#include <sci_defs/kokkos_defs.h>

#ifdef UINTAH_ENABLE_KOKKOS
#include <Kokkos_Core.hpp>
#endif //UINTAH_ENABLE_KOKKOS

#include <cstddef>


//The purpose of this file is to provide portability between Kokkos and non-Kokkos builds.
//For example, if a user calls a parallel_for loop but Kokkos is NOT provided, this will run the
//functor in a loop and also not use Kokkos views.  If Kokkos is provided, this creates a
//lambda expression and inside that it contains loops over the functor.  Kokkos Views are also used.
//At the moment we seek to only support regular CPU code, Kokkos OpenMP, and CUDA execution spaces,
//though it shouldn't be too difficult to expand it to others.  At the moment this doesn't extend it
//to CUDA kernels (without Kokkos), and that can get trickier (block/dim parameters) especially with
//regard to a parallel_reduce (many ways to "reduce" a value and return it back to host memory)


namespace Uintah {

class BlockRange
{
public:

  enum { rank = 3 };

  BlockRange(){}

  template <typename ArrayType>
  void setValues( ArrayType const & c0, ArrayType const & c1 )
  {
    for (int i=0; i<rank; ++i) {
      m_offset[i] = c0[i] < c1[i] ? c0[i] : c1[i];
      m_dim[i] =   (c0[i] < c1[i] ? c1[i] : c0[i]) - m_offset[i];
    }
  }

  template <typename ArrayType>
  BlockRange( ArrayType const & c0, ArrayType const & c1 )
  {
    setValues( c0, c1 );
  }

  BlockRange( const BlockRange& obj ) {
    for (int i=0; i<rank; ++i) {
      this->m_offset[i] = obj.m_offset[i];
      this->m_dim[i] = obj.m_dim[i];
    }
  }

#ifdef HAVE_CUDA
  template <typename ArrayType>
  void setValues(cudaStream_t* stream, ArrayType const & c0, ArrayType const & c1 )
  {
    for (int i=0; i<rank; ++i) {
      m_offset[i] = c0[i] < c1[i] ? c0[i] : c1[i];
      m_dim[i] =   (c0[i] < c1[i] ? c1[i] : c0[i]) - m_offset[i];
    }
    m_stream = stream;
  }

  template <typename ArrayType>
  BlockRange(cudaStream_t* stream, ArrayType const & c0, ArrayType const & c1 )
  {
    setValues( stream, c0, c1 );
  }



#endif

  int begin( int r ) const { return m_offset[r]; }
  int   end( int r ) const { return m_offset[r] + m_dim[r]; }

  size_t size() const
  {
    size_t result = 1u;
    for (int i=0; i<rank; ++i) {
      result *= m_dim[i];
    }
    return result;
  }

private:
  int m_offset[rank];
  int m_dim[rank];
#ifdef HAVE_CUDA
public:
  cudaStream_t* getStream() const { return m_stream; }
private:
  cudaStream_t* m_stream {nullptr};
#endif
};

template <typename Functor>
void serial_for( BlockRange const & r, const Functor & f )
{
  const int ib = r.begin(0); const int ie = r.end(0);
  const int jb = r.begin(1); const int je = r.end(1);
  const int kb = r.begin(2); const int ke = r.end(2);

  for (int k=kb; k<ke; ++k) {
  for (int j=jb; j<je; ++j) {
  for (int i=ib; i<ie; ++i) {
    f(i,j,k);
  }}}
}

#if defined( UINTAH_ENABLE_KOKKOS )

template <typename Functor>
void parallel_for( BlockRange const & r, const Functor & f )
{
  const int ib = r.begin(0); const int ie = r.end(0);
  const int jb = r.begin(1); const int je = r.end(1);
  const int kb = r.begin(2); const int ke = r.end(2);

#if defined( HAVE_CUDA )

  // Note: I can only seem to invoke functors with nvcc_wrapper.  I can't invoke lambdas.
  // I get obnoxious compiler errors.   At the moment this limits us to 1D loops.  Brad P Nov 23 2017.
  //Kokkos::parallel_for( Kokkos::RangePolicy<Kokkos::Cuda, int>(kb, ke).set_chunk_size(2), KOKKOS_LAMBDA(int k) {
  //Kokkos::parallel_for( Kokkos::RangePolicy<Kokkos::Cuda, int>(kb, ke).set_chunk_size(2), [=](int k) {
  //  for (int j=jb; j<je; ++j) {
  //  for (int i=ib; i<ie; ++i) {
  //    f(i,j,k);
  //  }}
  //});
  Kokkos::parallel_for( Kokkos::RangePolicy<Kokkos::Cuda, int>(kb, ke).set_chunk_size(2), f );
#else
  Kokkos::parallel_for( Kokkos::RangePolicy<Kokkos::OpenMP, int>(kb, ke).set_chunk_size(2), KOKKOS_LAMBDA(int k) {
    for (int j=jb; j<je; ++j) {
    for (int i=ib; i<ie; ++i) {
      f(i,j,k);
    }}
  });
#endif
}


//Runtime code has already started using parallel_for constructs.  These should NOT be executed on
//a GPU.  This function allows a developer to ensure the task only runs on CPU code.  Further, we
//will just run this without the use of Kokkos (this is so GPU builds don't require OpenMP as well).
template <typename Functor>
void parallel_for_cpu_only( BlockRange const & r, const Functor & f )
{
  const int ib = r.begin(0); const int ie = r.end(0);
  const int jb = r.begin(1); const int je = r.end(1);
  const int kb = r.begin(2); const int ke = r.end(2);

  for (int k=kb; k<ke; ++k) {
  for (int j=jb; j<je; ++j) {
  for (int i=ib; i<ie; ++i) {
    f(i,j,k);
  }}}
}

template <typename Functor, typename Option>
void parallel_for( BlockRange const & r, const Functor & f, const Option & op )
{
  const int ib = r.begin(0); const int ie = r.end(0);
  const int jb = r.begin(1); const int je = r.end(1);
  const int kb = r.begin(2); const int ke = r.end(2);

#if defined( HAVE_CUDA )
  Kokkos::parallel_for( Kokkos::RangePolicy<Kokkos::Cuda, int>(kb, ke).set_chunk_size(2), f );
  //Kokkos::parallel_for( Kokkos::RangePolicy<Kokkos::Cuda, int>(kb, ke).set_chunk_size(2), KOKKOS_LAMBDA(int k) {
  //    for (int j=jb; j<je; ++j) {
  //    for (int i=ib; i<ie; ++i) {
  //      f(op,i,j,k);
  //    }}
  //  });
#else
  Kokkos::parallel_for( Kokkos::RangePolicy<Kokkos::OpenMP, int>(kb, ke).set_chunk_size(2), KOKKOS_LAMBDA(int k) {
      for (int j=jb; j<je; ++j) {
      for (int i=ib; i<ie; ++i) {
        f(op,i,j,k);
      }}
    });
#endif
}


//This FunctorBuilder exists because I couldn't go the lambda approach.
//I was running into some conflict with Uintah/nvcc_wrapper/Kokkos/CUDA somewhere.
//So I went the alternative route and built a functor instead of building a lambda.
#if defined( HAVE_CUDA )
template < typename Functor, typename ReductionType >
struct FunctorBuilderReduce {
  //Functor& f = nullptr;

  //std::function is probably a wrong idea, CUDA doesn't support these.
  //std::function<void(int i, int j, int k, ReductionType & red)> f;
  int ib{0};
  int ie{0};
  int jb{0};
  int je{0};

  FunctorBuilderReduce(const BlockRange & r, const Functor & f) {
    ib = r.begin(0);
    ie = r.end(0);
    jb = r.begin(1);
    je = r.end(1);
  }
  void operator()(int k,  ReductionType & red) const {
    //const int ib = r.begin(0); const int ie = r.end(0);
    //const int jb = r.begin(1); const int je = r.end(1);

    for (int j=jb; j<je; ++j) {
      for (int i=ib; i<ie; ++i) {
        f(i,j,k,red);
      }
    }
  }
};
#endif

template <typename Functor, typename ReductionType>
void parallel_reduce_1D( BlockRange const & r, const Functor & f, ReductionType & red  ) {


#if !defined( HAVE_CUDA )
  if ( ! Parallel::usingDevice() ) {
    ReductionType tmp = red;
    Kokkos::RangePolicy<Kokkos::OpenMP> rangepolicy(r.begin(0), r.end(0));
    Kokkos::parallel_reduce( rangepolicy, f, tmp );
    red = tmp;
  }
#elif defined( HAVE_CUDA )
  //else {
    //This must be a single dimensional range policy, so use Kokkos::RangePolicy
    ReductionType *tmp;
    cudaMallocHost( (void**)&tmp, sizeof(ReductionType) );

    //No streaming, no launch bounds
    //Kokkos::RangePolicy<Kokkos::Cuda> rangepolicy(r.begin(0), r.end(0));
    //No streaming, launch bounds (512 gave 128 registers, 640 gave 96 registers)
    //Kokkos::RangePolicy<Kokkos::Cuda, Kokkos::LaunchBounds<512,1>> rangepolicy(r.begin(0), r.end(0));

    //Streaming
    Kokkos::Cuda instanceObject = Kokkos::Cuda( *(r.getStream()) );
 
    //Streaming, no launch bounds
    //Kokkos::RangePolicy<Kokkos::Cuda> rangepolicy(instanceObject, r.begin(0), r.end(0));
    //Streaming, launch bounds   
    Kokkos::RangePolicy<Kokkos::Cuda, Kokkos::LaunchBounds<512,1>> rangepolicy(instanceObject,  r.begin(0), r.end(0));

    Kokkos::parallel_reduce( rangepolicy, f, *tmp );  //TODO: Don't forget about these reduction values.
  //}
#endif



}

template <typename Functor, typename ReductionType>
void parallel_reduce_sum( BlockRange const & r, const Functor & f, ReductionType & red  )
{
  //const int ib = r.begin(0); const int ie = r.end(0);
  //const int jb = r.begin(1); const int je = r.end(1);
  //const int kb = r.begin(2); const int ke = r.end(2);

  ReductionType tmp = red;


#if defined( HAVE_CUDA )

    typedef typename Kokkos::MDRangePolicy<Kokkos::Cuda, Kokkos::Rank<3,
                                                         Kokkos::Iterate::Left,
                                                         Kokkos::Iterate::Left>
                                                         > MDPolicyType_3D;

    MDPolicyType_3D mdpolicy_3d( {{r.begin(0),r.begin(1),r.begin(2)}}, {{r.end(0),r.end(1),r.end(2)}} );

    Kokkos::parallel_reduce( mdpolicy_3d, f, tmp );


template <typename Functor, typename ReductionType>
void parallel_reduce_min( BlockRange const & r, const Functor & f, ReductionType & red  )
{
  const int ib = r.begin(0); const int ie = r.end(0);
  const int jb = r.begin(1); const int je = r.end(1);
  const int kb = r.begin(2); const int ke = r.end(2);

  ReductionType tmp = red;
  Kokkos::parallel_reduce( Kokkos::RangePolicy<Kokkos::OpenMP, int>(kb, ke).set_chunk_size(2), KOKKOS_LAMBDA(int k, ReductionType & tmp) {
    for (int j=jb; j<je; ++j) {
    for (int i=ib; i<ie; ++i) {
      f(i,j,k,tmp);
    }}
  }, Kokkos::Experimental::Min<ReductionType>(tmp));
  red = tmp;
};

#else
  typedef typename Kokkos::MDRangePolicy<Kokkos::OpenMP, Kokkos::Rank<3,
                                                       Kokkos::Iterate::Left,
                                                       Kokkos::Iterate::Left>
                                                       > MDPolicyType_3D;

  MDPolicyType_3D mdpolicy_3d( {{r.begin(0),r.begin(1),r.begin(2)}}, {{r.end(0),r.end(1),r.end(2)}} );

  Kokkos::parallel_reduce( mdpolicy_3d, f, tmp );

  //Kokkos::parallel_reduce( Kokkos::RangePolicy<Kokkos::OpenMP, int>(kb, ke).set_chunk_size(2), KOKKOS_LAMBDA(int k, ReductionType & tmp) {
  //  for (int j=jb; j<je; ++j) {
  //  for (int i=ib; i<ie; ++i) {
  //    f(i,j,k,tmp);
  //  }}
  //}, tmp);

#endif
  red = tmp;
}
#else //if !defined( UINTAH_ENABLE_KOKKOS )

template <typename Functor>
void parallel_for( BlockRange const & r, const Functor & f )
{
  const int ib = r.begin(0); const int ie = r.end(0);
  const int jb = r.begin(1); const int je = r.end(1);
  const int kb = r.begin(2); const int ke = r.end(2);

  for (int k=kb; k<ke; ++k) {
  for (int j=jb; j<je; ++j) {
  for (int i=ib; i<ie; ++i) {
    f(i,j,k);
  }}}
};

//Runtime code has already started using parallel_for constructs.  These should NOT be executed on
//a GPU.  This function allows a developer to ensure the task only runs on CPU code.
template <typename Functor>
void parallel_for_cpu_only( BlockRange const & r, const Functor & f )
{
  parallel_for( r, f );
}


template <typename Functor, typename Option>
void parallel_for( BlockRange const & r, const Functor & f, const Option& op )
{
  const int ib = r.begin(0); const int ie = r.end(0);
  const int jb = r.begin(1); const int je = r.end(1);
  const int kb = r.begin(2); const int ke = r.end(2);

  for (int k=kb; k<ke; ++k) {
  for (int j=jb; j<je; ++j) {
  for (int i=ib; i<ie; ++i) {
    f(op,i,j,k);
  }}}
};

template <typename Functor, typename ReductionType>
void parallel_reduce_sum( BlockRange const & r, const Functor & f, ReductionType & red  )
{
  const int ib = r.begin(0); const int ie = r.end(0);
  const int jb = r.begin(1); const int je = r.end(1);
  const int kb = r.begin(2); const int ke = r.end(2);

  ReductionType tmp = red;
  for (int k=kb; k<ke; ++k) {
  for (int j=jb; j<je; ++j) {
  for (int i=ib; i<ie; ++i) {
    f(i,j,k,tmp);
  }}}
  red = tmp;
};

template <typename Functor, typename ReductionType>
void parallel_reduce_min( BlockRange const & r, const Functor & f, ReductionType & red  )
{
  const int ib = r.begin(0); const int ie = r.end(0);
  const int jb = r.begin(1); const int je = r.end(1);
  const int kb = r.begin(2); const int ke = r.end(2);

  ReductionType tmp = red;
  for (int k=kb; k<ke; ++k) {
  for (int j=jb; j<je; ++j) {
  for (int i=ib; i<ie; ++i) {
    f(i,j,k,tmp);
  }}}
  red = tmp;
};

#endif


} // namespace Uintah

#endif // UINTAH_HOMEBREW_BLOCK_RANGE_HPP
