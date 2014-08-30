
#ifndef SpatialOps_test_stencil_helper_h
#define SpatialOps_test_stencil_helper_h

#include <spatialops/SpatialOpsTools.h>
#include <spatialops/Nebo.h>
#include <spatialops/OperatorDatabase.h>

#include <spatialops/structured/ExternalAllocators.h>
#include <spatialops/structured/SpatialField.h>
#include <spatialops/structured/stencil/FVStaggeredOperatorTypes.h>
#include <spatialops/structured/Grid.h>
#include <spatialops/structured/stencil/StencilBuilder.h>

#include <iostream>
#include <vector>

#include <test/TestHelper.h>

#ifdef ENABLE_CUDA
  #include <spatialops/structured/MemoryTypes.h>
  #include <spatialops/structured/MemoryWindow.h>
#endif

template< typename FieldT >
void function( const FieldT& x, const FieldT& y, const FieldT& z, FieldT& f ){
  using namespace SpatialOps;
  f <<= sin(x) + cos(y) + sin(z);
}
template< typename FieldT >
void function_der_x( const FieldT& x, const FieldT& y, const FieldT& z, FieldT& f ){
  using namespace SpatialOps;
  f <<= cos(x);
}
template< typename FieldT >
void function_der_y( const FieldT& x, const FieldT& y, const FieldT& z, FieldT& f ){
  using namespace SpatialOps;
  f <<= -sin(y);
}
template< typename FieldT >
void function_der_z( const FieldT& x, const FieldT& y, const FieldT& z, FieldT& f ){
  using namespace SpatialOps;
  f <<= cos(z);
}
template< typename FieldT >
void function_der2_x( const FieldT& x, const FieldT& y, const FieldT& z, FieldT& f ){
  using namespace SpatialOps;
  f <<= -sin(x);
}
template< typename FieldT >
void function_der2_y( const FieldT& x, const FieldT& y, const FieldT& z, FieldT& f ){
  using namespace SpatialOps;
  f <<= -cos(y);
}
template< typename FieldT >
void function_der2_z( const FieldT& x, const FieldT& y, const FieldT& z, FieldT& f ){
  using namespace SpatialOps;
  f <<= -sin(z);
}

template< typename FieldT, typename OpT, typename DirT, int IsSrcField >
struct FuncEvaluator;

template< typename FieldT, typename DirT, int IsSrcField >
struct FuncEvaluator<FieldT,SpatialOps::Interpolant,DirT,IsSrcField>{
  static void value( const FieldT& x, const FieldT& y, const FieldT& z, FieldT& f ){ function(x,y,z,f); }
};
template< typename FieldT, typename DirT, int IsSrcField >
struct FuncEvaluator<FieldT,SpatialOps::InterpolantX,DirT,IsSrcField>{
  static void value( const FieldT& x, const FieldT& y, const FieldT& z, FieldT& f ){ function(x,y,z,f); }
};
template< typename FieldT, typename DirT, int IsSrcField >
struct FuncEvaluator<FieldT,SpatialOps::InterpolantY,DirT,IsSrcField>{
  static void value( const FieldT& x, const FieldT& y, const FieldT& z, FieldT& f ){ function(x,y,z,f); }
};
template< typename FieldT, typename DirT, int IsSrcField >
struct FuncEvaluator<FieldT,SpatialOps::InterpolantZ,DirT,IsSrcField>{
  static void value( const FieldT& x, const FieldT& y, const FieldT& z, FieldT& f ){ function(x,y,z,f); }
};

template< typename FieldT >
struct FuncEvaluator<FieldT,SpatialOps::Gradient,SpatialOps::XDIR,0>{
  static void value( const FieldT& x, const FieldT& y, const FieldT& z, FieldT& f ){ function_der_x(x,y,z,f); }
};
template< typename FieldT >
struct FuncEvaluator<FieldT,SpatialOps::Gradient,SpatialOps::XDIR,1>{
  static void value( const FieldT& x, const FieldT& y, const FieldT& z, FieldT& f ){ function(x,y,z,f); }
};
template< typename FieldT >
struct FuncEvaluator<FieldT,SpatialOps::GradientX,SpatialOps::XDIR,0>{
  static void value( const FieldT& x, const FieldT& y, const FieldT& z, FieldT& f ){ function_der_x(x,y,z,f); }
};
template< typename FieldT >
struct FuncEvaluator<FieldT,SpatialOps::GradientX,SpatialOps::XDIR,1>{
  static void value( const FieldT& x, const FieldT& y, const FieldT& z, FieldT& f ){ function(x,y,z,f); }
};

template< typename FieldT >
struct FuncEvaluator<FieldT,SpatialOps::Gradient,SpatialOps::YDIR,0>{
  static void value( const FieldT& x, const FieldT& y, const FieldT& z, FieldT& f ){ function_der_y(x,y,z,f); }
};
template< typename FieldT >
struct FuncEvaluator<FieldT,SpatialOps::Gradient,SpatialOps::YDIR,1>{
  static void value( const FieldT& x, const FieldT& y, const FieldT& z, FieldT& f ){ function(x,y,z,f); }
};
template< typename FieldT >
struct FuncEvaluator<FieldT,SpatialOps::GradientY,SpatialOps::YDIR,0>{
  static void value( const FieldT& x, const FieldT& y, const FieldT& z, FieldT& f ){ function_der_y(x,y,z,f); }
};
template< typename FieldT >
struct FuncEvaluator<FieldT,SpatialOps::GradientY,SpatialOps::YDIR,1>{
  static void value( const FieldT& x, const FieldT& y, const FieldT& z, FieldT& f ){ function(x,y,z,f); }
};

template< typename FieldT >
struct FuncEvaluator<FieldT,SpatialOps::Gradient,SpatialOps::ZDIR,0>{
  static void value( const FieldT& x, const FieldT& y, const FieldT& z, FieldT& f ){ function_der_z(x,y,z,f); }
};
template< typename FieldT >
struct FuncEvaluator<FieldT,SpatialOps::Gradient,SpatialOps::ZDIR,1>{
  static void value( const FieldT& x, const FieldT& y, const FieldT& z, FieldT& f ){ function(x,y,z,f); }
};
template< typename FieldT >
struct FuncEvaluator<FieldT,SpatialOps::GradientZ,SpatialOps::ZDIR,0>{
  static void value( const FieldT& x, const FieldT& y, const FieldT& z, FieldT& f ){ function_der_z(x,y,z,f); }
};
template< typename FieldT >
struct FuncEvaluator<FieldT,SpatialOps::GradientZ,SpatialOps::ZDIR,1>{
  static void value( const FieldT& x, const FieldT& y, const FieldT& z, FieldT& f ){ function(x,y,z,f); }
};

template< typename FieldT >
struct FuncEvaluator<FieldT,SpatialOps::Divergence,SpatialOps::XDIR,0>{
  static void value( const FieldT& x, const FieldT& y, const FieldT& z, FieldT& f ){ function_der2_x(x,y,z,f); }
};
template< typename FieldT >
struct FuncEvaluator<FieldT,SpatialOps::Divergence,SpatialOps::XDIR,1>{
  static void value( const FieldT& x, const FieldT& y, const FieldT& z, FieldT& f ){ function_der_x(x,y,z,f); }
};

template< typename FieldT >
struct FuncEvaluator<FieldT,SpatialOps::Divergence,SpatialOps::YDIR,0>{
  static void value( const FieldT& x, const FieldT& y, const FieldT& z, FieldT& f ){ function_der2_y(x,y,z,f); }
};
template< typename FieldT >
struct FuncEvaluator<FieldT,SpatialOps::Divergence,SpatialOps::YDIR,1>{
  static void value( const FieldT& x, const FieldT& y, const FieldT& z, FieldT& f ){ function_der_y(x,y,z,f); }
};

template< typename FieldT >
struct FuncEvaluator<FieldT,SpatialOps::Divergence,SpatialOps::ZDIR,0>{
  static void value( const FieldT& x, const FieldT& y, const FieldT& z, FieldT& f ){ function_der2_z(x,y,z,f); }
};
template< typename FieldT >
struct FuncEvaluator<FieldT,SpatialOps::Divergence,SpatialOps::ZDIR,1>{
  static void value( const FieldT& x, const FieldT& y, const FieldT& z, FieldT& f ){ function_der_z(x,y,z,f); }
};

//====================================================================

template< typename FieldT >
double interior_norm( const FieldT& f1, const FieldT& f2 )
{
  using namespace SpatialOps;
  const FieldT f1interior( f1.window_without_ghost(), f1 );
  const FieldT f2interior( f2.window_without_ghost(), f2 );
  const double l2 = field_norm( f1interior-f2interior ) / field_norm(f2interior);

//   const double linf = field_max( abs(f1interior-f2interior) );
//   const double l1 = field_sum( abs(f1interior-f2interior) ) / f1.window_without_ghost().npts();
  return l2;
}

//====================================================================

bool check_convergence( const std::vector<double>& spacings,
                        const std::vector<double>& norms,
                        const double order )
{
  const size_t n = spacings.size();
  std::vector<double> errRatio(n,0), calcOrder(n-1,0);
  for( size_t i=0; i<n; ++i ){
    //const double ideal = norms[0] * std::pow( spacings[i] / spacings[0], order );
    errRatio[i] = norms[i] / norms[0];
  }
  for( size_t i=0; i<n-1; ++i ){
    const double num = log10( errRatio[i+1]/errRatio[i] );
    const double den = log10( spacings[i]/spacings[i+1] );
    calcOrder[i] = std::abs(num/den);
  }
  const double maxOrd = *std::max_element( calcOrder.begin(), calcOrder.end() );

  if( maxOrd < 0.99*order ){
    return false;
  }
  return true;
}

//===================================================================

template< typename OpT, typename DirT >
double apply_stencil( const SpatialOps::IntVec& npts,
                      const double length,
                      const bool bcPlus[3] )
{
  using namespace SpatialOps;
  typedef typename OpT::type           OpType;
  typedef typename OpT::SrcFieldType   SrcT;
  typedef typename OpT::DestFieldType  DestT;

  const GhostData sg(1);
  const GhostData dg(1);
  const BoundaryCellInfo sbc = BoundaryCellInfo::build< SrcT>( bcPlus[0],bcPlus[1],bcPlus[2] );
  const BoundaryCellInfo dbc = BoundaryCellInfo::build<DestT>( bcPlus[0],bcPlus[1],bcPlus[2] );
  const MemoryWindow smw = get_window_with_ghost( npts, sg, sbc );
  const MemoryWindow dmw = get_window_with_ghost( npts, dg, dbc );


  SrcT   src(smw,sbc,sg,NULL), xs(smw,sbc,sg,NULL), ys(smw,sbc,sg,NULL), zs(smw,sbc,sg,NULL);
  DestT dest(dmw,dbc,dg,NULL), xd(dmw,dbc,dg,NULL), yd(dmw,dbc,dg,NULL), zd(dmw,dbc,dg,NULL), destExact(dmw,dbc,dg,NULL);

  const Grid grid( npts, DoubleVec(length,length,length) );

  grid.set_coord<XDIR>( xs );  grid.set_coord<YDIR>( ys );  grid.set_coord<ZDIR>( zs );
  grid.set_coord<XDIR>( xd );  grid.set_coord<YDIR>( yd );  grid.set_coord<ZDIR>( zd );

  FuncEvaluator<SrcT,OpType,DirT,1>::value( xs, ys, zs, src      );
  FuncEvaluator<DestT,OpType,DirT,0>::value( xd, yd, zd, destExact);

  // resolve the operator
  OperatorDatabase opdb;
  build_stencils( npts[0], npts[1], npts[2], length, length, length, opdb );
  const OpT* const op = opdb.retrieve_operator<OpT>();

  #ifdef ENABLE_CUDA
  const short int devIdx = GPU_INDEX;
  const StorageMode mode = InternalStorage;
  DestT gpuDest( dmw, dbc, dg, NULL, mode, devIdx );
  #endif

  #ifdef ENABLE_CUDA
    src.add_device( GPU_INDEX );
    src.set_device_as_active( GPU_INDEX );
    op->apply_to_field( src, gpuDest );
    gpuDest.add_device( CPU_INDEX );
    return interior_norm( gpuDest, destExact );
  #else
    op->apply_to_field( src, dest );
    return interior_norm( dest, destExact );
  #endif
}

//===================================================================

template< typename OpT, typename SrcT, typename DestT, typename DirT >
bool
run_convergence( SpatialOps::IntVec npts,
                 const bool bcPlus[3],
                 const double length,
                 const double expectedOrder )
{
  using namespace SpatialOps;
  typedef typename SpatialOps::OperatorTypeBuilder<OpT,SrcT,DestT>::type  Op;

  const size_t nrefine = 5;

  // directional index for convergence test
  const int ix = (IsSameType<DirT,XDIR>::result) ? 0 : (IsSameType<DirT,YDIR>::result) ? 1 : 2;

  std::vector<double> norms(nrefine,0.0), spacings(nrefine,0.0);

  const int n = npts[ix];

  for( size_t icount=0; icount<nrefine; ++icount ){
    spacings[icount] = length/( (icount+1)*n );
    npts[ix] = n * (icount+1);
    norms[icount] = apply_stencil<Op,DirT>( npts, length, bcPlus );
  }

  return check_convergence( spacings, norms, expectedOrder );
}

template< typename OpT, typename SrcT, typename DestT, typename Dir1T, typename Dir2T >
bool
run_convergence( SpatialOps::IntVec npts,
                 const bool bcPlus[3],
                 const double length,
                 const double expectedOrder )
{
  using namespace SpatialOps;
  typedef typename SpatialOps::OperatorTypeBuilder<OpT,SrcT,DestT>::type  Op;

  const size_t nrefine = 5;

  // directional index for convergence test
  const int ix1 = (IsSameType<Dir1T,XDIR>::result) ? 0 : (IsSameType<Dir1T,YDIR>::result) ? 1 : 2;
  const int ix2 = (IsSameType<Dir2T,XDIR>::result) ? 0 : (IsSameType<Dir2T,YDIR>::result) ? 1 : 2;

  std::vector<double> norms(nrefine,0.0), spacings(nrefine,0.0);

  const int n1 = npts[ix1];
  const int n2 = npts[ix2];

  for( size_t icount=0; icount<nrefine; ++icount ){
    spacings[icount] = length/( (icount+1)*n1 );
    npts[ix1] = n1 * (icount+1);
    npts[ix2] = n2 * (icount+1);
    norms[icount] = apply_stencil<Op,Dir1T>( npts, length, bcPlus );
  }

  return check_convergence( spacings, norms, expectedOrder );
}

//===================================================================

#endif // SpatialOps_test_stencil_helper_h
