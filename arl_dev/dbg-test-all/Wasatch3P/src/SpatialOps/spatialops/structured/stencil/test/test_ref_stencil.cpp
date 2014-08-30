#include <spatialops/SpatialOpsTools.h>

#include <spatialops/structured/FVStaggeredFieldTypes.h>
#include <spatialops/OperatorDatabase.h>

#include <spatialops/structured/stencil/FVStaggeredOperatorTypes.h>
#include <spatialops/structured/stencil/StencilBuilder.h>

#include "ReferenceStencil.h"
#include <test/TestHelper.h>

#include <spatialops/Nebo.h>

#include <spatialops/structured/FieldComparisons.h>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

using namespace SpatialOps;

#include <stdexcept>
using std::cout;
using std::endl;

//--------------------------------------------------------------------

#ifndef NEBO_GPU_TEST
#define ACCEPTABLE_ERROR 0.0
#else
#define ACCEPTABLE_ERROR 1.0e-10
#endif

//--------------------------------------------------------------------

template<typename Field>
void initialize_field(Field & f)
{
  typename Field::iterator i = f.begin();
  MemoryWindow mw = f.window_with_ghost();
  int xLength = mw.extent(0);
  int yLength = mw.extent(1);
  int zLength = mw.extent(2);

  for(int x = 1; x <= xLength; x++) {
    for(int y = 1; y <= yLength; y++) {
      for(int z = 1; z <= zLength; z++) {
        *i = sin(x +
            y * xLength +
            z * xLength * yLength);
        i++;
      };
    };
  };
};

//--------------------------------------------------------------------

template<typename OpType, typename SrcType, typename DestType>
bool test_stencil2(const IntVec npts,
                   const OperatorDatabase & opdb,
                   bool bc[])
{
  //basic definitions:
  const GhostData  srcGhost(1);
  const GhostData destGhost(1);
  const BoundaryCellInfo  srcBC = BoundaryCellInfo::build< SrcType>(bc[0],bc[1],bc[2]);
  const BoundaryCellInfo destBC = BoundaryCellInfo::build<DestType>(bc[0],bc[1],bc[2]);
  const MemoryWindow mwSrc  = get_window_with_ghost(npts,  srcGhost,  srcBC);
  const MemoryWindow mwDest = get_window_with_ghost(npts, destGhost, destBC);

  SrcType  src ( mwSrc,   srcBC,  srcGhost, NULL );
  DestType ref ( mwDest, destBC, destGhost, NULL );
  DestType test( mwDest, destBC, destGhost, NULL );

  //initialize source field / zero out result fields:
  initialize_field(src);
  ref <<= 0.0;
  test <<= 0.0;

  //get operator:
  typedef typename SpatialOps::OperatorTypeBuilder<OpType,SrcType,DestType>::type Op;
  const Op* const op = opdb.retrieve_operator<Op>();

  //run reference:
  ref_stencil2_apply_to_field(op->coefs().get_coef(0),
                              op->coefs().get_coef(1),
                              src,
                              ref);

  //run operator:
  op->apply_to_field(src, test);

  return (field_equal(test, ref, ACCEPTABLE_ERROR));
};

//--------------------------------------------------------------------

template<typename OpType, typename SrcType, typename DestType>
bool test_stencil4(const IntVec npts,
                   const OperatorDatabase & opdb,
                   bool bc[])
{
  //basic definitions:
  const GhostData  srcGhost(1);
  const GhostData destGhost(1);
  const BoundaryCellInfo  srcBC = BoundaryCellInfo::build< SrcType>(bc[0],bc[1],bc[2]);
  const BoundaryCellInfo destBC = BoundaryCellInfo::build<DestType>(bc[0],bc[1],bc[2]);
  const MemoryWindow mwSrc  = get_window_with_ghost(npts,  srcGhost,  srcBC);
  const MemoryWindow mwDest = get_window_with_ghost(npts, destGhost, destBC);
  SrcType  src (mwSrc,   srcBC,  srcGhost, NULL);
  DestType ref (mwDest, destBC, destGhost, NULL);
  DestType test(mwDest, destBC, destGhost, NULL);

  //initialize source field / zero out result fields:
  initialize_field(src);
  ref <<= 0.0;
  test <<= 0.0;

  //get operator:
  typedef typename SpatialOps::OperatorTypeBuilder<OpType,SrcType,DestType>::type Op;
  const Op* const op = opdb.retrieve_operator<Op>();

  //run reference:
  ref_stencil4_apply_to_field(op->coefs().get_coef(0),
                              op->coefs().get_coef(1),
                              op->coefs().get_coef(2),
                              op->coefs().get_coef(3),
                              src,
                              ref);

  //run operator:
  op->apply_to_field(src, test);

  return (field_equal(test, ref, ACCEPTABLE_ERROR));
};

//--------------------------------------------------------------------

template<typename OpType, typename FieldType>
bool test_fd_stencil2(const IntVec npts,
                      const OperatorDatabase & opdb,
                      bool bc[])
{
  //basic definitions:
  const GhostData ghost(1);
  const BoundaryCellInfo bcinfo = BoundaryCellInfo::build<FieldType>(bc[0], bc[1], bc[2]);
  const MemoryWindow mw  = get_window_with_ghost(npts, ghost, bcinfo);
  FieldType src (mw, bcinfo, ghost, NULL);
  FieldType ref (mw, bcinfo, ghost, NULL);
  FieldType test(mw, bcinfo, ghost, NULL);

  //initialize source field / zero out result fields:
  initialize_field(src);
  ref <<= 0.0;
  test <<= 0.0;

  //get operator:
  typedef typename SpatialOps::OperatorTypeBuilder<OpType,FieldType,FieldType>::type Op;
  const Op* const op = opdb.retrieve_operator<Op>();

  //run reference:
  ref_fd_stencil2_apply_to_field<OpType,FieldType>(op->coefs().get_coef(0),
                                                   op->coefs().get_coef(1),
                                                   src,
                                                   ref);

  //run operator:
  op->apply_to_field(src, test);

  return (field_equal(test, ref, ACCEPTABLE_ERROR));
};

//--------------------------------------------------------------------

template<typename OpType, typename SrcType, typename DestType>
bool test_null_stencil(const IntVec npts,
                       const OperatorDatabase & opdb,
                       bool bc[])
{
  //basic definitions:
  const GhostData  srcGhost(1);
  const GhostData destGhost(1);
  const BoundaryCellInfo  srcBC = BoundaryCellInfo::build< SrcType>(bc[0],bc[1],bc[2]);
  const BoundaryCellInfo destBC = BoundaryCellInfo::build<DestType>(bc[0],bc[1],bc[2]);
  const MemoryWindow mwSrc  = get_window_with_ghost(npts,  srcGhost,  srcBC);
  const MemoryWindow mwDest = get_window_with_ghost(npts, destGhost, destBC);
  SrcType  src (mwSrc,   srcBC,  srcGhost, NULL);
  DestType ref (mwDest, destBC, destGhost, NULL);
  DestType test(mwDest, destBC, destGhost, NULL);

  //initialize source field / zero out result fields:
  initialize_field(src);
  ref <<= 0.0;
  test <<= 0.0;

  //get operator:
  typedef typename SpatialOps::OperatorTypeBuilder<OpType,SrcType,DestType>::type Op;
  const Op* const op = opdb.retrieve_operator<Op>();

  //run reference:
  ref_null_stencil_apply_to_field(src, ref);

  //run operator:
  op->apply_to_field(src, test);

  return (field_equal(test, ref, ACCEPTABLE_ERROR));
};

//--------------------------------------------------------------------

template<typename OpType, typename FieldType>
bool test_box_filter_stencil(const IntVec npts,
                             const OperatorDatabase & opdb,
                             bool bc[])
{
  //basic definitions:
  const GhostData ghost(1);
  const BoundaryCellInfo bcinfo = BoundaryCellInfo::build<FieldType>(bc[0], bc[1], bc[2]);
  const MemoryWindow mw = get_window_with_ghost(npts, ghost, bcinfo);
  FieldType src (mw, bcinfo, ghost, NULL);
  FieldType ref (mw, bcinfo, ghost, NULL);
  FieldType test(mw, bcinfo, ghost, NULL);

  //initialize source field / zero out result fields:
  initialize_field(src);
  ref <<= 0.0;
  test <<= 0.0;

  //get operator:
  typedef typename SpatialOps::OperatorTypeBuilder<OpType,FieldType,FieldType>::type Op;
  const Op* const op = opdb.retrieve_operator<Op>();

  //run reference:
  ref_box_filter_stencil_apply_to_field(src, ref);

  //run operator:
  op->apply_to_field(src, test);

  return (field_equal(test, ref, ACCEPTABLE_ERROR));
};

//--------------------------------------------------------------------

template<typename VolField>
inline bool test_basic_stencils(const IntVec npts,
                                const OperatorDatabase & opdb,
                                bool bc[])
{
  typedef typename FaceTypes<VolField>::XFace SurfXField;
  typedef typename FaceTypes<VolField>::YFace SurfYField;
  typedef typename FaceTypes<VolField>::ZFace SurfZField;

  TestHelper status(true);

  if( npts[0] > 1) {
    status( test_stencil2<Interpolant, VolField,   SurfXField>(npts, opdb, bc), "Interpolant VolField -> SurfXField (2)" );
    status( test_stencil2<Gradient,    VolField,   SurfXField>(npts, opdb, bc), "Gradient    VolField -> SurfXField (2)" );
    status( test_stencil2<Divergence,  SurfXField, VolField>  (npts, opdb, bc), "Divergence  SurfXField -> VolField (2)" );
  };

  if( npts[1] > 1) {
    status( test_stencil2<Interpolant, VolField,   SurfYField>(npts, opdb, bc), "Interpolant VolField -> SurfYField (2)" );
    status( test_stencil2<Gradient,    VolField,   SurfYField>(npts, opdb, bc), "Gradient    VolField -> SurfYField (2)" );
    status( test_stencil2<Divergence,  SurfYField, VolField>  (npts, opdb, bc), "Divergence  SurfYField -> VolField (2)" );
  };

  if( npts[2] > 1) {
    status( test_stencil2<Interpolant, VolField,   SurfZField>(npts, opdb, bc), "Interpolant VolField -> SurfZField (2)" );
    status( test_stencil2<Gradient,    VolField,   SurfZField>(npts, opdb, bc), "Gradient    VolField -> SurfZField (2)" );
    status( test_stencil2<Divergence,  SurfZField, VolField>  (npts, opdb, bc), "Divergence  SurfZField -> VolField (2)" );
  };

  return status.ok();
};

//--------------------------------------------------------------------

#define TEST_EXTENTS( SRC, DEST,                                                                                                 \
                      DirT,                                                                                                      \
                      S2Ox,  S2Oy,  S2Oz,                                                                                        \
                      DOx,   DOy,   DOz,                                                                                         \
                      ULx,   ULy,   ULz,                                                                                         \
                      name )                                                                                                     \
    {                                                                                                                            \
      using std::string;                                                                                                         \
      typedef ExtentsAndOffsets<SRC,DEST> Extents;                                                                               \
      status( IsSameType< Extents::Dir,        DirT                          >::result, string(name) + string(" dir"       ) );  \
      status( IsSameType< Extents::Src2Offset, IndexTriplet<S2Ox,S2Oy,S2Oz > >::result, string(name) + string(" s2 offset" ) );  \
      status( IsSameType< Extents::DestOffset, IndexTriplet<DOx, DOy, DOz  > >::result, string(name) + string(" d  offset" ) );  \
      status( IsSameType< Extents::Src1Extent, IndexTriplet<ULx, ULy, ULz  > >::result, string(name) + string(" UB"        ) );  \
    }
//-------------------------------------------------------------------

bool test_compile_time()
{
  using namespace SpatialOps;
  using namespace RefStencil2Detail;

  TestHelper status(false);

  status( IsSameType< ActiveDir< YVolField, SVolField   >::type, YDIR >::result, "YVol->SVol (y)" );
  status( IsSameType< ActiveDir< YVolField, XSurfYField >::type, XDIR >::result, "YVol->ZSY  (x)" );
  status( IsSameType< ActiveDir< YVolField, ZSurfYField >::type, ZDIR >::result, "YVol->ZSY  (z)" );

  status( IsSameType< ActiveDir< ZVolField, SVolField   >::type, ZDIR >::result, "ZVol->SVol (z)" );
  status( IsSameType< ActiveDir< ZVolField, XSurfZField >::type, XDIR >::result, "ZVol->XSZ  (x)" );
  status( IsSameType< ActiveDir< ZVolField, YSurfZField >::type, YDIR >::result, "ZVol->YSZ  (y)" );

  status( IsSameType< ActiveDir< SVolField, XVolField >::type, XDIR >::result, "SVol->XVol (x)" );
  status( IsSameType< ActiveDir< SVolField, YVolField >::type, YDIR >::result, "SVol->YVol (y)" );
  status( IsSameType< ActiveDir< SVolField, ZVolField >::type, ZDIR >::result, "SVol->ZVol (z)" );

  TEST_EXTENTS( SVolField, SSurfXField, XDIR,  1,0,0,  1,0,0,  -1, 0, 0,  "SVol->SSX" )
  TEST_EXTENTS( SVolField, SSurfYField, YDIR,  0,1,0,  0,1,0,   0,-1, 0,  "SVol->SSY" )
  TEST_EXTENTS( SVolField, SSurfZField, ZDIR,  0,0,1,  0,0,1,   0, 0,-1,  "SVol->SSZ" )
  TEST_EXTENTS( SSurfXField, SVolField, XDIR,  1,0,0,  0,0,0,  -1, 0, 0,  "SSX->SVol" )
  TEST_EXTENTS( SSurfYField, SVolField, YDIR,  0,1,0,  0,0,0,   0,-1, 0,  "SSY->SVol" )
  TEST_EXTENTS( SSurfZField, SVolField, ZDIR,  0,0,1,  0,0,0,   0, 0,-1,  "SSZ->SVol" )

  TEST_EXTENTS( XVolField, XSurfXField, XDIR,  1,0,0,  0,0,0,  -1, 0, 0,  "XVol->XSX" )
  TEST_EXTENTS( XVolField, XSurfYField, YDIR,  0,1,0,  0,1,0,   0,-1, 0,  "XVol->XSY" )
  TEST_EXTENTS( XVolField, XSurfZField, ZDIR,  0,0,1,  0,0,1,   0, 0,-1,  "XVol->XSZ" )

  TEST_EXTENTS( XSurfXField, XVolField, XDIR,  1,0,0,  1,0,0,  -1, 0, 0,  "XSX->XVol" )
  TEST_EXTENTS( XSurfYField, XVolField, YDIR,  0,1,0,  0,0,0,   0,-1, 0,  "XSY->XVol" )
  TEST_EXTENTS( XSurfZField, XVolField, ZDIR,  0,0,1,  0,0,0,   0, 0,-1,  "XSZ->XVol" )

  TEST_EXTENTS( YVolField, YSurfXField, XDIR,  1,0,0,  1,0,0,  -1, 0, 0,  "YVol->YSX" )
  TEST_EXTENTS( YVolField, YSurfYField, YDIR,  0,1,0,  0,0,0,   0,-1, 0,  "YVol->YSY" )
  TEST_EXTENTS( YVolField, YSurfZField, ZDIR,  0,0,1,  0,0,1,   0, 0,-1,  "YVol->YSZ" )
  TEST_EXTENTS( YSurfXField, YVolField, XDIR,  1,0,0,  0,0,0,  -1, 0, 0,  "YSX->YVol" )
  TEST_EXTENTS( YSurfYField, YVolField, YDIR,  0,1,0,  0,1,0,   0,-1, 0,  "YSY->YVol" )
  TEST_EXTENTS( YSurfZField, YVolField, ZDIR,  0,0,1,  0,0,0,   0, 0,-1,  "YSZ->YVol" )

  TEST_EXTENTS( ZVolField, ZSurfXField, XDIR,  1,0,0,  1,0,0,  -1, 0, 0,  "ZVol->ZSX" )
  TEST_EXTENTS( ZVolField, ZSurfYField, YDIR,  0,1,0,  0,1,0,   0,-1, 0,  "ZVol->ZSY" )
  TEST_EXTENTS( ZVolField, ZSurfZField, ZDIR,  0,0,1,  0,0,0,   0, 0,-1,  "ZVol->ZSZ" )
  TEST_EXTENTS( ZSurfXField, ZVolField, XDIR,  1,0,0,  0,0,0,  -1, 0, 0,  "ZSX->ZVol" )
  TEST_EXTENTS( ZSurfYField, ZVolField, YDIR,  0,1,0,  0,0,0,   0,-1, 0,  "ZSY->ZVol" )
  TEST_EXTENTS( ZSurfZField, ZVolField, ZDIR,  0,0,1,  0,0,1,   0, 0,-1,  "ZSZ->ZVol" )

  TEST_EXTENTS( XVolField, SVolField,   XDIR,  1,0,0,  0,0,0,  -1, 0, 0,  "XVol->SVol" )
  TEST_EXTENTS( XVolField, YSurfXField, YDIR,  0,1,0,  0,1,0,   0,-1, 0,  "XVol->YSX"  )
  TEST_EXTENTS( XVolField, ZSurfXField, ZDIR,  0,0,1,  0,0,1,   0, 0,-1,  "XVol->ZSX"  )

  return status.ok();
}

//--------------------------------------------------------------------

int main( int iarg, char* carg[] )
{
  int nx, ny, nz;
  bool bc[] = { false, false, false };

  {
    po::options_description desc("Supported Options");
    desc.add_options()
      ( "help", "print help message\n" )
      ( "nx",  po::value<int>(&nx)->default_value(11), "number of points in x-dir for base mesh" )
      ( "ny",  po::value<int>(&ny)->default_value(11), "number of points in y-dir for base mesh" )
      ( "nz",  po::value<int>(&nz)->default_value(11), "number of points in z-dir for base mesh" )
      ( "bcx", "physical boundary on +x side?" )
      ( "bcy", "physical boundary on +y side?" )
      ( "bcz", "physical boundary on +z side?" );

    po::variables_map args;
    po::store( po::parse_command_line(iarg,carg,desc), args );
    po::notify(args);

    if( args.count("bcx") ) bc[0] = true;
    if( args.count("bcy") ) bc[1] = true;
    if( args.count("bcz") ) bc[2] = true;

    if( args.count("help") ){
      cout << desc << endl
           << "Examples:" << endl
           << " test_stencil --nx 5 --ny 10 --nz 3 --bcx" << endl
           << " test_stencil --bcx --bcy --bcz" << endl
           << " test_stencil --nx 50 --bcz" << endl
           << endl;
      return -1;
    }
  }

  TestHelper status( true );
  const IntVec npts(nx,ny,nz);

  {
    const std::string bcx = bc[0] ? "ON" : "OFF";
    const std::string bcy = bc[1] ? "ON" : "OFF";
    const std::string bcz = bc[2] ? "ON" : "OFF";
    cout << "Run information: " << endl
         << "  bcx    : " << bcx << endl
         << "  bcy    : " << bcy << endl
         << "  bcz    : " << bcz << endl
         << "  domain : " << npts << endl
         << endl;
  }

  status( test_compile_time(), "Compile time type introspection tests" );
  cout << endl;

  const double length = 10.0;

  OperatorDatabase opdb;
  build_stencils( npts[0], npts[1], npts[2], length, length, length, opdb );

  try{
      //Stencil2 tests:

      status( test_basic_stencils<SVolField>(npts, opdb, bc), "SVol operators");
      if( npts[0]>1 ) { status( test_basic_stencils<XVolField>(npts, opdb, bc), "XVol operators"); };
      if( npts[1]>1 ) { status( test_basic_stencils<YVolField>(npts, opdb, bc), "YVol operators"); };
      if( npts[2]>1 ) { status( test_basic_stencils<ZVolField>(npts, opdb, bc), "ZVol operators"); };

      if( npts[0]>1 && npts[1]>1 ) status( test_stencil2<Interpolant, XVolField, YSurfXField>(npts, opdb, bc), "Interpolant XVolField -> YSurfXField (2)" );
      if( npts[0]>1 && npts[1]>1 ) status( test_stencil2<Gradient,    XVolField, YSurfXField>(npts, opdb, bc), "Gradient    XVolField -> YSurfXField (2)" );

      if( npts[0]>1 && npts[2]>1 ) status( test_stencil2<Interpolant, XVolField, ZSurfXField>(npts, opdb, bc), "Interpolant XVolField -> ZSurfXField (2)" );
      if( npts[0]>1 && npts[2]>1 ) status( test_stencil2<Gradient,    XVolField, ZSurfXField>(npts, opdb, bc), "Gradient    XVolField -> ZSurfXField (2)" );

      if( npts[1]>1 && npts[0]>1 ) status( test_stencil2<Interpolant, YVolField, XSurfYField>(npts, opdb, bc), "Interpolant YVolField -> XSurfYField (2)" );
      if( npts[1]>1 && npts[0]>1 ) status( test_stencil2<Gradient,    YVolField, XSurfYField>(npts, opdb, bc), "Gradient    YVolField -> XSurfYField (2)" );

      if( npts[1]>1 && npts[2]>1 ) status( test_stencil2<Interpolant, YVolField, ZSurfYField>(npts, opdb, bc), "Interpolant YVolField -> ZSurfYField (2)" );
      if( npts[1]>1 && npts[2]>1 ) status( test_stencil2<Gradient,    YVolField, ZSurfYField>(npts, opdb, bc), "Gradient    YVolField -> ZSurfYField (2)" );

      if( npts[2]>1 && npts[0]>1 ) status( test_stencil2<Interpolant, ZVolField, XSurfZField>(npts, opdb, bc), "Interpolant ZVolField -> XSurfZField (2)" );
      if( npts[2]>1 && npts[0]>1 ) status( test_stencil2<Gradient,    ZVolField, XSurfZField>(npts, opdb, bc), "Gradient    ZVolField -> XSurfZField (2)" );

      if( npts[2]>1 && npts[1]>1 ) status( test_stencil2<Interpolant, ZVolField, YSurfZField>(npts, opdb, bc), "Interpolant ZVolField -> YSurfZField (2)" );
      if( npts[2]>1 && npts[1]>1 ) status( test_stencil2<Gradient,    ZVolField, YSurfZField>(npts, opdb, bc), "Gradient    ZVolField -> YSurfZField (2)" );

      if( npts[0]>1 ) status( test_stencil2<Interpolant, SVolField, XVolField>(npts, opdb, bc), "Interpolant SVolField -> XVolField (2)" );
      if( npts[0]>1 ) status( test_stencil2<Gradient,    SVolField, XVolField>(npts, opdb, bc), "Gradient    SVolField -> XVolField (2)" );

      if( npts[1]>1 ) status( test_stencil2<Interpolant, SVolField, YVolField>(npts, opdb, bc), "Interpolant SVolField -> YVolField (2)" );
      if( npts[1]>1 ) status( test_stencil2<Gradient,    SVolField, YVolField>(npts, opdb, bc), "Gradient    SVolField -> YVolField (2)" );

      if( npts[2]>1 ) status( test_stencil2<Interpolant, SVolField, ZVolField>(npts, opdb, bc), "Interpolant SVolField -> ZVolField (2)" );
      if( npts[2]>1 ) status( test_stencil2<Gradient,    SVolField, ZVolField>(npts, opdb, bc), "Gradient    SVolField -> ZVolField (2)" );

      if( npts[0]>1 ) status( test_stencil2<Interpolant, XVolField, SVolField>(npts, opdb, bc), "Interpolant XVolField -> SVolField (2)" );
      if( npts[0]>1 ) status( test_stencil2<Gradient,    XVolField, SVolField>(npts, opdb, bc), "Gradient    XVolField -> SVolField (2)" );

      if( npts[1]>1 ) status( test_stencil2<Interpolant, YVolField, SVolField>(npts, opdb, bc), "Interpolant YVolField -> SVolField (2)" );
      if( npts[1]>1 ) status( test_stencil2<Gradient,    YVolField, SVolField>(npts, opdb, bc), "Gradient    YVolField -> SVolField (2)" );

      if( npts[2]>1 ) status( test_stencil2<Interpolant, ZVolField, SVolField>(npts, opdb, bc), "Interpolant ZVolField -> SVolField (2)" );
      if( npts[2]>1 ) status( test_stencil2<Gradient,    ZVolField, SVolField>(npts, opdb, bc), "Gradient    ZVolField -> SVolField (2)" );

      if( npts[0]>1 ) status( test_stencil2<Interpolant, XSurfXField, XVolField>(npts, opdb, bc), "Interpolant XSurfXField -> XVolField (2)" );
      if( npts[1]>1 ) status( test_stencil2<Interpolant, XSurfYField, XVolField>(npts, opdb, bc), "Interpolant XSurfYField -> XVolField (2)" );
      if( npts[2]>1 ) status( test_stencil2<Interpolant, XSurfZField, XVolField>(npts, opdb, bc), "Interpolant XSurfZField -> XVolField (2)" );

      if( npts[0]>1 ) status( test_stencil2<Interpolant, YSurfXField, YVolField>(npts, opdb, bc), "Interpolant YSurfXField -> YVolField (2)" );
      if( npts[1]>1 ) status( test_stencil2<Interpolant, YSurfYField, YVolField>(npts, opdb, bc), "Interpolant YSurfYField -> YVolField (2)" );
      if( npts[2]>1 ) status( test_stencil2<Interpolant, YSurfZField, YVolField>(npts, opdb, bc), "Interpolant YSurfZField -> YVolField (2)" );

      if( npts[0]>1 ) status( test_stencil2<Interpolant, ZSurfXField, ZVolField>(npts, opdb, bc), "Interpolant ZSurfXField -> ZVolField (2)" );
      if( npts[1]>1 ) status( test_stencil2<Interpolant, ZSurfYField, ZVolField>(npts, opdb, bc), "Interpolant ZSurfYField -> ZVolField (2)" );
      if( npts[2]>1 ) status( test_stencil2<Interpolant, ZSurfZField, ZVolField>(npts, opdb, bc), "Interpolant ZSurfZField -> ZVolField (2)" );

      //Stencil4 tests:

      if( npts[0]>1 && npts[1]>1 ) status( test_stencil4<Interpolant, SVolField, XSurfYField>(npts, opdb, bc), "Interpolant SVolField -> XSurfYField (4)" );
      if( npts[0]>1 && npts[2]>1 ) status( test_stencil4<Interpolant, SVolField, XSurfZField>(npts, opdb, bc), "Interpolant SVolField -> XSurfZField (4)" );

      if( npts[1]>1 && npts[0]>1 ) status( test_stencil4<Interpolant, SVolField, YSurfXField>(npts, opdb, bc), "Interpolant SVolField -> YSurfXField (4)" );
      if( npts[1]>1 && npts[2]>1 ) status( test_stencil4<Interpolant, SVolField, YSurfZField>(npts, opdb, bc), "Interpolant SVolField -> YSurfZField (4)" );

      if( npts[2]>1 && npts[0]>1 ) status( test_stencil4<Interpolant, SVolField, ZSurfXField>(npts, opdb, bc), "Interpolant SVolField -> ZSurfXField (4)" );
      if( npts[2]>1 && npts[1]>1 ) status( test_stencil4<Interpolant, SVolField, ZSurfYField>(npts, opdb, bc), "Interpolant SVolField -> ZSurfYField (4)" );

      if( npts[0]>1 && npts[1]>1 ) status( test_stencil4<Interpolant, XSurfYField, SVolField>(npts, opdb, bc), "Interpolant XSurfYField -> SVolField (4)" );
      if( npts[0]>1 && npts[2]>1 ) status( test_stencil4<Interpolant, XSurfZField, SVolField>(npts, opdb, bc), "Interpolant XSurfZField -> SVolField (4)" );

      if( npts[1]>1 && npts[0]>1 ) status( test_stencil4<Interpolant, YSurfXField, SVolField>(npts, opdb, bc), "Interpolant YSurfXField -> SVolField (4)" );
      if( npts[1]>1 && npts[2]>1 ) status( test_stencil4<Interpolant, YSurfZField, SVolField>(npts, opdb, bc), "Interpolant YSurfZField -> SVolField (4)" );

      if( npts[2]>1 && npts[0]>1 ) status( test_stencil4<Interpolant, ZSurfXField, SVolField>(npts, opdb, bc), "Interpolant ZSurfXField -> SVolField (4)" );
      if( npts[2]>1 && npts[1]>1 ) status( test_stencil4<Interpolant, ZSurfYField, SVolField>(npts, opdb, bc), "Interpolant ZSurfYField -> SVolField (4)" );

      if( npts[0]>1 && npts[1]>1 ) status( test_stencil4<Interpolant, XVolField, YVolField>(npts, opdb, bc), "Interpolant XVolField -> YVolField (4)" );
      if( npts[0]>1 && npts[2]>1 ) status( test_stencil4<Interpolant, XVolField, ZVolField>(npts, opdb, bc), "Interpolant XVolField -> ZVolField (4)" );

      if( npts[1]>1 && npts[0]>1 ) status( test_stencil4<Interpolant, YVolField, XVolField>(npts, opdb, bc), "Interpolant YVolField -> XVolField (4)" );
      if( npts[1]>1 && npts[2]>1 ) status( test_stencil4<Interpolant, YVolField, ZVolField>(npts, opdb, bc), "Interpolant YVolField -> ZVolField (4)" );

      if( npts[2]>1 && npts[0]>1 ) status( test_stencil4<Interpolant, ZVolField, XVolField>(npts, opdb, bc), "Interpolant ZVolField -> XVolField (4)" );
      if( npts[2]>1 && npts[1]>1 ) status( test_stencil4<Interpolant, ZVolField, YVolField>(npts, opdb, bc), "Interpolant ZVolField -> YVolField (4)" );

      //Finite Difference (FDStencil2) tests:

      if( npts[0]>1 ) status( test_fd_stencil2<InterpolantX, SVolField>(npts, opdb, bc), "InterpolantX SVolField -> SVolField (FD 2)" );
      if( npts[1]>1 ) status( test_fd_stencil2<InterpolantY, SVolField>(npts, opdb, bc), "InterpolantY SVolField -> SVolField (FD 2)" );
      if( npts[2]>1 ) status( test_fd_stencil2<InterpolantZ, SVolField>(npts, opdb, bc), "InterpolantZ SVolField -> SVolField (FD 2)" );

      if( npts[0]>1 ) status( test_fd_stencil2<GradientX, SVolField>(npts, opdb, bc), "GradientX    SVolField -> SVolField (FD 2)" );
      if( npts[1]>1 ) status( test_fd_stencil2<GradientY, SVolField>(npts, opdb, bc), "GradientY    SVolField -> SVolField (FD 2)" );
      if( npts[2]>1 ) status( test_fd_stencil2<GradientZ, SVolField>(npts, opdb, bc), "GradientZ    SVolField -> SVolField (FD 2)" );

      if( npts[0]>1 ) status( test_fd_stencil2<InterpolantX, XVolField>(npts, opdb, bc), "InterpolantX XVolField -> XVolField (FD 2)" );
      if( npts[1]>1 ) status( test_fd_stencil2<InterpolantY, XVolField>(npts, opdb, bc), "InterpolantY XVolField -> XVolField (FD 2)" );
      if( npts[2]>1 ) status( test_fd_stencil2<InterpolantZ, XVolField>(npts, opdb, bc), "InterpolantZ XVolField -> XVolField (FD 2)" );

      if( npts[0]>1 ) status( test_fd_stencil2<GradientX, XVolField>(npts, opdb, bc), "GradientX    XVolField -> XVolField (FD 2)" );
      if( npts[1]>1 ) status( test_fd_stencil2<GradientY, XVolField>(npts, opdb, bc), "GradientY    XVolField -> XVolField (FD 2)" );
      if( npts[2]>1 ) status( test_fd_stencil2<GradientZ, XVolField>(npts, opdb, bc), "GradientZ    XVolField -> XVolField (FD 2)" );

      if( npts[0]>1 ) status( test_fd_stencil2<InterpolantX, YVolField>(npts, opdb, bc), "InterpolantX YVolField -> YVolField (FD 2)" );
      if( npts[1]>1 ) status( test_fd_stencil2<InterpolantY, YVolField>(npts, opdb, bc), "InterpolantY YVolField -> YVolField (FD 2)" );
      if( npts[2]>1 ) status( test_fd_stencil2<InterpolantZ, YVolField>(npts, opdb, bc), "InterpolantZ YVolField -> YVolField (FD 2)" );

      if( npts[0]>1 ) status( test_fd_stencil2<GradientX, YVolField>(npts, opdb, bc), "GradientX    YVolField -> YVolField (FD 2)" );
      if( npts[1]>1 ) status( test_fd_stencil2<GradientY, YVolField>(npts, opdb, bc), "GradientY    YVolField -> YVolField (FD 2)" );
      if( npts[2]>1 ) status( test_fd_stencil2<GradientZ, YVolField>(npts, opdb, bc), "GradientZ    YVolField -> YVolField (FD 2)" );

      if( npts[0]>1 ) status( test_fd_stencil2<InterpolantX, ZVolField>(npts, opdb, bc), "InterpolantX ZVolField -> ZVolField (FD 2)" );
      if( npts[1]>1 ) status( test_fd_stencil2<InterpolantY, ZVolField>(npts, opdb, bc), "InterpolantY ZVolField -> ZVolField (FD 2)" );
      if( npts[2]>1 ) status( test_fd_stencil2<InterpolantZ, ZVolField>(npts, opdb, bc), "InterpolantZ ZVolField -> ZVolField (FD 2)" );

      if( npts[0]>1 ) status( test_fd_stencil2<GradientX, ZVolField>(npts, opdb, bc), "GradientX    ZVolField -> ZVolField (FD 2)" );
      if( npts[1]>1 ) status( test_fd_stencil2<GradientY, ZVolField>(npts, opdb, bc), "GradientY    ZVolField -> ZVolField (FD 2)" );
      if( npts[2]>1 ) status( test_fd_stencil2<GradientZ, ZVolField>(npts, opdb, bc), "GradientZ    ZVolField -> ZVolField (FD 2)" );

      //NullStencil tests:
      status( test_null_stencil<Interpolant, SVolField, SVolField>(npts, opdb, bc), "Interpolant SVolField -> SVolField (Null)" );
      status( test_null_stencil<Interpolant, XVolField, XVolField>(npts, opdb, bc), "Interpolant XVolField -> XVolField (Null)" );
      status( test_null_stencil<Interpolant, YVolField, YVolField>(npts, opdb, bc), "Interpolant YVolField -> YVolField (Null)" );
      status( test_null_stencil<Interpolant, ZVolField, ZVolField>(npts, opdb, bc), "Interpolant ZVolField -> ZVolField (Null)" );

      status( test_null_stencil<Interpolant, XVolField, SSurfXField>(npts, opdb, bc), "Interpolant XVolField -> SSurfXField (Null)" );
      status( test_null_stencil<Interpolant, YVolField, SSurfYField>(npts, opdb, bc), "Interpolant YVolField -> SSurfYField (Null)" );
      status( test_null_stencil<Interpolant, ZVolField, SSurfZField>(npts, opdb, bc), "Interpolant ZVolField -> SSurfZField (Null)" );

      status( test_null_stencil<Interpolant, SVolField, XSurfXField>(npts, opdb, bc), "Interpolant SVolField -> XSurfXField (Null)" );
      status( test_null_stencil<Interpolant, SVolField, YSurfYField>(npts, opdb, bc), "Interpolant SVolField -> YSurfYField (Null)" );
      status( test_null_stencil<Interpolant, SVolField, ZSurfZField>(npts, opdb, bc), "Interpolant SVolField -> ZSurfZField (Null)" );

      //Box filter tests:
      if( npts[0]>1 && npts[1]>1 && npts[2] ) {
          status( test_box_filter_stencil<Filter, SVolField>(npts, opdb, bc), "Filter SVolField -> SVolField (box filter)" );
          status( test_box_filter_stencil<Filter, XVolField>(npts, opdb, bc), "Filter XVolField -> XVolField (box filter)" );
          status( test_box_filter_stencil<Filter, YVolField>(npts, opdb, bc), "Filter YVolField -> YVolField (box filter)" );
          status( test_box_filter_stencil<Filter, ZVolField>(npts, opdb, bc), "Filter ZVolField -> ZVolField (box filter)" );
      }

      if( status.ok() ){
          cout << "ALL TESTS PASSED :)" << endl;
          return 0;
      }
  }
  catch( std::exception& e ){
      cout << e.what() << endl;
  }

  cout << "******************************" << endl
       << " At least one test FAILED! :(" << endl
       << "******************************" << endl;
  return -1;
}

//--------------------------------------------------------------------

