#include <spatialops/SpatialOpsTools.h>

#include <spatialops/structured/FVStaggeredFieldTypes.h>
#include <spatialops/OperatorDatabase.h>

#include <spatialops/structured/stencil/FVStaggeredOperatorTypes.h>
#include <spatialops/structured/stencil/StencilBuilder.h>

#include "test_stencil_helper.h"
#include <test/TestHelper.h>

#include <boost/program_options.hpp>
namespace po = boost::program_options;

using namespace SpatialOps;
using std::cout;
using std::endl;

#include <stdexcept>

//--------------------------------------------------------------------

template< typename Vol >
bool
run_variants( const IntVec npts,
              const bool* bcPlus )
{
  TestHelper status(true);

  typedef typename FaceTypes<Vol>::XFace  XFace;
  typedef typename FaceTypes<Vol>::YFace  YFace;
  typedef typename FaceTypes<Vol>::ZFace  ZFace;

  const double length = 10.0;

  if( npts[0]>1 ){
    status( run_convergence<Interpolant,Vol,XFace,XDIR>( npts, bcPlus, length, 2.0 ), "x-Interpolant" );
    status( run_convergence<Gradient,   Vol,XFace,XDIR>( npts, bcPlus, length, 2.0 ), "x-Gradient"    );
    status( run_convergence<Divergence, XFace,Vol,XDIR>( npts, bcPlus, length, 2.0 ), "x-Divergence"  );
  }

  if( npts[1]>1 ){
    status( run_convergence<Interpolant,Vol,YFace,YDIR>( npts, bcPlus, length, 2.0 ), "y-Interpolant" );
    status( run_convergence<Gradient,   Vol,YFace,YDIR>( npts, bcPlus, length, 2.0 ), "y-Gradient"    );
    status( run_convergence<Divergence, YFace,Vol,YDIR>( npts, bcPlus, length, 2.0 ), "y-Divergence"  );
  }

  if( npts[2]>1 ){
    status( run_convergence<Interpolant,Vol,ZFace,ZDIR>( npts, bcPlus, length, 2.0 ), "z-Interpolant" );
    status( run_convergence<Gradient,   Vol,ZFace,ZDIR>( npts, bcPlus, length, 2.0 ), "z-Gradient"    );
    status( run_convergence<Divergence, ZFace,Vol,ZDIR>( npts, bcPlus, length, 2.0 ), "z-Divergence"  );
  }
  return status.ok();
}

//--------------------------------------------------------------------

int main( int iarg, char* carg[] )
{ int nx, ny, nz;
  bool bcplus[] = { false, false, false };
  {
    po::options_description desc("Supported Options");
    desc.add_options()
      ( "help", "print help message\n" )
      ( "nx",   po::value<int>(&nx)->default_value(11), "number of points in x-dir for base mesh" )
      ( "ny",   po::value<int>(&ny)->default_value(11), "number of points in y-dir for base mesh" )
      ( "nz",   po::value<int>(&nz)->default_value(11), "number of points in z-dir for base mesh" )
      ( "bcx",  "physical boundary on +x side?" )
      ( "bcy",  "physical boundary on +y side?" )
      ( "bcz",  "physical boundary on +z side?" );

    po::variables_map args;
    po::store( po::parse_command_line(iarg,carg,desc), args );
    po::notify(args);

    if( args.count("bcx") ) bcplus[0] = true;
    if( args.count("bcy") ) bcplus[1] = true;
    if( args.count("bcz") ) bcplus[2] = true;

    if( args.count("help") ){
      cout << desc << endl
           << "Examples:" << endl
           << " test_stencil2 --nx 5 --ny 10 --nz 3 --bcx" << endl
           << " test_stencil2 --bcx --bcy --bcz" << endl
           << " test_stencil2 --nx 50 --bcz" << endl
           << endl;
      return -1;
    }
  }

  TestHelper status( true );
  const IntVec npts(nx,ny,nz);
  {
    const std::string bcx = bcplus[0] ? "ON" : "OFF";
    const std::string bcy = bcplus[1] ? "ON" : "OFF";
    const std::string bcz = bcplus[2] ? "ON" : "OFF";
    cout << "Run information: " << endl
         << "  bcx    : " << bcx << endl
         << "  bcy    : " << bcy << endl
         << "  bcz    : " << bcz << endl
         << "  domain : " << npts << endl
         << endl;
  }

  const double length = 10.0;

  try{

    status( run_variants< SVolField >( npts, bcplus ), "SVol operators" );

    if( npts[0] > 1 ) status( run_variants< XVolField >( npts, bcplus ), "XVol operators" );
    if( npts[1] > 1 ) status( run_variants< YVolField >( npts, bcplus ), "YVol operators" );
    if( npts[2] > 1 ) status( run_variants< ZVolField >( npts, bcplus ), "ZVol operators" );

    if( npts[0]>1 & npts[1]>1 ) status( run_convergence< Interpolant, XVolField,   YSurfXField, YDIR >( npts, bcplus, length, 2.0 ), "InterpXVolYSurfX" );
    if( npts[0]>1 & npts[2]>1 ) status( run_convergence< Interpolant, XVolField,   ZSurfXField, ZDIR >( npts, bcplus, length, 2.0 ), "InterpXVolZSurfX" );

    if( npts[0]>1 & npts[1]>1 ) status( run_convergence< Gradient,    XVolField,   YSurfXField, YDIR >( npts, bcplus, length, 2.0 ), "GradXVolYSurfX" );
    if( npts[0]>1 & npts[2]>1 ) status( run_convergence< Gradient,    XVolField,   ZSurfXField, ZDIR >( npts, bcplus, length, 2.0 ), "GradXVolZSurfX" );

    if( npts[1]>1 & npts[0]>1 ) status( run_convergence< Interpolant, YVolField,   XSurfYField, XDIR >( npts, bcplus, length, 2.0 ), "InterpYVolXSurfY" );
    if( npts[1]>1 & npts[2]>1 ) status( run_convergence< Interpolant, YVolField,   ZSurfYField, ZDIR >( npts, bcplus, length, 2.0 ), "InterpYVolZSurfY" );

    if( npts[1]>1 & npts[0]>1 ) status( run_convergence< Gradient,    YVolField,   XSurfYField, XDIR >( npts, bcplus, length, 2.0 ), "GradYVolXSurfY" );
    if( npts[1]>1 & npts[2]>1 ) status( run_convergence< Gradient,    YVolField,   ZSurfYField, ZDIR >( npts, bcplus, length, 2.0 ), "GradYVolZSurfY" );

    if( npts[2]>1 & npts[0]>1 ) status( run_convergence< Interpolant, ZVolField,   XSurfZField, XDIR >( npts, bcplus, length, 2.0 ), "InterpZVolXSurfZ" );
    if( npts[2]>1 & npts[1]>1 ) status( run_convergence< Interpolant, ZVolField,   YSurfZField, YDIR >( npts, bcplus, length, 2.0 ), "InterpZVolYSurfZ" );

    if( npts[2]>1 & npts[0]>1 ) status( run_convergence< Gradient,    ZVolField,   XSurfZField, XDIR >( npts, bcplus, length, 2.0 ), "GradZVolXSurfZ" );
    if( npts[2]>1 & npts[1]>1 ) status( run_convergence< Gradient,    ZVolField,   YSurfZField, YDIR >( npts, bcplus, length, 2.0 ), "GradZVolYSurfZ" );

    if( npts[0]>1 )             status( run_convergence< Interpolant, SVolField,   XVolField,   XDIR >( npts, bcplus, length, 2.0 ), "InterpSVolXVol" );
    if( npts[1]>1 )             status( run_convergence< Interpolant, SVolField,   YVolField,   YDIR >( npts, bcplus, length, 2.0 ), "InterpSVolYVol" );
    if( npts[2]>1 )             status( run_convergence< Interpolant, SVolField,   ZVolField,   ZDIR >( npts, bcplus, length, 2.0 ), "InterpSVolZVol" );

    if( npts[0]>1 )             status( run_convergence< Interpolant, XVolField,   SVolField,   XDIR >( npts, bcplus, length, 2.0 ), "InterpXVolSVol" );
    if( npts[1]>1 )             status( run_convergence< Interpolant, YVolField,   SVolField,   YDIR >( npts, bcplus, length, 2.0 ), "InterpYVolSVol" );
    if( npts[2]>1 )             status( run_convergence< Interpolant, ZVolField,   SVolField,   ZDIR >( npts, bcplus, length, 2.0 ), "InterpZVolSVol" );

    if( npts[0]>1 )             status( run_convergence< Interpolant, XSurfXField, XVolField,   XDIR >( npts, bcplus, length, 2.0 ), "InterpXSXXVol" );
    if( npts[1]>1 )             status( run_convergence< Interpolant, XSurfYField, XVolField,   YDIR >( npts, bcplus, length, 2.0 ), "InterpXSYXVol" );
    if( npts[2]>1 )             status( run_convergence< Interpolant, XSurfZField, XVolField,   ZDIR >( npts, bcplus, length, 2.0 ), "InterpXSZXVol" );

    if( npts[0]>1 )             status( run_convergence< Interpolant, YSurfXField, YVolField,   XDIR >( npts, bcplus, length, 2.0 ), "InterpYSXYVol" );
    if( npts[1]>1 )             status( run_convergence< Interpolant, YSurfYField, YVolField,   YDIR >( npts, bcplus, length, 2.0 ), "InterpYSYYVol" );
    if( npts[2]>1 )             status( run_convergence< Interpolant, YSurfZField, YVolField,   ZDIR >( npts, bcplus, length, 2.0 ), "InterpYSZYVol" );

    if( npts[0]>1 )             status( run_convergence< Interpolant, ZSurfXField, ZVolField,   XDIR >( npts, bcplus, length, 2.0 ), "InterpZSXZVol" );
    if( npts[1]>1 )             status( run_convergence< Interpolant, ZSurfYField, ZVolField,   YDIR >( npts, bcplus, length, 2.0 ), "InterpZSYZVol" );
    if( npts[2]>1 )             status( run_convergence< Interpolant, ZSurfZField, ZVolField,   ZDIR >( npts, bcplus, length, 2.0 ), "InterpZSZZVol" );

    if( npts[0]>1 )             status( run_convergence< InterpolantX, SVolField,  SVolField,   XDIR >( npts, bcplus, length, 2.0 ), "Interp SVol->SVol (X)" );
    if( npts[1]>1 )             status( run_convergence< InterpolantY, SVolField,  SVolField,   YDIR >( npts, bcplus, length, 2.0 ), "Interp SVol->SVol (Y)" );
    if( npts[2]>1 )             status( run_convergence< InterpolantZ, SVolField,  SVolField,   ZDIR >( npts, bcplus, length, 2.0 ), "Interp SVol->SVol (Z)" );
    if( npts[0]>1 )             status( run_convergence< InterpolantX, XVolField,  XVolField,   XDIR >( npts, bcplus, length, 2.0 ), "Interp XVol->XVol (X)" );
    if( npts[1]>1 )             status( run_convergence< InterpolantY, XVolField,  XVolField,   YDIR >( npts, bcplus, length, 2.0 ), "Interp XVol->XVol (Y)" );
    if( npts[2]>1 )             status( run_convergence< InterpolantZ, XVolField,  XVolField,   ZDIR >( npts, bcplus, length, 2.0 ), "Interp XVol->XVol (Z)" );
    if( npts[0]>1 )             status( run_convergence< InterpolantX, YVolField,  YVolField,   XDIR >( npts, bcplus, length, 2.0 ), "Interp YVol->YVol (X)" );
    if( npts[1]>1 )             status( run_convergence< InterpolantY, YVolField,  YVolField,   YDIR >( npts, bcplus, length, 2.0 ), "Interp YVol->YVol (Y)" );
    if( npts[2]>1 )             status( run_convergence< InterpolantZ, YVolField,  YVolField,   ZDIR >( npts, bcplus, length, 2.0 ), "Interp YVol->YVol (Z)" );
    if( npts[0]>1 )             status( run_convergence< InterpolantX, ZVolField,  ZVolField,   XDIR >( npts, bcplus, length, 2.0 ), "Interp ZVol->ZVol (X)" );
    if( npts[1]>1 )             status( run_convergence< InterpolantY, ZVolField,  ZVolField,   YDIR >( npts, bcplus, length, 2.0 ), "Interp ZVol->ZVol (Y)" );
    if( npts[2]>1 )             status( run_convergence< InterpolantZ, ZVolField,  ZVolField,   ZDIR >( npts, bcplus, length, 2.0 ), "Interp ZVol->ZVol (Z)" );

    if( npts[0]>1 )             status( run_convergence< GradientX,    SVolField,  SVolField,   XDIR >( npts, bcplus, length, 2.0 ), "Grad SVol->SVol (X)" );
    if( npts[1]>1 )             status( run_convergence< GradientY,    SVolField,  SVolField,   YDIR >( npts, bcplus, length, 2.0 ), "Grad SVol->SVol (Y)" );
    if( npts[2]>1 )             status( run_convergence< GradientZ,    SVolField,  SVolField,   ZDIR >( npts, bcplus, length, 2.0 ), "Grad SVol->SVol (Z)" );
    if( npts[0]>1 )             status( run_convergence< GradientX,    XVolField,  XVolField,   XDIR >( npts, bcplus, length, 2.0 ), "Grad XVol->XVol (X)" );
    if( npts[1]>1 )             status( run_convergence< GradientY,    XVolField,  XVolField,   YDIR >( npts, bcplus, length, 2.0 ), "Grad XVol->XVol (Y)" );
    if( npts[2]>1 )             status( run_convergence< GradientZ,    XVolField,  XVolField,   ZDIR >( npts, bcplus, length, 2.0 ), "Grad XVol->XVol (Z)" );
    if( npts[0]>1 )             status( run_convergence< GradientX,    YVolField,  YVolField,   XDIR >( npts, bcplus, length, 2.0 ), "Grad YVol->YVol (X)" );
    if( npts[1]>1 )             status( run_convergence< GradientY,    YVolField,  YVolField,   YDIR >( npts, bcplus, length, 2.0 ), "Grad YVol->YVol (Y)" );
    if( npts[2]>1 )             status( run_convergence< GradientZ,    YVolField,  YVolField,   ZDIR >( npts, bcplus, length, 2.0 ), "Grad YVol->YVol (Z)" );
    if( npts[0]>1 )             status( run_convergence< GradientX,    ZVolField,  ZVolField,   XDIR >( npts, bcplus, length, 2.0 ), "Grad ZVol->ZVol (X)" );
    if( npts[1]>1 )             status( run_convergence< GradientY,    ZVolField,  ZVolField,   YDIR >( npts, bcplus, length, 2.0 ), "Grad ZVol->ZVol (Y)" );
    if( npts[2]>1 )             status( run_convergence< GradientZ,    ZVolField,  ZVolField,   ZDIR >( npts, bcplus, length, 2.0 ), "Grad ZVol->ZVol (Z)" );

    if( status.ok() ){
      cout << "ALL TESTS PASSED :)" << endl;
      return 0;
    }
  }
  catch( std::runtime_error& e ){
    cout << e.what() << endl;
  }

  cout << "******************************" << endl
       << " At least one test FAILED! :(" << endl
       << "******************************" << endl;
  return -1;
}

//--------------------------------------------------------------------
