#include <spatialops/Nebo.h>

#include <spatialops/structured/FVStaggeredFieldTypes.h>

#include "test_stencil_helper.h"
#include <test/TestHelper.h>


#include <boost/program_options.hpp>
namespace po = boost::program_options;

using namespace SpatialOps;

#include <stdexcept>
using std::cout;
using std::endl;


int main( int iarg, char* carg[] )
{
  IntVec npts;
  bool bcplus[] = { false, false, false };

  {
    po::options_description desc("Supported Options");
    desc.add_options()
      ( "help", "print help message\n" )
      ( "nx",   po::value<int>(&npts[0])->default_value(5), "number of points in x-dir for base mesh" )
      ( "ny",   po::value<int>(&npts[1])->default_value(5), "number of points in y-dir for base mesh" )
      ( "nz",   po::value<int>(&npts[2])->default_value(5), "number of points in z-dir for base mesh" )
      ( "bcx",  "indicates physical boundary on +x side" )
      ( "bcy",  "indicates physical boundary on +y side" )
      ( "bcz",  "indicates physical boundary on +z side" );

    po::variables_map args;
    po::store( po::parse_command_line(iarg,carg,desc), args );
    po::notify(args);

    if( args.count("bcx") ) bcplus[0] = true;
    if( args.count("bcy") ) bcplus[1] = true;
    if( args.count("bcz") ) bcplus[2] = true;

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

  {
    std::string bcx = bcplus[0] ? "ON" : "OFF";
    std::string bcy = bcplus[1] ? "ON" : "OFF";
    std::string bcz = bcplus[2] ? "ON" : "OFF";
    cout << "Run information: " << endl
         << "  bcx    : " << bcx << endl
         << "  bcy    : " << bcy << endl
         << "  bcz    : " << bcz << endl
         << "  domain : " << npts << endl
         << endl;
  }

  try{
    TestHelper status( true );

    const double length = 10.0;

    if( npts[0]>1 && npts[1]>1 ) status( run_convergence<Interpolant,SVolField,XSurfYField,XDIR,YDIR>( npts, bcplus, length, 2.0 ), "SVol->XSurfY" );
    if( npts[0]>1 && npts[2]>1 ) status( run_convergence<Interpolant,SVolField,XSurfZField,XDIR,ZDIR>( npts, bcplus, length, 2.0 ), "SVol->XSurfZ" );
    if( npts[1]>1 && npts[0]>1 ) status( run_convergence<Interpolant,SVolField,YSurfXField,XDIR,YDIR>( npts, bcplus, length, 2.0 ), "SVol->YSurfX" );
    if( npts[1]>1 && npts[2]>1 ) status( run_convergence<Interpolant,SVolField,YSurfZField,YDIR,ZDIR>( npts, bcplus, length, 2.0 ), "SVol->YSurfZ" );

    if( npts[2]>1 && npts[0]>1 ) status( run_convergence<Interpolant,SVolField,ZSurfXField,XDIR,ZDIR>( npts, bcplus, length, 2.0 ), "SVol->ZSurfX" );
    if( npts[2]>1 && npts[1]>1 ) status( run_convergence<Interpolant,SVolField,ZSurfYField,YDIR,ZDIR>( npts, bcplus, length, 2.0 ), "SVol->ZSurfY" );

    if( npts[0]>1 && npts[1]>1 ) status( run_convergence<Interpolant,XSurfYField,SVolField,XDIR,YDIR>( npts, bcplus, length, 2.0 ), "XSurfY->SVol" );
    if( npts[0]>1 && npts[1]>1 ) status( run_convergence<Interpolant,XSurfZField,SVolField,XDIR,ZDIR>( npts, bcplus, length, 2.0 ), "XSurfZ->SVol" );
    if( npts[0]>1 && npts[1]>1 ) status( run_convergence<Interpolant,YSurfXField,SVolField,YDIR,XDIR>( npts, bcplus, length, 2.0 ), "YSurfX->SVol" );
    if( npts[0]>1 && npts[1]>1 ) status( run_convergence<Interpolant,YSurfZField,SVolField,YDIR,ZDIR>( npts, bcplus, length, 2.0 ), "YSurfZ->SVol" );
    if( npts[0]>1 && npts[1]>1 ) status( run_convergence<Interpolant,ZSurfXField,SVolField,ZDIR,XDIR>( npts, bcplus, length, 2.0 ), "ZSurfX->SVol" );
    if( npts[0]>1 && npts[1]>1 ) status( run_convergence<Interpolant,ZSurfYField,SVolField,ZDIR,YDIR>( npts, bcplus, length, 2.0 ), "ZSurfY->SVol" );

    if( status.ok() ){
      cout << "PASS" << endl;
      return 0;
    }
  }
  catch( std::exception& err ){
    cout << err.what() << std::endl;
  }
  cout << "FAIL" << endl;
  return -1;
}
