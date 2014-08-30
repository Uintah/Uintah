#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include <cmath>

#include <tabprops/TabPropsConfig.h>
#include <tabprops/StateTable.h>
#include <tabprops/Archive.h>

#include "TestHelper.h"

using namespace std;

//--------------------------------------------------------------------
// unit test for 1-D calculation.
bool test_1d()
{
  TestHelper status(false);

  const int n=10;
  const int order = 3;

  // set up the raw data
  vector<double> x, y;
  for( int i=0; i<n; i++ ){
    x.push_back( 4.0*double(i)/double(n-1) );
    y.push_back( 1.5*std::sin(3.1415 * x[i]) );
  }

  // interpolate the data
  Interp1D interp( order, x, y );

  {
    std::ofstream outFile("sp1d.out", std::ios_base::out|std::ios_base::trunc );
    OutputArchive oa(outFile);
    oa << BOOST_SERIALIZATION_NVP( interp );
  }
  {
    std::ifstream inFile("sp1d.out", std::ios_base::in );
    InputArchive ia(inFile);
    Interp1D interpRead;
    ia >> BOOST_SERIALIZATION_NVP( interpRead );
    status( interp == interpRead, "Write/Read 1D" );
  }

#ifndef NDEBUG
  ofstream fout;
  fout.open("true.dat");
#endif
  for( int i=0; i<n; i++ ){
    const double yval = interp.value(x[i]);
    const double err = std::abs( y[i]-yval );
    ostringstream msg;
    msg << "1d interpolate data at x=" << x[i] << " (err=" << err << ")";
    status( err < 1e-10, msg.str() );
#ifndef NDEBUG
    fout << x[i] << "  " << y[i] << "  " << yval << endl;
#endif
  }
#ifndef NDEBUG
  fout.close();
  fout.open("interp.dat");
  for( int i=0; i<3*n; i++ ){
    const double xinterp = 4.0*double(i)/double(3*n-1);
    const double yinterp = interp.value( xinterp );
    fout << xinterp << "  " << yinterp << endl;
  }
  fout.close();
#endif

  // test copy constructor
  Interp1D interp1( interp );
  status(interp1 == interp, "1D copy constructor" );

  // test assignment operator
  Interp1D interpcp = interp1;
  status(interpcp == interp1, "1D assignment operator" );

  return status.ok();
}

//--------------------------------------------------------------------
// unit test for 2D
bool test_2d()
{
  TestHelper status(false);

  const int n1=12;  const int n2=13;  const int nn=n1*n2;
  const int order = 3;

  // set up the raw data
  vector<double> x(n1,0), y(n2,0), phi(nn,0);
  for( int i=0; i<n1; ++i )  x[i] = 30.0*double(i)/double(n1-1);
  for( int i=0; i<n2; ++i )  y[i] = 20.5*double(i)/double(n2-1);

  {
    int kk=0;
    for( int j=0; j<n2; ++j )
      for( int i=0; i<n1; ++i, ++kk )
        phi[kk] = sin(3.1415*x[i]) + cos(3.1415*y[j]);
  }

  // interpolate the data
  Interp2D interp( order, x, y, phi );

  {
    std::ofstream outFile("interp2d.out", std::ios_base::out|std::ios_base::trunc );
    OutputArchive oa(outFile);
    oa << BOOST_SERIALIZATION_NVP( interp );
  }
  {
    std::ifstream inFile("interp2d.out", std::ios_base::in );
    InputArchive ia(inFile);
    Interp2D interp2;
    ia >> BOOST_SERIALIZATION_NVP( interp2 );
    status( interp == interp2, "Interp2D Write/Read" );
  }

  // verify that the interpolated surface indeed interpolates the raw data.
#ifndef NDEBUG
  ofstream fout;
  fout.open("2d.dat");
#endif
  size_t k=0;
  double query[2];
  for( size_t j=0; j<n2; ++j ){
    for( size_t i=0; i<n1; ++i, ++k ){
      query[0]=x[i];
      query[1]=y[j];
      const double phiInterp = interp.value( query );
#ifndef NDEBUG
      fout << x[i] << "  "
	   << y[j] << "  "
	   << phi[k] << "  "
	   << phiInterp << endl;
#endif
      const double err = std::abs( phiInterp - phi[k] );
      if( err > 1e-12 ){
        cout << phi[k] << " : " << phiInterp << " : " << sin(3.1415*x[i])+cos(3.1415*y[j]) << endl;
      }
      status( err<1e-12, "2D interpolation" );
    }
  }
#ifndef NDEBUG
  fout.close();
#endif

  // test copy constructor
  Interp2D interpp2( interp );
  status( interpp2 == interp, "2D copy constructor" );

  // test assignment operator
  Interp2D interp3 = interp;
  status( interp3 == interp, "2D assignment operator" );

  return status.ok();
}

//--------------------------------------------------------------------
// unit test for 3D interpolants
bool test_3d()
{
  using namespace std;

  TestHelper status(false);

  const double pi = 3.1415;
  const int n[]={15,14,13};
  const int order = 3;

  // set up the raw data
  vector<double> x, y, z, phi;
  for( int i=0; i<n[0]; i++ )  x.push_back( 0.05*double(i)/double(n[0]-1) );
  for( int i=0; i<n[1]; i++ )  y.push_back( 0.25*double(i)/double(n[1]-1) );
  for( int i=0; i<n[2]; i++ )  z.push_back( 1.0 *double(i)/double(n[2]-1) );

  for( int k=0; k<n[2]; k++ )
    for( int j=0; j<n[1]; j++ )
      for( int i=0; i<n[0]; i++ )
	phi.push_back( sin(pi*x[i]) +
		       cos(pi*y[j]) +
		       z[k]*z[k] );

  Interp3D interp( order, x, y, z, phi );
  {
    Interp3D i2(order,x,y,z,phi);
    status( interp==i2, "comparison operator" );
  }

  {
    std::ofstream outFile("interp3d.out", std::ios_base::out|std::ios_base::trunc );
    OutputArchive oa(outFile);
    oa << BOOST_SERIALIZATION_NVP( interp );
  }
  {
    std::ifstream inFile("interp3d.out", std::ios_base::in );
    InputArchive ia(inFile);
    Interp3D interpRead;
    ia >> BOOST_SERIALIZATION_NVP( interpRead );
    status( interp == interpRead, "Interp3D Write/Read" );
  }

#ifndef NDEBUG
  ofstream fout;
  fout.open("3d.dat");
#endif
  // verify that the interpolated surface indeed interpolates the raw data.
  int l=0;
  int ninterp=0;
  double query[3];
  bool isok=true;
  for( int k=0; k<n[2]; k++ ){
    query[2]=z[k];
    for( int j=0; j<n[1]; j++ ){
      query[1]=y[j];
      for( int i=0; i<n[0]; i++ ){
	query[0]=x[i];
	const double phiInterp = interp.value( query );
	const double err = std::abs(phiInterp - phi[l]);
#ifndef NDEBUG
	fout << x[i] << " "
	     << y[j] << " "
	     << z[k] << " "
	     << phi[l] << " "
	     << phiInterp << " " << err << endl;
#endif
	isok = isok ? ( err<1.0e-12 ) : isok;
	++l;
      }
    }
  }
  status( isok, "raw data is interpolated" );
#ifndef NDEBUG
  fout.close();
#endif

  // test copy constructor
  Interp3D interp2( interp );
  Interp3D interp3 = interp;
  status( interp2 == interp, "Interp3D copy constructor" );
  status( interp3 == interp, "Interp3D assignment operator" );

  return status.ok();
}
//--------------------------------------------------------------------
// unit test for 4D interpolants
bool test_4d()
{
  using namespace std;

  TestHelper status(false);

  const double pi = 3.1415;
  const int n[]={9,8,6,7};
  const int order = 3;

  // set up the raw data
  vector<double> x1, x2, x3, x4, phi;
  for( int i=0; i<n[0]; ++i )  x1.push_back( 0.05*double(i)/double(n[0]-1) );
  for( int i=0; i<n[1]; ++i )  x2.push_back( 0.25*double(i)/double(n[1]-1) );
  for( int i=0; i<n[2]; ++i )  x3.push_back( 1.0 *double(i)/double(n[2]-1) );
  for( int i=0; i<n[3]; ++i )  x4.push_back( 1.0 *double(i)/double(n[3]-1) );

  for( int l=0; l<n[3]; ++l )
    for( int k=0; k<n[2]; ++k )
      for( int j=0; j<n[1]; ++j )
	for( int i=0; i<n[0]; ++i )
	  phi.push_back( sin(pi*x1[i]) +
			 cos(pi*x2[j]) +
			 x3[k] +
			 sin(2.0*pi*x4[l])*cos(2.0*pi*x4[l]) );

  Interp4D interp( order, x1, x2, x3, x4, phi );
  {
    Interp4D i2(order,x1,x2,x3,x4,phi);
    status( interp==i2, "comparison operator" );
  }

  {
    std::ofstream outFile("interp4d.out", std::ios_base::out|std::ios_base::trunc );
    OutputArchive oa(outFile);
    oa << BOOST_SERIALIZATION_NVP( interp );
  }
  {
    std::ifstream inFile("interp4d.out", std::ios_base::in );
    InputArchive ia(inFile);
    Interp4D bspRead;
    ia >> BOOST_SERIALIZATION_NVP( bspRead );
    status( interp == bspRead, "Interp4D Write/Read" );
  }

#ifndef NDEBUG
  ofstream fout;
  fout.open("4d.dat");
#endif
  // verify that the interpolated surface indeed interpolates the raw data.
  int m=0;
  int ninterp=0;
  double query[4];
  for( int l=0; l<n[3]; ++l ){
    query[3]=x4[l];
    for( int k=0; k<n[2]; ++k ){
      query[2]=x3[k];
      for( int j=0; j<n[1]; ++j ){
	query[1]=x2[j];
	for( int i=0; i<n[0]; ++i ){
	  query[0]=x1[i];
	  const double phiInterp = interp.value( query );
	  ninterp++;
	  const double err = fabs(phiInterp - phi[m]);
#ifndef NDEBUG
	  fout << x1[i] << " "
	       << x2[j] << " "
	       << x3[k] << " "
	       << x4[l] << " "
	       << phi[m] << " "
	       << phiInterp << endl;
#endif
          status( err < 1.0e-12, "Interp4D interpolation" );
	  m++;
	}
      }
    }
  }
#ifndef NDEBUG
  fout.close();
#endif

  // test copy constructor
  Interp4D interp2( interp );
  status( interp2 == interp, "Interp4D copy constructor" );

  // test assignment operator
  Interp4D interp3 = interp;
  status( interp3 == interp, "Interp4D assignment operator" );

  return status.ok();
}

//--------------------------------------------------------------------
// unit test for 5D interpolants
bool test_5d()
{
  using namespace std;

  TestHelper status(false);

  const double pi = 3.1415;
  const int n[]={9,8,7,8,9};
  const int order = 3;

  // set up the raw data
  vector<double> x1, x2, x3, x4, x5, phi;
  for( int i=0; i<n[0]; i++ )  x1.push_back( 0.05*double(i)/double(n[0]-1) );
  for( int i=0; i<n[1]; i++ )  x2.push_back( 0.25*double(i)/double(n[1]-1) );
  for( int i=0; i<n[2]; i++ )  x3.push_back( 1.0 *double(i)/double(n[2]-1) );
  for( int i=0; i<n[3]; i++ )  x4.push_back( 1.0 *double(i)/double(n[3]-1) );
  for( int i=0; i<n[4]; i++ )  x5.push_back( 1.0 *double(i)/double(n[4]-1) );

  for( int m=0; m<n[4]; m++ )
    for( int l=0; l<n[3]; l++ )
      for( int k=0; k<n[2]; k++ )
	for( int j=0; j<n[1]; j++ )
	  for( int i=0; i<n[0]; i++ )
	    phi.push_back( sin(pi*x1[i]) +
			   cos(pi*x2[j]) +
			   x3[k] +
			   sin(2.0*pi*x4[l])*cos(2.0*pi*x4[l]) +
			   sin(3.0*pi*x5[m])*cos(0.5*pi*x5[m]) );

  Interp5D bsp( order, x1, x2, x3, x4, x5, phi );

  {
    std::ofstream outFile("5d.out", std::ios_base::out|std::ios_base::trunc );
    OutputArchive oa(outFile);
    oa << BOOST_SERIALIZATION_NVP( bsp );
  }
  {
    std::ifstream inFile("5d.out", std::ios_base::in );
    InputArchive ia(inFile);
    Interp5D bspRead;
    ia >> BOOST_SERIALIZATION_NVP( bspRead );
    status( bsp == bspRead, "Interp5D Write/Read" );
  }

#ifndef NDEBUG
  ofstream fout;
  fout.open("5d.dat");
#endif
  // verify that we indeed interpolate the raw data.
  int ix=0;
  int ninterp=0;
  double query[5];
  for( int m=0; m<n[4]; m++ ){
    query[4]=x5[m];
    for( int l=0; l<n[3]; l++ ){
      query[3]=x4[l];
      for( int k=0; k<n[2]; k++ ){
	query[2]=x3[k];
	for( int j=0; j<n[1]; j++ ){
	  query[1]=x2[j];
	  for( int i=0; i<n[0]; i++ ){
	    query[0]=x1[i];
	    const double phiInterp = bsp.value( query );
	    ninterp++;
	    const double err = fabs(phiInterp - phi[ix]);
#ifndef NDEBUG
	    fout << x1[i] << " "
		 << x2[j] << " "
		 << x3[k] << " "
		 << x4[l] << " "
		 << x5[m] << " "
		 << phi[ix] << " "
		 << phiInterp << endl;
#endif
            status( err < 1.0e-12, "5D interpolation" );
	    ix++;
	  }
	}
      }
    }
  }
#ifndef NDEBUG
  fout.close();
#endif

  // test copy constructor
  Interp5D bsp2( bsp );
  status( bsp2 == bsp, "5D copy constructor" );

  // test assignment operator
  Interp5D bsp3 = bsp;
  status( bsp3 == bsp, "5D assignment operator" );

  return status.ok();
}

bool test_state_tbl()
{
  TestHelper status(true);

  const int n = 15;
  vector<double> x, phi1, phi2;
  for( int i=0; i<n; ++i ){
    x.push_back( 4.0*double(i)/double(n-1) );
    phi1.push_back( 1.5*std::sin(3.1415 * x[i]) );
    phi2.push_back( 1.5*std::cos(3.1415 * x[i]) );
  }

  Interp1D sp1( 2, x, phi1 );
  Interp1D sp2( 3, x, phi2 );

  vector<string> indepNames;
  indepNames.push_back("x1");
  StateTable tbl( indepNames.size() );
  vector<string> junkNames;
  junkNames.push_back("bad name 1");

  try{
    tbl.add_entry( "phi1", &sp1, indepNames );
    tbl.add_entry( "phi2", &sp2, indepNames );
  }
  catch (const std::runtime_error & e ){
    ostringstream msg;
    msg << "fatal error adding table entries" << endl << e.what() << endl;
    status(false,msg.str());
  }

  try{
    tbl.write_table("written.tbl");
    StateTable t2;
    t2.read_table( "written.tbl" );
    return status.ok();
  }
  catch( std::exception& err ){
    std::cout << err.what() << std::endl
              << "error in table IO" << std::endl;
  }
  return false;
}


//====================================================================
//====================================================================
//====================================================================


int main()
{
  TestHelper status(true);

  try{
    status( test_1d(), "1D interp" );
    status( test_2d(), "2D interp" );
    status( test_3d(), "3D interp" );
    status( test_4d(), "4D interp" );
    status( test_5d(), "5D interp" );
    status( test_state_tbl(), "state table" );
  }
  catch( std::exception& err ){
    cout << err.what() << endl;
    return -1;
  }
  if( status.ok() ){
    cout << "PASS" << endl;
    return 0;
  }
  cout << "FAIL" << endl;
  return -1;
}

