/*

The MIT License

Copyright (c) 1997-2010 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#include <CCA/Components/Arches/Mixing/TabProps/BSpline.h>

#include <CCA/Components/Arches/Mixing/TabProps/LU.h>

#include <Core/Exceptions/InternalError.h>
#include <Core/Thread/Thread.h>

#include <assert.h>
#include <cmath>
#include <algorithm>

#include <string>
#include <sstream>
#include <iostream> 

using namespace Uintah;
using namespace SCIRun;
using namespace std;

//--------------------------------------------------------------------
void
basis_funs( const int i,              // index for location of interest
            const int p,              // order of approximation
            const double u,           // location of interest
            const vector<double> & U, // knot vector
            vector<double> & N )      // shape function array
{
  //
  // see "The NURBS Book" second edition, ALG A2.2 (p. 70)
  //
  assert( p <= 10 );
  double left[10], right[10];
  for( int j=0; j<10; j++ ){ left[j]=0.0; right[j]=0.0; }
  assert( p<=10 );

  N[0] = 1.0;
  for( int j=1; j<=p; j++ ){
    left[j] = u-U[i+1-j];
    right[j] = U[i+j]-u;
    double saved = 0.0;
    
    for( int r=0; r<j; r++ ){
      double tmp = N[r]/(right[r+1]+left[j-r]);
      N[r] = saved + right[r+1]*tmp;
      saved = left[j-r]*tmp;
    }
    
    N[j] = saved;
  }
}
//--------------------------------------------------------------------
int find_indx( const int n,               // number of control points
	       const int p,               // order of spline
	       const double u,            // location of interest
	       const vector<double> & U ) // knot vector
{
  if( u < U[0] || u > U[n+1] ) {
    ostringstream errmsg;
    errmsg << "Request (" << u << ") exceeds bounds.\n"
	   << "Valid range: [" << U[0] << "," << U[n+1] << "]\n";
    throw InternalError( errmsg.str(), __FILE__, __LINE__ );
  }
  
  //
  // see "The NURBS Book" second edition, ALG A2.1 (p. 68)
  //
  if( u == U[n+1]){
    return n-1;
  }
  
  int low = p;
  int high = n+1;
  int mid = (low+high)/2;
  
  while( u < U[mid]  ||  u >= U[mid+1] ){
    if( u < U[mid] ){
      high = mid;
    }else{
      low = mid;
    }
    mid = (low+high)/2;
  }
  return mid;
}
//--------------------------------------------------------------------
double get_uk( const double indepVar,
	       const double maxIndepVarVal,
	       const double minIndepVarVal,
	       const bool clipExtrema )
{
  const double value = clipExtrema ? min( std::max(indepVar,minIndepVarVal), maxIndepVarVal )
                                   : indepVar;
    
  return (value-minIndepVarVal) / (maxIndepVarVal-minIndepVarVal);
}
 
//--------------------------------------------------------------------

void
set_uk( const vector<double> & indepVars,
        vector<double> & uk,
        double & maxIndepVarVal,
        double & minIndepVarVal )
{
  //
  // Set uk by scaling it to lie between 0 and 1. Return the scaling
  // values so we can map the independent variable into parameter
  // space for the interpolation.
  //
  uk.clear();
  minIndepVarVal = *std::min_element( indepVars.begin(), indepVars.end() );
  maxIndepVarVal = *std::max_element( indepVars.begin(), indepVars.end() );
  for( int i=0; i<(int)indepVars.size(); i++ ){
    uk.push_back( get_uk( indepVars[i], maxIndepVarVal, minIndepVarVal, false ) );
  }
}

//--------------------------------------------------------------------

int
set_knot_vector( const vector<double> & uk,
                 const int order,
                 vector<double> & knots )
{
  // jcs fix this.  we only need structured meshes with increasing uk.
  //     We should ensure that it is increasing, however.
  vector<double> uksrt = uk;
  std::sort( uksrt.begin(), uksrt.end() );
  vector<double>::iterator ik = std::unique( uksrt.begin(), uksrt.end() );
  uksrt.erase(ik,uksrt.end());

  const int nn = uksrt.size();

  // set up the knot vector by averaging the (sorted, reduced) uk
  knots.clear();
  for( int i=0; i<=order; i++ ){
    knots.push_back(0.0);
  }

  for( int j=1; j<nn-order; j++ ){
    double tmp = 0.0;
    
    for( int i=j; i<j+order; i++ ){
      tmp += uksrt[i];
    }
    tmp /= double(order);

    // avoid duplicates.  Only push back if this is a "new" entry.
    if( std::find( knots.begin(), knots.end(), tmp ) == knots.end() ){
      knots.push_back(tmp);
    }
  }
  for( int i=0; i<=order; i++ ){
    knots.push_back(1.0);
  }

  // ensure that the knot vector is non-decreasing
  for( int i=1; i<(int)knots.size(); i++ ){
    assert( knots[i] >= knots[i-1] );
  }

  return nn;
}

//--------------------------------------------------------------------
// unit test for computation of basis functions
// test is hard-coded for 2nd order and a particular knot sequence
#define numof(t) (sizeof(t)/sizeof(t[0]))

bool
test_basis_fun()
{
  using namespace std;
  cout << "  Testing basis function evaluation ... ";

  bool isOkay = true;

  // the knot vector
  double UU[] = { 0., 0., 0., 1., 2., 3., 4., 4., 5., 5., 5. };
  const int n = numof(UU);
  vector<double> U;  for( int i=0; i<n; i++ ) U.push_back(UU[i]);

  const int order = 2;   const int npts = n-order-1;
  double u = 5./2.;   int ix = find_indx( npts, order, u, U );
  if( ix != 4 ){
    isOkay = false;
  }
  const double tol = 1.0e-12;
  vector<double> Ni0(1,0.0), Ni1(2,0.0), Ni2(3,0.0);
  basis_funs( ix, 0, u, U, Ni0 );  // linear
  basis_funs( ix, 1, u, U, Ni1 );  // quadratic
  basis_funs( ix, 2, u, U, Ni2 );  // cubic

  if( std::fabs(Ni0[0] - 1.0) > tol ){
    isOkay = false;
  }
  
  if( std::fabs(Ni1[0] - 0.5) > tol ||
      std::fabs(Ni1[1] - 0.5) > tol ){
    isOkay = false;
  }
  
  if( std::fabs(Ni2[0] - 1./8.) > tol ||
      std::fabs(Ni2[1] - 6./8.) > tol ||
      std::fabs(Ni2[2] - 1./8.) > tol ){
    isOkay = false;
  }
  u=4.6; ix=find_indx(npts,order,u,U);
  basis_funs( ix, order, u, U, Ni2 );
  if( std::fabs(Ni2[0] - 0.16) > tol ||
      std::fabs(Ni2[1] - 0.48) > tol ||
      std::fabs(Ni2[2] - 0.36) > tol ){
    isOkay = false;
  }
  
  u=0.0; 
  ix=find_indx(npts,order,u,U);  
  basis_funs( ix, order, u, U, Ni2 );
  u=5.0; 
  ix=find_indx(npts,order,u,U);  
  basis_funs( ix, order, u, U, Ni2 );

  isOkay ? cout << "PASS." << endl : cout << "FAIL!" << endl;

  return isOkay;
} // end test_basis_fun()

//====================================================================

BSpline::BSpline( const int order,
		  const int dimension,
		  const bool doClip ) :
  order_( order ),
  dim_( dimension ),
  enableValueClipping_( doClip )
{
#if !defined( HAVE_HDF5 ) 
  cout << "\n";
  cout << "ERROR: The TabProps table reader needs HDF5, since TabProps tables are in HDF5 format,\n" 
       << " but you didn't specify an installation of HDF5 when you ran configure.\n";
  cout << "\n";
  Thread::exitAll( -1 );
#endif
}

BSpline::~BSpline()
{
}

//====================================================================

BSpline1D::BSpline1D( const int order,
		      const vector<double> & indepVars,
		      const vector<double> & depVars,
		      const bool allowClipping ) :
  BSpline( order, 1, allowClipping ),
  npts_( indepVars.size() )
{
  basisFun_.assign(order_+1,0.0);
  compute_control_pts( indepVars, depVars );
}

//--------------------------------------------------------------------

BSpline1D::BSpline1D( const bool allowClipping ) :
  BSpline( 0, 1, allowClipping ),
  npts_( 0 )
{
}

//--------------------------------------------------------------------

BSpline1D::BSpline1D( const BSpline1D& src ) :
  BSpline( src.order_, 1, src.enableValueClipping_ )
{
  npts_ = src.npts_;
  maxIndepVarVal_ = src.maxIndepVarVal_;
  minIndepVarVal_ = src.minIndepVarVal_;
  knots_ = src.knots_;
  controlPts_ = src.controlPts_;
  basisFun_.assign( order_+1, 0.0 );
}

BSpline1D::~BSpline1D()
{
}

//--------------------------------------------------------------------

void
BSpline1D::dump()
{
  using namespace std;

  vector<double>::const_iterator ii;

  cout << "-------------------------------------------------------" << endl
       << " order: " << order_ << ",  npts: " << npts_ << ", nknots: " << (int)knots_.size()
       << ", max/min: " << maxIndepVarVal_ << "/" << minIndepVarVal_ << endl
       << " knots: ";
  for( ii=knots_.begin(); ii!= knots_.end(); ii++ ){
    cout << *ii << ", ";
  }
  
  cout << endl;
  cout << " ctrlPts: ";
  for( ii=controlPts_.begin(); ii!=controlPts_.end(); ii++ ){
    cout << *ii << ", ";
  }
  cout << endl;
}
//--------------------------------------------------------------------
void
BSpline1D::compute_control_pts( const vector<double> & indepVars,
				const vector<double> & depVars )
{
  vector<double> uk;
  set_uk( indepVars, uk, maxIndepVarVal_, minIndepVarVal_ );
  set_knot_vector( uk, order_, knots_ );

  // set up the linear system
  LU A( npts_, order_+1 );

  vector<double> b( npts_, 0.0 );

  for( int i=0; i<npts_; i++ ){
    const int ix = find_indx( npts_, order_, uk[i], knots_ );
    basis_funs( ix, order_, uk[i], knots_, basisFun_ );
    const int shift = ix-order_;
    for( int j=0; j<(int)basisFun_.size(); j++ ){
      A(i,shift+j) = basisFun_[j];
    }
  }

  // perform the LU decomposition
  A.decompose();

  // back-substitute to get control points for each coordinate direction.
  // use the "b" vector to shuffle information.  The back-substitution
  // over-writes the "b" vector with the solution vector.
  for( int i=0; i<npts_; i++ )  b[i] = depVars[i];
  A.back_subs( &b[0] );

  // allocate storage and zero the control point vector.
  controlPts_.assign(npts_,0.0);
  for( int i=0; i<npts_; i++ ){
    controlPts_[i] = b[i];
  }
}
//--------------------------------------------------------------------
double
BSpline1D::value( const double * indepVar ) const
{
  double result = 0.0;

  // obtain the parametric value for the independent variable
  const double uk = get_uk( *indepVar, maxIndepVarVal_, minIndepVarVal_, enableValueClipping_ );

  assert( uk <= 1.0 );
  assert( uk >= 0.0 );

  // get the index for the starting knot corresponding to this value
  const int ix = find_indx( npts_, order_, uk, knots_ );

  // compute the basis functions
  basis_funs( ix, order_, uk, knots_, basisFun_ );

  // compute the dependent variable
  const int shift = ix-order_;
  vector<double>::const_iterator ibf = basisFun_.begin();
  vector<double>::const_iterator icp = controlPts_.begin() + shift;
  for( ; ibf!=basisFun_.end(); ibf++, icp++ ){
    result += (*ibf)*(*icp);
  }

  return result;
}
//--------------------------------------------------------------------

void
BSpline1D::write_hdf5( const hid_t & group ) const
{
#if defined( HAVE_HDF5 ) 
  const hid_t h5s = H5Screate( H5S_SCALAR );  // use this later too.

  //------------------------------
  // write order to the file
  //------------------------------
  const hid_t aOrder = H5Acreate( group, "order", H5T_NATIVE_INT, h5s, H5P_DEFAULT );
  H5Awrite( aOrder, H5T_NATIVE_INT, &order_ );
  H5Aclose( aOrder );
  

  //------------------------------
  // write min/max to the file
  //------------------------------
  const hid_t aMax = H5Acreate( group, "MaxIndepVarValue", H5T_NATIVE_DOUBLE, h5s, H5P_DEFAULT );
  H5Awrite( aMax, H5T_NATIVE_DOUBLE, &maxIndepVarVal_ );
  H5Aclose( aMax );

  const hid_t aMin = H5Acreate( group, "MinIndepVarValue", H5T_NATIVE_DOUBLE, h5s, H5P_DEFAULT );
  H5Awrite( aMin, H5T_NATIVE_DOUBLE, &minIndepVarVal_ );
  H5Aclose( aMin );

  // we are done with the dataspace
  H5Sclose( h5s );
  
  //------------------------------
  // write knots to the file
  //------------------------------
  // create a dataspace for the knots - 1D vector
  const hsize_t nknot = (int)knots_.size();
  const hid_t knotDS = H5Screate_simple( 1, &nknot, NULL );

  // create and write the dataset
  const hid_t knotData = H5Dcreate( group, "knots", H5T_NATIVE_DOUBLE, knotDS, H5P_DEFAULT );
  H5Dwrite( knotData, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &knots_[0] );
  H5Dclose( knotData );
  H5Sclose( knotDS );
  
  //------------------------------
  // write control points to the file
  //------------------------------

  // create a dataspace for the knots - 1D vector
  const hsize_t ncp = (int)controlPts_.size();
  const hid_t contptDS = H5Screate_simple( 1, &ncp, NULL );

  // create a dataset and write it
  const hid_t cpData = H5Dcreate( group, "ControlPoints", H5T_NATIVE_DOUBLE, contptDS, H5P_DEFAULT );
  H5Dwrite( cpData, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &controlPts_[0] );
  H5Dclose( cpData );
  H5Sclose( contptDS );
#endif
}

//--------------------------------------------------------------------
void
BSpline1D::read_hdf5( const hid_t & group )
{
#if defined( HAVE_HDF5 )
  // get the order
  hid_t h5a = H5Aopen_name( group, "order" );
  H5Aread( h5a, H5T_NATIVE_INT, &order_ );
  H5Aclose( h5a );


  // get min/max
  h5a = H5Aopen_name( group, "MinIndepVarValue" );
  H5Aread( h5a, H5T_NATIVE_DOUBLE, &minIndepVarVal_ );
  H5Aclose( h5a );

  h5a = H5Aopen_name( group, "MaxIndepVarValue" );
  H5Aread( h5a, H5T_NATIVE_DOUBLE, &maxIndepVarVal_ );
  H5Aclose( h5a );

  
  // get knots
  hid_t knotData = H5Dopen( group, "knots" );
  const int nKnots = H5Dget_storage_size(knotData)/sizeof(double);
  knots_.assign( nKnots, 0.0 );
  H5Dread( knotData, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &knots_[0] );
  H5Dclose( knotData );

  
  // get control points
  hid_t cpData = H5Dopen( group, "ControlPoints" );
  const int ncp = H5Dget_storage_size(cpData)/sizeof(double);
  controlPts_.assign( ncp, 0.0 );
  H5Dread( cpData, H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT, &controlPts_[0] );
  H5Dclose( cpData );

  npts_ = controlPts_.size();
#endif
}

//====================================================================

//--------------------------------------------------------------------
BSpline2D::BSpline2D( const int order,
		      const std::vector<double> & indepVars1,
		      const std::vector<double> & indepVars2,
		      const std::vector<double> & depVars,
		      const bool allowClipping ) :
  BSpline( order, 2, allowClipping )
{
  const int nn = depVars.size();
  const int n1n2 = indepVars1.size()*indepVars2.size();

  if( nn != n1n2 ){
    std::ostringstream errmsg;
    errmsg << "ERROR in BSpline2d::BSpline2d()!  Inconsistent dimensionality between" << std::endl
	   << "      independent (" << n1n2 << ") and dependent (" << nn << ") variables"
	   << std::endl;
    throw InternalError( errmsg.str(), __FILE__, __LINE__ );
  }
  
  compute_control_pts( indepVars1, indepVars2, depVars );
}
//--------------------------------------------------------------------
BSpline2D::BSpline2D( const bool allowClipping )
  : BSpline( 0, 2, allowClipping )
{
}
//--------------------------------------------------------------------
BSpline2D::BSpline2D( const BSpline2D& src )
  : BSpline( src.order_, 2, src.enableValueClipping_ )
{
  // deep copy
  vector<const BSpline1D*>::const_iterator isp;
  for( isp=src.dim2Splines_.begin(); isp!=src.dim2Splines_.end(); isp++ ){
    dim2Splines_.push_back( new BSpline1D( **isp ) );
  }
  sp1_ = new BSpline1D( *(src.sp1_) );
}
//--------------------------------------------------------------------
BSpline2D::~BSpline2D()
{
  vector<const BSpline1D*>::iterator ii;
  for( ii=dim2Splines_.begin(); ii!=dim2Splines_.end(); ii++ ){
    delete *ii;
  }

  delete sp1_;
}
//--------------------------------------------------------------------
void
BSpline2D::compute_control_pts( const std::vector<double> & indepVars1,
				const std::vector<double> & indepVars2,
				const std::vector<double> & depVars )
{
  const int n = indepVars1.size();
  const int m = indepVars2.size();

  vector<double> dpvrs(n,0.0);
  vector< vector<double> > RR;

  // spline the first dimension at each entry in the
  // second dimension to obtain the "R" vector.
  // "i" is index for the first dim; "j" is index for second dim.
  bool first = true;
  for( int j=0; j<m; j++ ){

    // obtain the dependent variable vector for this curve
    const int shift = j*n;
    for(int i=0; i<n; i++){
      dpvrs[i]=depVars[i+shift];
    }
    // spline it
    BSpline1D sp1d( order_, indepVars1, dpvrs );

    // save off the control points
    RR.push_back( sp1d.get_control_pts() );

    if( first ){
      // create the 1D-spline to be used for the first dimension in
      // lookups later.  Note that we can do this because the knot
      // sequence is constant across all lines in a given direction.
      sp1_ = new BSpline1D( sp1d );
      first = false;
    }
  }

  // spline the second dimension at each entry in the
  // first dimension to obtain the "P" vector
  dpvrs.assign(m,0.0);
  for( int i=0; i<n; i++ ){

    // load the dependent variables vector
    for(int j=0; j<m; j++){
      dpvrs[j] = RR[j][i];
    }
    // spline this one and save it
    BSpline1D * sp1d = new BSpline1D( order_, indepVars2, dpvrs );
    dim2Splines_.push_back( sp1d );
  }
}
//--------------------------------------------------------------------
double
BSpline2D::value( const double* indepVar ) const
{
  //   Q   = sum_j N_j(v) R_j
  //   R_j = sum_i N_i(u) P_{i,j}

  vector<double> & R = sp1_->get_control_pts();
  assert( R.size() == dim2Splines_.size() );

  // obtain the "R" vector, and assign it to the first spline dimension
  // optimized to only do computations of the locations in the "R" vector
  // that will actually be used by the subsequent 1-D interpolation.
  const int p = sp1_->get_order();
  const int ix = find_indx( sp1_->get_npts(),
			    sp1_->get_order(),
			    get_uk(indepVar[0],sp1_->get_maxval(), sp1_->get_minval(), enableValueClipping_),
			    sp1_->get_knot_vector() );
  const int shift = ix-p;
  vector<const BSpline1D*>::const_iterator jj = dim2Splines_.begin()+shift;
  vector<double>::iterator ir = R.begin()+shift;
  vector<double>::const_iterator irend = ir+p+1;
  for( ; ir!=irend; ir++, jj++ ){
    *ir = (*jj)->value( &indepVar[1] );
  }
  return sp1_->value( &indepVar[0] );


  /*
  // here is the brute-force way (compute all control points in the 1-D interpolant)
  vector<BSpline1D*>::const_iterator jj = dim2Splines_.begin();
  vector<double>::iterator ir = R.begin();
  for( ; ir!=R.end(); ir++, jj++ ){
    *ir = (*jj)->value( indepVar[1] );
  }

  return sp1_->value( indepVar[0] );
  */
}
//--------------------------------------------------------------------
void
BSpline2D::write_hdf5( const hid_t & group ) const
{
#if defined( HAVE_HDF5 )
  const hid_t h5s = H5Screate( H5S_SCALAR );

  // write out the information for how many 1-D splines we have
  hid_t h5a = H5Acreate( group, "Number1Dsplines", H5T_NATIVE_INT, h5s, H5P_DEFAULT );
  const int nsp = (int)dim2Splines_.size();
  H5Awrite( h5a, H5T_NATIVE_INT, &nsp );
  H5Aclose( h5a );

  H5Sclose( h5s );
  
  // write out each of the 1-D splines in the array...
  int icount = 0;
  vector<const BSpline1D*>::const_iterator ibs;
  for( ibs=dim2Splines_.begin(); ibs!=dim2Splines_.end(); ibs++, icount++ ){
    // create a group for this spline
    std::ostringstream gname;
    gname << "sp1d_" << icount;
    const hid_t childGroup = H5Gcreate( group, gname.str().c_str(), 0 );
    (*ibs)->write_hdf5( childGroup );
    H5Gclose( childGroup );
  }

  // dump out the spline for the other dimension
  sp1_->write_hdf5( group );
#endif
}

//--------------------------------------------------------------------

void
BSpline2D::read_hdf5( const hid_t & group )
{
#if defined( HAVE_HDF5 )
  int nsp=0;
  hid_t h5a = H5Aopen_name( group, "Number1Dsplines" );
  H5Aread( h5a, H5T_NATIVE_INT, &nsp );
  H5Aclose( h5a );

  for( int i=0; i<nsp; i++ ){
    std::ostringstream gname;
    gname << "sp1d_" << i;
    const hid_t childGroup = H5Gopen( group, gname.str().c_str() );
    BSpline1D * sp = new BSpline1D( enableValueClipping_ );
    sp->read_hdf5( childGroup );
    dim2Splines_.push_back( sp );
    H5Gclose( childGroup );
  }

  sp1_ = new BSpline1D( enableValueClipping_ );
  sp1_->read_hdf5( group );
#endif
}

//====================================================================

BSpline3D::BSpline3D( const int order,
		      const std::vector<double> & x1,
		      const std::vector<double> & x2,
		      const std::vector<double> & x3,
		      const std::vector<double> & phi,
		      const bool allowClipping ) :
  BSpline( order, 3, allowClipping ),
    n1_( (int)x1.size() ),
    n2_( (int)x2.size() ),
    n3_( (int)x3.size() ),
    sp1_( NULL )
{
  const int nn = (int)phi.size();

  if( nn != n1_*n2_*n3_ ){
    std::ostringstream errmsg;
    errmsg << "ERROR in BSpline3D::BSpline3D()!  Inconsistent dimensionality between" << std::endl
	   << "      independent (" << n1_*n2_*n3_ << ") and dependent (" << nn << ") variables"
	   << std::endl;
    throw InternalError( errmsg.str(), __FILE__, __LINE__ );
  }
  
  compute_control_pts( x1, x2, x3, phi );
}

//--------------------------------------------------------------------

BSpline3D::BSpline3D( const bool allowClipping ) :
  BSpline( 0, 3, allowClipping ),
  n1_(0), n2_(0), n3_(0),
  sp1_( NULL )
{
}

//--------------------------------------------------------------------

BSpline3D::BSpline3D( const BSpline3D& src ) :
  BSpline( src.order_, 3, src.enableValueClipping_ ),
  n1_( src.n1_ ),
  n2_( src.n2_ ),
  n3_( src.n3_ )
{
  // deep copy
  sp1_ = new BSpline1D( *(src.sp1_) );

  vector<const BSpline2D*>::const_iterator isp;
  for( isp=src.sp2d_.begin(); isp!=src.sp2d_.end(); isp++ ){
    sp2d_.push_back( new BSpline2D( **isp ) );
  }
}

//--------------------------------------------------------------------

BSpline3D::~BSpline3D()
{
  vector<const BSpline2D*>::const_iterator ibs;
  for( ibs=sp2d_.begin(); ibs!=sp2d_.end(); ibs++ ) {
    delete *ibs;
  }

  delete sp1_;
}

//--------------------------------------------------------------------

void
BSpline3D::compute_control_pts( const std::vector<double> & x1,
				const std::vector<double> & x2,
				const std::vector<double> & x3,
				const std::vector<double> & phi )
{
  //
  // consider n1_ curves through planes in j-k space.  Spline these
  // to get R_{ijk} from
  //
  //   Q = sum_i N_i(u) R_{ijk}
  //
  // Then we will do a 2D fit.
  //

  vector< vector<double> > RR;
  //  vector<double> dpvrs( n1_*n2_ );
  vector<double> dpvrs( n1_ );

  bool first = true;
  for( int k=0; k<n3_; k++ ){
    for( int j=0; j<n2_; j++ ){

      // get the dependent variable vector for this curve
      const int shift = n1_*( k*n2_ + j );
      for( int i=0; i<n1_; i++ ){
        dpvrs[i] = phi[i+shift];
      }
      // spline it.
      BSpline1D sp( order_, x1, dpvrs );

      // save off the control points for use in the next step...
      RR.push_back( sp.get_control_pts() );

      if( first ){
	// create the 1D-spline to be used for the first dimension in
	// lookups later.  Note that we can do this because the knot
	// sequence is constant across all lines in a given direction.
	// the control points will change, however....
 	sp1_ = new BSpline1D( sp );
	first = false;
      }
    }
  }

  // resize the dep var array for the 2d interpolant construction
  dpvrs.assign( n2_*n3_, 0.0 );

  // spline the set of 2-D surfaces
  for( int i=0; i<n1_; i++ ){

    // load the dependent variables vector
    int l=0;
    for( int k=0; k<n3_; k++ ){
      const int shift = k*n2_;
      for( int j=0; j<n2_; j++ ){
	dpvrs[l] = RR[j+shift][i];
	l++;
      }
    }

    // spline and store this 2D surface spline.
    BSpline2D * sp = new BSpline2D( order_, x2, x3, dpvrs );
    sp2d_.push_back( sp );
  }
}
//--------------------------------------------------------------------
double
BSpline3D::value( const double* x ) const
{
  //     Q   = \sum_i N_i(u) R_{ijk}
  // R_{ijk} = \sum_j N_j(v) \sum_k N_k(w) P_{ijk}

  vector<double> & R = sp1_->get_control_pts();
  assert( R.size() == sp2d_.size() );

  // obtain the "R" vector, and assign it to the first spline dimension
  // optimized to only do computations of the locations in the "R" vector
  // that will actually be used by the subsequent 1-D interpolation.
  const int p = sp1_->get_order();
  const int ix = find_indx( sp1_->get_npts(),
			    sp1_->get_order(),
			    get_uk(x[0],sp1_->get_maxval(), sp1_->get_minval(), enableValueClipping_),
			    sp1_->get_knot_vector() );

  const int shift = ix-p;
  vector<double>::iterator ir = R.begin()+shift;
  vector<double>::const_iterator irend = ir+p+1;
  vector<const BSpline2D*>::const_iterator jj = sp2d_.begin()+shift;
  const double query[2] = {x[1],x[2]};
  for( ; ir!=irend; ir++, jj++ ){
    *ir = (*jj)->value( query );
  }

  // get the interpolated value.
  return sp1_->value( &x[0] );

  /*
  // the brute-force method, calculating every point along the line...
  vector<const BSpline2D*>::const_iterator jj = sp2d_.begin();
  vector<double>::iterator ir = R.begin();
  for( ; ir!=R.end(); ir++, jj++ ){
    *ir = (*jj)->value( x[1], x[2] );
  }

  // get the interpolated value.
  return sp1_->value( x[0] );
  */
}
//--------------------------------------------------------------------

void
BSpline3D::write_hdf5( const hid_t & group ) const
{
#if defined( HAVE_HDF5 )
  const hid_t h5s = H5Screate( H5S_SCALAR );

  // write out the information for how many 2-D splines we have
  hid_t h5a = H5Acreate( group, "Number2Dsplines", H5T_NATIVE_INT, h5s, H5P_DEFAULT );
  const int nsp = (int)sp2d_.size();
  H5Awrite( h5a, H5T_NATIVE_INT, &nsp );
  H5Aclose( h5a );

  H5Sclose( h5s );

  // write out each of the 2-D splines in the array...
  int icount = 0;
  vector<const BSpline2D*>::const_iterator ibs;
  for( ibs=sp2d_.begin(); ibs!=sp2d_.end(); ibs++, icount++ ){
    // create a group for this spline
    std::ostringstream gname;
    gname << "sp2d_" << icount;
    const hid_t childGroup = H5Gcreate( group, gname.str().c_str(), 0 );
    (*ibs)->write_hdf5( childGroup );
    H5Gclose( childGroup );
  }

  // dump out the spline for the other dimension
  sp1_->write_hdf5( group );
#endif
}

//--------------------------------------------------------------------

void
BSpline3D::read_hdf5( const hid_t & group )
{
#if defined( HAVE_HDF5 )
  int nsp=0;
  hid_t h5a = H5Aopen_name( group, "Number2Dsplines" );
  H5Aread( h5a, H5T_NATIVE_INT, &nsp );
  H5Aclose( h5a );

  for( int i=0; i<nsp; i++ ){
    std::ostringstream gname;
    gname << "sp2d_" << i;
    const hid_t childGroup = H5Gopen( group, gname.str().c_str() );
    BSpline2D * sp = new BSpline2D( enableValueClipping_ );
    sp->read_hdf5( childGroup );
    sp2d_.push_back( sp );
    H5Gclose( childGroup );
  }

  sp1_ = new BSpline1D( enableValueClipping_ );
  sp1_->read_hdf5( group );
#endif
}

//--------------------------------------------------------------------

//====================================================================

//--------------------------------------------------------------------
BSpline4D::BSpline4D( const int order,
		      const std::vector<double> & x1,
		      const std::vector<double> & x2,
		      const std::vector<double> & x3,
		      const std::vector<double> & x4,
		      const std::vector<double> & phi,
		      const bool allowClipping )
  : BSpline( order, 4, allowClipping ),
    n1_( (int)x1.size() ),
    n2_( (int)x2.size() ),
    n3_( (int)x3.size() ),
    n4_( (int)x4.size() ),
    sp1_( NULL )
{
  if( n1_*n2_*n3_*n4_ != (int)phi.size() ) {
    ostringstream errmsg;
    errmsg << "ERROR in BSpline4D::BSpline4D()!  Inconsistent dimensionality between" << std::endl
	   << "      independent (" << n1_*n2_*n3_*n4_ << ") and dependent (" << phi.size()
	   << ") variables" << std::endl;
    throw InternalError( errmsg.str(), __FILE__, __LINE__ );
  }
  
  compute_control_pts( x1, x2, x3, x4, phi );
}
//--------------------------------------------------------------------
BSpline4D::BSpline4D( const bool allowClipping )
  : BSpline( 0, 4, allowClipping ),
    n1_(0), n2_(0), n3_(0), n4_(0),
    sp1_( NULL )
{
}
//--------------------------------------------------------------------
BSpline4D::BSpline4D( const BSpline4D& src )
  : BSpline( src.order_, 4, src.enableValueClipping_ ),
    n1_( src.n1_ ),
    n2_( src.n2_ ),
    n3_( src.n3_ ),
    n4_( src.n4_ )
{
  // deep copy
  sp1_ = new BSpline1D( *(src.sp1_) );

  vector<const BSpline3D*>::const_iterator isp;
  for( isp=src.sp3d_.begin(); isp!=src.sp3d_.end(); isp++ ){
    sp3d_.push_back( new BSpline3D( **isp ) );
  }
}
//--------------------------------------------------------------------
BSpline4D::~BSpline4D()
{
  vector<const BSpline3D*>::const_iterator isp;
  for( isp=sp3d_.begin(); isp!=sp3d_.end(); isp++ )
    delete *isp;

  delete sp1_;
}
//--------------------------------------------------------------------
void
BSpline4D::compute_control_pts( const std::vector<double> & x1,
				const std::vector<double> & x2,
				const std::vector<double> & x3,
				const std::vector<double> & x4,
				const std::vector<double> & phi )
{
  //
  // consider n1_ curves through volumes in j-k-l space.  Spline these
  // to get R_{ijkl} from
  //
  //   Q = sum_i N_i(u) R_{ijkl}
  //
  // Then we will do a 3D fit.
  //

  vector< vector<double> > RR;
  vector<double> dpvrs( n1_ );

  bool first = true;
  for( int l=0; l<n4_; l++ ){
    for( int k=0; k<n3_; k++ ){
      for( int j=0; j<n2_; j++ ){

	// get the dependent variable vector for this curve
	const int shift =   n1_*( j + n2_*(k+l*n3_) );
	for( int i=0; i<n1_; i++ ) dpvrs[i] = phi[i+shift];

	// spline it.
	BSpline1D sp( order_, x1, dpvrs );

	// save off the control points for use in the next step...
	RR.push_back( sp.get_control_pts() );

	if( first ){
	  // create the 1D-spline to be used for the first dimension in
	  // lookups later.  Note that we can do this because the knot
	  // sequence is constant across all lines in a given direction.
	  // the control points will change, however....
	  sp1_ = new BSpline1D( sp );
	  first = false;
	}
      }
    }
  }

  // resize the dep var array for the 3d interpolant construction
  dpvrs.assign( n2_*n3_*n4_, 0.0 );

  // spline the set of 3-D volumes
  for( int i=0; i<n1_; i++ ){

    // load the dependent variables vector
    int m=0;
    for( int l=0; l<n4_; l++ ){
      for( int k=0; k<n3_; k++ ){
	const int shift = n2_*(k+ l*n3_);
	for( int j=0; j<n2_; j++ ){
	  dpvrs[m] = RR[j+shift][i];
	  m++;
	}
      }
    }

    // spline and store this 2D surface spline.
    BSpline3D * sp = new BSpline3D( order_, x2, x3, x4, dpvrs );
    sp3d_.push_back( sp );
  }
}
//--------------------------------------------------------------------
double
BSpline4D::value( const double* x ) const
{
  //       Q  = \sum_i N_i(u) R_{ijkl}
  // R_{ijkl} = \sum_j N_j(v) \sum_k N_k(w) P_{ijkl}

  vector<double> & R = sp1_->get_control_pts();
  assert( R.size() == sp3d_.size() );

  // obtain the "R" vector, and assign it to the first spline dimension
  // optimized to only do computations of the locations in the "R" vector
  // that will actually be used by the subsequent 1-D interpolation.
  const int p = sp1_->get_order();
  const int ix = find_indx( sp1_->get_npts(),
			    sp1_->get_order(),
			    get_uk(x[0],sp1_->get_maxval(), sp1_->get_minval(), enableValueClipping_),
			    sp1_->get_knot_vector() );

  const int shift = ix-p;
  vector<double>::iterator ir = R.begin()+shift;
  vector<double>::const_iterator irend = ir+p+1;
  vector<const BSpline3D*>::const_iterator jj = sp3d_.begin()+shift;
  const double query[3] = {x[1],x[2],x[3]};
  for( ; ir!=irend; ir++, jj++ ){
    //    *ir = (*jj)->value( x[1], x[2], x[3] );
    *ir = (*jj)->value( query );
  }

  // get the interpolated value.
  return sp1_->value( &x[0] );

}
//--------------------------------------------------------------------

void
BSpline4D::write_hdf5( const hid_t & group ) const
{
#if defined( HAVE_HDF5 )
  const hid_t h5s = H5Screate( H5S_SCALAR );

  // write out the information for how many 3-D splines we have
  hid_t h5a = H5Acreate( group, "Number3Dsplines", H5T_NATIVE_INT, h5s, H5P_DEFAULT );
  const int nsp = (int)sp3d_.size();
  H5Awrite( h5a, H5T_NATIVE_INT, &nsp );
  H5Aclose( h5a );

  H5Sclose( h5s );

  // write out each of the 3-D splines in the array...
  int icount = 0;
  vector<const BSpline3D*>::const_iterator ibs;
  for( ibs=sp3d_.begin(); ibs!=sp3d_.end(); ibs++, icount++ ){
    // create a group for this spline
    std::ostringstream gname;
    gname << "sp3d_" << icount;
    const hid_t childGroup = H5Gcreate( group, gname.str().c_str(), 0 );
    (*ibs)->write_hdf5( childGroup );
    H5Gclose( childGroup );
  }

  // dump out the spline for the other dimension
  sp1_->write_hdf5( group );
#endif
}

//--------------------------------------------------------------------

void
BSpline4D::read_hdf5( const hid_t & group )
{
#if defined( HAVE_HDF5 )
  int nsp=0;
  hid_t h5a = H5Aopen_name( group, "Number3Dsplines" );
  H5Aread( h5a, H5T_NATIVE_INT, &nsp );
  H5Aclose( h5a );

  for( int i=0; i<nsp; i++ ){
    std::ostringstream gname;
    gname << "sp3d_" << i;
    const hid_t childGroup = H5Gopen( group, gname.str().c_str() );
    BSpline3D * sp = new BSpline3D( enableValueClipping_ );
    sp->read_hdf5( childGroup );
    sp3d_.push_back( sp );
    H5Gclose( childGroup );
  }

  sp1_ = new BSpline1D( enableValueClipping_ );
  sp1_->read_hdf5( group );
#endif
}

//====================================================================

BSpline5D::BSpline5D( const int order,
		      const std::vector<double> & x1,
		      const std::vector<double> & x2,
		      const std::vector<double> & x3,
		      const std::vector<double> & x4,
		      const std::vector<double> & x5,
		      const std::vector<double> & phi,
		      const bool allowClipping ) :
  BSpline( order, 5, allowClipping ),
  n1_( (int)x1.size() ),
  n2_( (int)x2.size() ),
  n3_( (int)x3.size() ),
  n4_( (int)x4.size() ),
  n5_( (int)x5.size() ),
  sp1_( NULL )
{
  if( n1_*n2_*n3_*n4_*n5_ != (int)phi.size() ) {
    std::ostringstream errmsg;
    errmsg << "ERROR in BSpline5D::BSpline5D()!  Inconsistent dimensionality between"  << std::endl
	   << "      independent (" << n1_*n2_*n3_*n4_*n5_ << ") and dependent (" << phi.size()
	   << ") variables"  << std::endl;
    throw InternalError( errmsg.str(), __FILE__, __LINE__ );
  }
  
  compute_control_pts( x1, x2, x3, x4, x5, phi );
}

//--------------------------------------------------------------------

BSpline5D::BSpline5D( const bool allowClipping ) :
  BSpline( 0, 5 ),
  n1_(0), n2_(0), n3_(0), n4_(0), n5_(0),
  sp1_( NULL )
{
}

//--------------------------------------------------------------------

BSpline5D::BSpline5D( const BSpline5D& src ) :
  BSpline( src.order_, 5, src.enableValueClipping_ ),
  n1_( src.n1_ ),
  n2_( src.n2_ ),
  n3_( src.n3_ ),
  n4_( src.n4_ ),
  n5_( src.n5_ )
{
  // deep copy
  sp1_ = new BSpline1D( *(src.sp1_) );

  vector<const BSpline4D*>::const_iterator isp;
  for( isp=src.sp4d_.begin(); isp!=src.sp4d_.end(); isp++ )
    sp4d_.push_back( new BSpline4D(**isp) );
}

//--------------------------------------------------------------------

BSpline5D::~BSpline5D()
{
  vector<const BSpline4D*>::const_iterator isp;
  for( isp=sp4d_.begin(); isp!=sp4d_.end(); isp++ ) {
    delete *isp;
  }
  delete sp1_;
}

//--------------------------------------------------------------------

void
BSpline5D::compute_control_pts( const std::vector<double> & x1,
				const std::vector<double> & x2,
				const std::vector<double> & x3,
				const std::vector<double> & x4,
				const std::vector<double> & x5,
				const std::vector<double> & phi )
{
  //
  // consider n1_ curves through volumes in j-k-l space.  Spline these
  // to get R_{ijkl} from
  //
  //   Q = sum_i N_i(u) R_{ijkl}
  //
  // Then we will do a 4D fit.
  //

  vector< vector<double> > RR;
  vector<double> dpvrs( n1_ );

  bool first = true;
  for( int m=0; m<n5_; m++ ){
    const int mshift = m*n4_*n3_*n2_*n1_;
    for( int l=0; l<n4_; l++ ){
      const int lshift = mshift + l*n3_*n2_*n1_;
      for( int k=0; k<n3_; k++ ){
	const int kshift = lshift + k*n2_*n1_;
	for( int j=0; j<n2_; j++ ){
	  const int jshift = kshift+j*n1_;

	  // get the dependent variable vector for this curve
	  for( int i=0; i<n1_; i++ ){ 
          dpvrs[i] = phi[i+jshift];
         }
	  // spline it.
	  BSpline1D sp( order_, x1, dpvrs );

	  // save off the control points for use in the next step...
	  RR.push_back( sp.get_control_pts() );

	  if( first ){
	    // create the 1D-spline to be used for the first dimension in
	    // lookups later.  Note that we can do this because the knot
	    // sequence is constant across all lines in a given direction.
	    // the control points will change, however....
	    sp1_ = new BSpline1D( sp );
	    first = false;
	  }
	}
      }
    }
  }

  // resize the dep var array for the 4d interpolant construction
  dpvrs.assign( n2_*n3_*n4_*n5_, 0.0 );
  const int ndpvrs = (int)dpvrs.size();
  const int nRR = (int)RR.size();
  
  // spline the set of 4-D hyper-volumes
  for( int i=0; i<n1_; i++ ){
    // load the dependent variables vector
    int n=0;
    for( int m=0; m<n5_; m++ ){
      for( int l=0; l<n4_; l++ ){
	for( int k=0; k<n3_; k++ ){
	  const int shift = n2_*(k+ n3_*(l+n4_*(m)));
	  for( int j=0; j<n2_; j++ ){
	    if( n > ndpvrs-1 ) std::cout << "ERROR ON INDEX" << std::endl;
	    if( j+shift > nRR-1 ) std::cout << "ERROR2 on INDEX" << std::endl;
	    dpvrs[n] = RR[j+shift][i];
	    n++;
	  }
	}
      }
    }
    // spline and store this 4D surface spline.
    BSpline4D * sp = new BSpline4D( order_, x2, x3, x4, x5, dpvrs );
    sp4d_.push_back( sp );
  }
}
//--------------------------------------------------------------------
double
BSpline5D::value( const double* x ) const
{
  //       Q  = \sum_i N_i(u) R_{ijkl}
  // R_{ijkl} = \sum_j N_j(v) \sum_k N_k(w) P_{ijkl}

  vector<double> & R = sp1_->get_control_pts();
  assert( R.size() == sp4d_.size() );

  // obtain the "R" vector, and assign it to the first spline dimension
  // optimized to only do computations of the locations in the "R" vector
  // that will actually be used by the subsequent 1-D interpolation.
  const int p = sp1_->get_order();
  const int ix = find_indx( sp1_->get_npts(),
			    sp1_->get_order(),
			    get_uk(x[0],sp1_->get_maxval(), sp1_->get_minval(), enableValueClipping_),
			    sp1_->get_knot_vector() );

  const int shift = ix-p;
  vector<double>::iterator ir = R.begin()+shift;
  vector<double>::const_iterator irend = ir+p+1;
  vector<const BSpline4D*>::const_iterator jj = sp4d_.begin()+shift;
  const double query[4] = {x[1],x[2],x[3],x[4]};
  for( ; ir!=irend; ir++, jj++ ){
    *ir = (*jj)->value( query );
  }

  // get the interpolated value.
  return sp1_->value( &x[0] );
}
//--------------------------------------------------------------------

void
BSpline5D::write_hdf5( const hid_t & group ) const
{
#if defined( HAVE_HDF5 )
  const hid_t h5s = H5Screate( H5S_SCALAR );

  // write out the information for how many 4-D splines we have
  hid_t h5a = H5Acreate( group, "Number4Dsplines", H5T_NATIVE_INT, h5s, H5P_DEFAULT );
  const int nsp = (int)sp4d_.size();
  H5Awrite( h5a, H5T_NATIVE_INT, &nsp );
  H5Aclose( h5a );

  H5Sclose( h5s );
  
  // write out each of the 4-D splines in the array...
  int icount = 0;
  vector<const BSpline4D*>::const_iterator ibs;
  for( ibs=sp4d_.begin(); ibs!=sp4d_.end(); ibs++, icount++ ){
    // create a group for this spline
    std::ostringstream gname;
    gname << "sp4d_" << icount;
    const hid_t childGroup = H5Gcreate( group, gname.str().c_str(), 0 );
    (*ibs)->write_hdf5( childGroup );
    H5Gclose( childGroup );
  }

  // dump out the spline for the other dimension
  sp1_->write_hdf5( group );
#endif
}

//--------------------------------------------------------------------

void
BSpline5D::read_hdf5( const hid_t & group )
{
#if defined( HAVE_HDF5 )
  int nsp=0;
  hid_t h5a = H5Aopen_name( group, "Number4Dsplines" );
  H5Aread( h5a, H5T_NATIVE_INT, &nsp );
  H5Aclose( h5a );

  for( int i=0; i<nsp; i++ ){
    std::ostringstream gname;
    gname << "sp4d_" << i;
    const hid_t childGroup = H5Gopen( group, gname.str().c_str() );
    BSpline4D * sp = new BSpline4D( enableValueClipping_ );
    sp->read_hdf5( childGroup );
    sp4d_.push_back( sp );
    H5Gclose( childGroup );
  }

  sp1_ = new BSpline1D( enableValueClipping_ );
  sp1_->read_hdf5( group );
#endif
}
