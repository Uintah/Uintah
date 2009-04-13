#include <CCA/Components/SpatialOps/LU.h>

#include <cmath>
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <vector>

using namespace std;
using namespace Uintah;

//--------------------------------------------------------------------
LU::LU( const int dim, const int bandwidth )
  : dim_( dim ),
    AA_( dim, bandwidth )
{
}
//--------------------------------------------------------------------
LU::~LU()
{
}
//--------------------------------------------------------------------
void
LU::decompose()
{

  // Algorithm from Numerical Recipes in C, by Press et al

  int i, imax, j, k;
  double big, dum, sum, temp;
  double tiny = 1e-10;
  vector<double> vv;

  isSingular_ = false;

  // loop over rows to get the implicit scaling information
  for (i=1; i<=dim_; ++i) {
    big = 0.0;
    for (j=1; j<=dim_; ++j) {
      temp = fabs(AA_(i-1,j-1));
      if ( temp > big ) {
        big = temp;
      }
    }
    if (big == 0.0) {
      // return error "Singular matrix in routine ludcmp"
      //cout << "Singular matrix in routine LU::decompose." << endl;
      isSingular_ = true;
      isReady_ = true;
      return;
    }
    // save the scaling
    vv.push_back(1.0/big);
  }
  cout << endl;

  // Loop over columns for Crout's method
  for (j=1; j<=dim_; ++j) {
    
    // Inner loop 1: solve for elements of U (beta_ij in book's notation) (don't do i=j)
    if (j>1) {
      for (i=1; i<j; ++i) {
        sum = AA_(i-1,j-1);
        for (k=1; k<i; ++k) {
          sum -= AA_(i-1,k-1)*AA_(k-1,j-1);
        }
        AA_(i-1,j-1)=sum;
      }
    }

    // initialize search for biggest pivot element
    big = 0.0;

    // Inner loop 2: solve for elements of L (alpha_ij in book's notation) (include i=j)
    for (i=j; i<=dim_; ++i) {
      sum = AA_(i-1,j-1);
      if (j>1) {
        for (k=1; k<j; ++k) {
          sum -= AA_(i-1,k-1)*AA_(k-1,j-1);
        }
        AA_(i-1,j-1)=sum;
      }
      dum = vv[i-1]*fabs(sum);
      if ( dum >= big ) {
        // is the figure of merit for the pivot better than the best so far?
        big = dum;
        imax = i;
      }
    }

    // Inner loop 3: check if you need to interchange rows
    if (j != imax) {
      // yes, you do
      for (k=1; k<=dim_; ++k) {
        dum = AA_(imax-1,k-1);
        AA_(imax-1,k-1) = AA_(j-1,k-1);
        AA_(j-1,k-1)=dum;
      }
      // interchange scale factor too
      vv[imax-1]=vv[j-1];
    }

    indx.push_back(imax);

    // Inner loop 4: replace zero w/ tiny value
    if (AA_(j-1,j-1) == 0.0) {
      AA_(j-1,j-1) = tiny;
    }

    // Inner loop 5: divide by the pivot element
    if (j != dim_)
    {
      dum = 1.0/AA_(j-1,j-1);
      for (i=j+1; i<=dim_; ++i) {
        double temp = AA_(i-1,j-1) * dum;
        AA_(i-1,j-1) = temp;
      }
    }
  
  }// end loop over columns for Crout's method

  if ( AA_(dim_-1,dim_-1) == 0 ) {
    AA_(dim_-1,dim_-1) = tiny;
  }

  isReady_ = true;
}
//--------------------------------------------------------------------
void
LU::back_subs( double* rhs )
{
  if( ! isReady_ )
    throw std::runtime_error( "LU::back_subs() cannot be executed until LU::decompose() has been called!" );

  if( isSingular_ ) {
    for (int i=0; i<dim_; ++i) {
      rhs[i] = 0;
    }
    return;
  }

  // AA_ now contains the LU-decomposition of the original "A" matrix.
  // rhs[0] is untouched for now since L(0,0) = 1.

  // algorithm from Numerical Recipes (C):

  int i, j, ii=0, ip=0;
  float sum;

  // forward substitution
  for (i=1; i<=dim_; ++i) {
    ip = indx[i-1];
    sum=rhs[ip-1];
    rhs[ip-1]=rhs[i-1];
    if (ii) {
      for (j=ii; j<=i-1; j++) {
        sum -= AA_(i-1,j-1)*rhs[j-1];
      }
    } else if (sum) {
      // a nonzero element was encountered
      // from now on, sum in above loop must be done
      ii = i;
    }
    rhs[i-1] = sum;
  }

  // back-substitution
  for (i=dim_; i>=1; i--) {
    sum = rhs[i-1];
    for (j=i+1; j<=dim_; j++) {
      sum -= AA_(i-1,j-1)*rhs[j-1];
    }
    // store component of solution vector
    rhs[i-1] = sum/AA_(i-1,i-1);
  }

}
//--------------------------------------------------------------------
LU::SparseMatrix::SparseMatrix( const int dim,
				const int bandwidth )
  : dim_( dim ),
    band_( bandwidth )
{
  // for now, store densely
  AA_ = new double*[dim];
  for( int i=0; i<dim; i++ ){
    AA_[i] = new double[dim];
    for( int j=0; j<dim; j++ )  AA_[i][j]=0.0;
  }
}
//--------------------------------------------------------------------
LU::SparseMatrix::~SparseMatrix()
{
  for( int i=0; i<dim_; i++ ) delete [] AA_[i];
  delete [] AA_;
}
//--------------------------------------------------------------------
void
LU::dump()
{
  using std::cout;
  using std::endl;

  for( int i=0; i<dim_; i++ ){
    for( int j=0; j<dim_; j++ ){
      cout << std::setw(9) << std::setprecision(4) << AA_(i,j) << "  ";
    }
    cout << endl;
  }
  cout << "-----------------------------------------------------" << endl;
}
//--------------------------------------------------------------------
