#include <CCA/Components/Arches/LU.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Exceptions/InvalidValue.h>
#include <float.h>
//#include <cmath>
//#include <iostream>
//#include <iomanip>
//#include <stdexcept>

using namespace std;
using namespace Uintah;

//=========================================================
// Constructors/destructors

// Default constructor
LU::LU( const int dim, const int bandwidth )
  : dim_( dim ),
    AA_( dim, bandwidth )
{
  isDecomposed_ = false;
  isRefined_ = false;
}

// Copy constructor for LU object
LU::LU( LU &CopyThis ) : dim_( CopyThis.getDimension() ), 
                         AA_( CopyThis.getDimension(), CopyThis.getDimension() )
{
  const int CopyThisDim = CopyThis.getDimension();
  for (int i=0; i < CopyThisDim; ++i) {
    for (int j=0; j < CopyThisDim; ++j) {
      AA_(i,j) = CopyThis(i,j);
    }
  }

  isDecomposed_ = false;
  isRefined_ = false;
}

// Default destructor
LU::~LU()
{
}

//========================================================



//=========================================================
// Solution functions

/** @details    The solve() method calls the
  *             appropriate (private) methods 
  *             after it decides whether or not to
  *             use iterative refinement to solve AX=B.   \n
  */
void 
LU::solve( double* rhs, double* soln, bool doIterativeSolve )
{
  decompose();
  back_subs( &rhs[0], &soln[0] );
  if (doIterativeSolve) {
    iterative_refinement( &rhs[0], &soln[0] );
  }
}

//========================================================



//========================================================
// Utility functions

/* -------------- not used ---------------------
double
LU::getNorm( double* a, unsigned int type )
{
  double TheNorm = 0;
  
  // Norm types:
  // 0 - L_infty norm
  // 1 - L1/Manhattan/taxicab norm
  // 2 - L2 norm
  //  ...
  // p - Lp norm
  if (type > 5) {
    proc0cout << "You want WHAT kind of norm???" << endl;
    proc0cout << "I will assume you just want an L_infty norm." << endl;
    type = 0;
  }

  if (type == 0) {
    // L_infty norm
    for (int z = 0; z < dim_; ++z) {
      if (abs(a[z]) > TheNorm) {
        TheNorm = abs(a[z]);
      }
    }
  } else if (type <= 5) {
    // L_p norm
    // || x ||_p   [=]   ( sum( |x_i|^p ) )^(1/p)
    for (int z = 0; z < dim_; ++z) {
      TheNorm += pow( pow(abs(a[z]),(int)type) , 1.0/type );
    }
  } 
  
  return TheNorm;
}
*/

void
LU::dump()
{
  for( int i=0; i<dim_; i++ ){
    for( int j=0; j<dim_; j++ ){
      proc0cout << std::setw(9) << std::setprecision(4) << AA_(i,j) << "  ";
    }
    proc0cout << endl;
  }
  proc0cout << "-----------------------------------------------------" << endl;
}

//========================================================



//========================================================
// LU Decomposition & Back-Substitution via Crout's Method

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
      // A matrix contains all-zero row, so it is singular; set solution vector equal to 0
      isSingular_ = true;
      isDecomposed_ = true;
      return;
    }
    // save the scaling
    vv.push_back(1.0/big);
  }
  //cout << endl;

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

  isDecomposed_ = true;
}


void
LU::back_subs( double* rhs, double* soln )
{
  if( ! isDecomposed_ ) {
    string err_msg = "ERROR:LU:back_subs(): This method cannot be called until LU::decompose() has been executed.\n";
    throw InvalidValue(err_msg,__FILE__,__LINE__);
  } 

  // Pointers of type double* are actually pointing to the first element of an array
  // So... I have no idea how to get the size of these vectors
  // We have to assume the user is passing vectors of the right size.

  // Copy rhs vector into soln vector
  for ( int counter=0; counter < dim_; ++counter) {
    soln[counter] = rhs[counter];
  }

  if( isSingular_ ) {
    for (int i=0; i<dim_; ++i) {
      soln[i] = 0;
    }
    return;
  }

  // AA_ now contains the LU-decomposition of the original "A" matrix.
  // soln[0] is untouched for now since L(0,0) = 1.

  // Algorithm from Numerical Recipes (C):

  int i, j, ii=0, ip=0;
  float sum;

  // forward substitution
  for (i=1; i<=dim_; ++i) {
    ip = indx[i-1];
    sum=soln[ip-1];
    soln[ip-1]=soln[i-1];
    if (ii) {
      for (j=ii; j<=i-1; j++) {
        sum -= AA_(i-1,j-1)*soln[j-1];
      }
    } else if (sum) {
      // a nonzero element was encountered
      // from now on, sum in above loop must be done
      ii = i;
    }
    soln[i-1] = sum;
  }

  // back-substitution
  for (i=dim_; i>=1; i--) {
    sum = soln[i-1];
    for (j=i+1; j<=dim_; j++) {
      sum -= AA_(i-1,j-1)*soln[j-1];
    }
    // store component of solution vector
    soln[i-1] = sum/AA_(i-1,i-1);
  }

}

//========================================================



// ===============================================
// Iterative Refinement
vector<long double>
LU::iterative_refinement( double* rhs, double* soln )
{
  if (!isDecomposed_) {
    string err_msg = "ERROR:LU:iterative_refinement(): This method cannot be called until LU::decompose() has been executed.\n";
    throw InvalidValue(err_msg,__FILE__,__LINE__);
  } else if (isRefined_) {
    string err_msg = "ERROR:LU:iterative_refinement(): This method cannot be called twice!\n";
    throw InvalidValue(err_msg,__FILE__,__LINE__);
  }
  
  vector<long double> X_i(dim_);      /// Refined solution vector for iteration i
  vector<long double> res(dim_);      /// Residual matrix, double working precision (long doubles)
  vector<double> dX_ip1(dim_);        /// dX for iteration i+1 (curr. iter)
  vector<double> dX_i(dim_);          /// dX for iteration i (prev. iter)
  
  if( isSingular_ ) {
    for (int i=0; i<dim_; ++i) {
      X_i[i] = 0;
    }
    return X_i;
  }


  // put X_1 (soln passed as parameter) into X_i for first iteration
  for (int z = 0; z < dim_; ++z) {
    X_i[z] = soln[z];
  }

  // initialize relative norms
  double relnorm_dX_i    = 10^20; /// Relative L_infty norm of dX at previous iteration
  double relnorm_dX_ip1  = 10^20; /// Relative L_infty norm of dX at current iteration
  double relnorm_X_i     = 10^20; /// Relative L_infty norm of X                        
  final_rel_norm  = 10^20;
  
  // initialize convergence rates
  rho_thresh = 0.5;
  rho_max = 0; /// Maximum ratio of dX norms achieved
  
  // initialize solution state
  int x_state;    /// State of X:         \n
                  /// 0 = working         \n   
                  /// 1 = converged       \n   
                  /// 2 = no-progress
  x_state = 0;    // set x-state = working

  // refinement iteration loop...  
  imax = 20;
  for (int i = 0; i < imax; ++i) {
    // compute residual with very high intermediate precision:
    // does this mean A/X/B have to be very high interm. prec.?
    // or is residual matrix only matrix that has to be high prec.?
    getResidual(&soln[0], &rhs[0], &res[0]);

    // solve AdX(i+1) = r(i) using normal precision
    back_subs( &rhs[0], &dX_ip1[0] );

    relnorm_X_i = getNorm( &X_i[0], 0);
    relnorm_dX_ip1 = getNorm( &dX_ip1[0], 0 );

    // update x_state based on various stopping criteria
    update_xstate( relnorm_X_i, relnorm_dX_i, relnorm_dX_ip1, &x_state );
 
    if (x_state != 0) {
      break; //terminate the loop
    }

    //update X_ip1 (update X_i in place)
    for (int z = 0; z < dim_; ++z) {
      X_i[z] = X_i[z] - dX_ip1[z];
    }

  }
  
  // only update the final relative norm after last step if x_stae = working
  if (x_state == 0 ) {
    final_rel_norm = relnorm_dX_ip1/relnorm_X_i;
  }
  
  isRefined_ = true;
  return X_i;
}



void
LU::update_xstate( double norm_X_i, 
                   double norm_dX_i, 
                   double norm_dX_ip1,
                   int* x_state )
{
   if (!isDecomposed_) {
    string err_msg = "ERROR:LU:update_xstate(): This method cannot be called until LU::decompose() has been executed.\n";
    throw InvalidValue(err_msg,__FILE__,__LINE__);
  } else if (isRefined_) {
    string err_msg = "ERROR:LU:update_xstate(): This method cannot be called beause iterative refinement has already taken place!\n";
    throw InvalidValue(err_msg,__FILE__,__LINE__);
  }
  
  if (norm_dX_ip1/norm_X_i <= DBL_EPSILON) {
    (*x_state) = 1; //converged: tiny dX stopping criterion
  
  } else if ( (norm_dX_ip1 / norm_dX_i) < rho_thresh ) {
    (*x_state) = 2; //no progress: lack of progress stopping criterion
  
  } else {
    rho_max = max(rho_max, norm_dX_ip1/norm_dX_i);
  }
  
  if (x_state != 0) { //not working
    final_rel_norm = norm_dX_ip1/norm_X_i;
  }

}

// ===============================================



// ===============================================
// Sparse Matrix class

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

LU::SparseMatrix::~SparseMatrix()
{
  for( int i=0; i<dim_; i++ ) delete [] AA_[i];
  delete [] AA_;
}
//--------------------------------------------------------------------



