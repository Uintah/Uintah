#include <CCA/Components/Arches/LU.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Exceptions/InvalidValue.h>

#include <float.h>
#include <iostream>
#include <iomanip>

using namespace std;
using namespace Uintah;

//=========================================================
// Constructors/destructors

// Default constructor
LU::LU( const int dim )
  : dim_( dim ),
    AA_( dim )
{
    isDecomposed_ = false;
    isRefined_ = false;
}

// Copy constructor for LU object
LU::LU( LU &CopyThis ) : dim_( CopyThis.getDimension() ), 
                         AA_( CopyThis.getDimension() )
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



//========================================================
// LU Decomposition & Back-Substitution via Crout's Method
void
LU::decompose()
{
  int i, imax, j, k;
  double big, dum, sum, temp;
  double tiny = 1e-10;
  vector<double> vv;

  isSingular_ = false;
  isDecomposed_ = false;

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
      isDecomposed_ = true;
      isSingular_ = true;
      return;
    }
    // save the scaling
    vv.push_back(1.0/big);
  }

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
    if (j != dim_) {
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

  determinant = AA_.calculateDeterminant();
  //if( fabs(determinant) < 1e-16 ) {
  //  proc0cout << "A is singular, determinant = " << determinant << "." << endl;
  //  isSingular_ = true;
  //}

  isDecomposed_ = true;
}


void
LU::back_subs( double* rhs, double* soln )
{
  if( ! isDecomposed_ && ! isSingular_ ) {
    string err_msg = "ERROR:LU:back_subs(): This method cannot be called until LU::decompose() has been executed.\n";
    throw InvalidValue(err_msg,__FILE__,__LINE__);
  } else if (isSingular_) {
    for (int i=0; i<dim_; ++i) {
      soln[i] = 0;
    }
    return;
  }

  // NOTE: it is impossible to determine the size of an array from an array pointer
  // We have to assume the user is passing vectors of the right size.

  // Copy rhs vector into soln vector
  for ( int counter=0; counter < dim_; ++counter) {
    soln[counter] = rhs[counter];
  }

  // AA_ now contains the LU-decomposition of the original "A" matrix.
  // soln[0] is untouched for now since L(0,0) = 1.

  int i, j, ii=0, ip;
  double sum;

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



void
LU::back_subs( long double* rhs, double* soln )
{
  if( ! isDecomposed_ && !isSingular_ ) {
    string err_msg = "ERROR:LU:back_subs(): This method cannot be called until LU::decompose() has been executed.\n";
    throw InvalidValue(err_msg,__FILE__,__LINE__);
  } 

  if( isSingular_ ) {
    for (int i=0; i<dim_; ++i) {
      soln[i] = 0;
    }
    return;
  } else {
    // Copy rhs vector into soln vector
    for ( int counter=0; counter < dim_; ++counter) {
      soln[counter] = static_cast<double>(rhs[counter]);
    }
  }

  // AA_ now contains the LU-decomposition of the original "A" matrix.
  // soln[0] is untouched for now since L(0,0) = 1.

  // Algorithm from Numerical Recipes (C):

  int i, j, ii=0, ip;
  double sum;

  // Check size here
  // FIXME: I can't figure out how to get the size of an array from a pointer to its first element
  //if( rhs->size() != dim_ || soln->size() != dim_ ) {
    // Throw error:
    // LU::decompose(): Bad vector sizes
    // dim_     = ___
    // X.size() = ___
    // B.size() = ___
  //}

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



void
LU::iterative_refinement( LU Aoriginal, double* rhs, double* soln, long double* refined_soln )
{
  if (!isDecomposed_ && !isSingular_) {
    string err_msg = "ERROR:LU:iterative_refinement(): This method cannot be called until LU::decompose() has been executed.\n";
    throw InvalidValue(err_msg,__FILE__,__LINE__);
  } else if (isRefined_) {
    string err_msg = "ERROR:LU:iterative_refinement(): This method cannot be called twice!\n";
    throw InvalidValue(err_msg,__FILE__,__LINE__);
  } else if (isSingular_) {
    for (int i=0; i<dim_; ++i) {
      refined_soln[i] = 0;
    }
    return;
  }

  vector<double> dX_ip1(dim_, 0.0);        /// dX for iteration i+1 (curr. iter)
  vector<double> dX_i(dim_, 0.0);          /// dX for iteration i (prev. iter)
  vector<long double> res(dim_, 0.0);      /// Residual matrix, double working precision (long doubles)

  // copy coarse (passed) solution "soln" into refined (empty) solution "refined_soln" for 1st iteration
  for (int z = 0; z < dim_; ++z) {
    refined_soln[z] = static_cast<long double>(soln[z]);
  }

  // initialize relative norms
  norm_dX_i     = 0; //L-infinity norm of dX at timestep i
  norm_dX_ip1   = 0; //L-infinity norm of dX at timestep i+1
  norm_dX_final = 0; //L-infinity norm of dX at LAST timestep
  norm_X_i      = 0; //L-infinity norm of X at timestep i
  
  // initialize convergence rates
  rho_thresh = 0.5; // should be less than 1
  rho_max = 0; /// Maximum ratio of dX norms achieved
  
  // initialize solution state
  int x_state;    
  x_state = 0;    // 0 = working 
                  // 1 = converged 
                  // 2 = no-progress 

  imax = 20;
  // timestep i yields r(i), dx(i+1), x(i+1)
  for (int i = 0; i < imax; ++i) {

    // get norm of X(i) (soln for previous iteration)
    norm_X_i = getNorm( &refined_soln[0], 0 );

    // compute the residual res(i) in high precision
    // (note: use ORIGINAL A, because otherwise it's decomposed and RHS won't match!!!)
    Aoriginal.getResidual(&rhs[0], &refined_soln[0], &res[0]);

    // solve A * dX(i+1) = res(i) using normal precision
    back_subs( &res[0], &dX_ip1[0] );

    // norm of dX(i+1)
    norm_dX_ip1 = getNorm( &dX_ip1[0], 0 );

    // we need i > 0 b/c we need a solution vector for current AND previous timestep
    if( i > 0 ) {
      // update x_state based on various stopping criteria
      update_xstate( norm_X_i, norm_dX_i, norm_dX_ip1, &x_state );
    }
 
    if (x_state != 0 ) {
      break; //terminate the loop
    }

    //update X_ip1 (update X_i in place)
    for (int z = 0; z < dim_; ++z) {
      refined_soln[z] = refined_soln[z] - dX_ip1[z];
    }
    norm_dX_i = norm_dX_ip1;

  }
  
  // only update the final relative norm after last step if x_state = working
  // (otherwise it's already been done)
  if (x_state == 0 ) {
    norm_dX_final = norm_dX_ip1/norm_X_i;
  }

  // calculate condition number estimate
  // L-infinity norm of error vector e(i):
  if( rho_max != 1 ) {
    condition_estimate = norm_dX_final/( 1 - rho_max );
  } else {
    condition_estimate = 0;
  }
  
  isRefined_ = true;

}



void
LU::update_xstate( double norm_X_i, 
                   double norm_dX_i, 
                   double norm_dX_ip1,
                   int* x_state )
{
   if (!isDecomposed_ && !isSingular_) {
    string err_msg = "ERROR:LU:update_xstate(): This method cannot be called until LU::decompose() has been executed.\n";
    throw InvalidValue( err_msg, __FILE__, __LINE__ );
  } else if (isRefined_) {
    string err_msg = "ERROR:LU:update_xstate(): This method cannot be called beause iterative refinement has already taken place!\n";
    throw InvalidValue( err_msg, __FILE__, __LINE__ );
  }
  
  // stopping cirteria 1: correction dX(i+1) changes solution X(i) too little
  if (norm_dX_ip1/norm_X_i <= DBL_EPSILON) {
    (*x_state) = 1; //converged

  // stopping criteria 2: convergence slows down sufficiently
  //} else if ( (norm_dX_ip1/norm_dX_i) >= rho_thresh && (norm_dX_ip1/norm_dX_i) <= 1.0 ) {
  //  (*x_state) = 2; //lack of progress
  
  // stopping criteria 3 is imax...
  } else {
    rho_max = max(rho_max, norm_dX_ip1/norm_dX_i);
  }
  
  if (x_state != 0) { //not working
    norm_dX_final = norm_dX_ip1/norm_X_i;
  }

}

// ===============================================



//========================================================
// Utility functions
void
LU::dump()
{
  AA_.dump();
}

double
LU::getNorm( double* a, int type ) {
    double TheNorm = 0;
         
    if (type > 5) {
      type = 0;
    }
    if (type == 0) {
    // L_infty norm = max(abs(z)) 
      for (int z = 0; z < dim_; ++z) {
          if (abs(a[z]) > TheNorm) {
            TheNorm = abs(a[z]);
          }
        }
    } else if( type < 0 ) {
    // L_negative_infty norm = min(abs(z))
        TheNorm = a[0];
        for( int z=1; z<dim_; ++z ) {
          if( abs(a[z]) < TheNorm ) {
            TheNorm = abs(a[z]);
          }
        }
    } else if (type <= 5) {
    // L_p norm = 
    // || x ||_p   [=]   ( sum( |x_i|^p ) )^(1/p)
      for (int z = 0; z < dim_; ++z) {
        double rhs = pow( abs(a[z]), type );
        TheNorm += rhs;
      }
      TheNorm = pow( TheNorm, 1.0/type );
    } 
    return TheNorm;
}

double
LU::getNorm( long double* a, int type ) {
    double TheNorm = 0;
         
    if (type > 5) {
      type = 0;
    }
    if (type == 0) {
    // L_infty norm = max(abs(z)) 
      for (int z = 0; z < dim_; ++z) {
          if (abs(a[z]) > TheNorm) {
            TheNorm = abs(a[z]);
          }
        }
    } else if( type < 0 ) {
    // L_negative_infty norm = min(abs(z))
        TheNorm = a[0];
        for( int z=1; z<dim_; ++z ) {
          if( abs(a[z]) < TheNorm ) {
            TheNorm = abs(a[z]);
          }
        }
    } else if (type <= 5) {
    // L_p norm = 
    // || x ||_p   [=]   ( sum( |x_i|^p ) )^(1/p)
      for (int z = 0; z < dim_; ++z) {
        double rhs = pow( abs(a[z]), type );
        TheNorm += rhs;
      }
      TheNorm = pow( TheNorm, 1.0/type );
    } 
    return TheNorm;
}

//========================================================





// ===============================================
// Sparse Matrix class

// Default constructor
LU::DenseMatrix::DenseMatrix( const int dim ) : dim_( dim )
{
    AA_ = new double*[dim];
    for( int i=0; i < dim; ++i ) {
        AA_[i] = new double[dim];
        for( int j=0; j<dim; ++j ) {
            AA_[i][j] = 0.0;
        }
    }
}

LU::DenseMatrix::~DenseMatrix()
{
  for( int i=0; i<dim_; i++ ) delete [] AA_[i];
  delete[] AA_;
}



void
LU::DenseMatrix::dump()
{
  for( int i=0; i<dim_; ++i ) {
    for( int j=0; j<dim_; ++j ) {
      proc0cout << std::setw(9) << std::setprecision(4) << AA_[i][j] << "  ";
    }
    proc0cout << endl;
  }
}


double
LU::DenseMatrix::calculateDeterminant()
{
  double det = 1;
  for( int i = 0; i < dim_; ++i ) {
    det *= AA_[i][i];
  }
  return det;
}


// ===============================================

