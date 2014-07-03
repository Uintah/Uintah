#ifndef Uintah_Components_Arches_CQMOMInversion_h
#define Uintah_Components_Arches_CQMOMInversion_h

/*
 * The MIT License
 *
 * Copyright (c) 1997-2014 The University of Utah
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

//------------------------ CQMOMInversion.h -----------------------------------

#include <sci_defs/uintah_defs.h> // For FIX_NAME

// declare lapack eigenvalue solver
extern "C"{
# define DSYEV FIX_NAME(dsyev)
  void DSYEV( char* jobz, char* uplo, int* n, double* a, int* lda,
              double* w, double* work, int* lwork, int* info );
  
# define DGESV FIX_NAME(dgesv)
  void DGESV(int *n, int *nrhs, double *a, int *lda,
             int *ipiv, double *b, int *ldb, int *info);
}

//uncomment to debug matricies
//#define cqmom_dbg
using namespace std;
using namespace Uintah;
//-------------------------------------------------------

/**
 * @class    CQMOMInversion
 * @author   Alex Abboud
 * @date     May 2014 first iteration
 *
 * @brief    This class is the actual CQMOM calculation.  Given a set of moments, this will return all the weights and abscissas needed.
 *
 *
 * @details  Given M internal coordinates, and N_i quadrature nodes this will return a flat array for the weights and abscissas of the CQMOM.
 *           For the first iteration this will only include one permutation of the CQMOM method.  If significant numerical error occurs, then
 *           all of the permutations may be required to be calculated.
 *
 */

//-------------------------------------------------------

/***************************
 N-D Vandermonde alogrithm - from Numerical recipes
 **************************/
// solves sum_0^{n-1} x_i^k w_i = q_k for w_i
// x, q input, w output
void vandermondeSolve ( const std::vector<double> &x, std::vector<double> &w, const std::vector<double> &q)
{
  int n = q.size();
  double b,s,t,xx;
  std::vector<double> c (n,0.0);
  
  if (n == 1) {
    w[0] = q[0];
  } else {
    c[n-1] = -x[0];
    for (int i = 1; i < n; i++) {
      xx = -x[i];
      for (int j=(n-1-i); j<(n-1); j++) {
        c[j] += xx * c[j+1];
      }
      c[n-1] += xx;
    }
    //NOTE: This c is the same for all k columns, take advantage of this later
    
    for (int i = 0; i < n;i++) {
      xx=x[i];
      t=b=1.0;
      s=q[n-1];
      for ( int k=n-1; k>0; k--) {
        b = c[k] + xx * b;
        s += q[k-1] * b;
        t = xx *t+b;
      }
      if ( t!= 0.0 ) {
        w[i] = s/t;
      } else {
        w[i] = q[i];
      }
    }
  }
}


/***************************
 N-D Wheeler alogrithm
 ****************************/
void wheelerAlgorithm(const std::vector<double>& moments, std::vector<double>& w, std::vector<double>& x)
{
  int nEnv = (int) moments.size()/2; //get # nodes
  int nMom = moments.size();
#ifdef cqmom_dbg
  cout << "Wheeler Start" << endl;
  for (int i = 0; i < nMom; i++) {
    cout << "m[" << i << "]=" << moments[i] << endl;
  }
#endif
  if (nEnv == 1) {
    w[0] = moments[0];
    x[0] = moments[1]/moments[0];
    return;
  }
  
  std::vector<double> a (nEnv, 0.0);
  std::vector<double> b (nEnv, 0.0);
  std::vector< std::vector<double> > sig (2*nEnv+1, std::vector<double> (2*nEnv+1,0.0) );
  
  for ( int i = 1; i<(2*nEnv+1); i++) {
    sig[1][i] = moments[i-1];
  }
  a[0] = moments[1]/moments[0];
  
  for (int k = 2; k<(nEnv+1); k++) {
    for (int j = k; j<(2*nEnv-k+2); j++) {
      sig[k][j] = sig[k-1][j+1]-a[k-2]*sig[k-1][j]-b[k-2]*sig[k-2][j];
    }
    a[k-1] = sig[k][k+1]/sig[k][k] - sig[k-1][k]/sig[k-1][k-1];
    b[k-1] = sig[k][k]/sig[k-1][k-1];
  }

#ifdef cqmom_dbg
  for (int i=0; i<nEnv; i++) {
    cout << "a[" << i << "] = " << a[i] <<endl;
  }
  for (int i=0; i<nEnv; i++) {
    cout << "b[" << i << "] = " << b[i] <<endl;
  }

  for (int i=0; i<2*nEnv+1; i++) {
    for (int j = 0; j<2*nEnv+1; j++) {
      cout << "S[" << i << "][" << j << "] = " << sig[i][j] << "  ";
    }
    cout << endl;
  }
#endif
  
  //check the b vector for realizable space
  for ( int i = 0; i<nEnv; i++ ) {
    if (b[i] < 0) {
      stringstream err_msg;
      err_msg << "ERROR: Arches: CQMOMInversion: Negative b vector encountered. Moment set: " << endl;
      for (int i = 0; i<nMom; i++) {
        err_msg << "m[" << i << "]=" << moments[i] << endl;
      }
      throw InvalidValue(err_msg.str(),__FILE__,__LINE__);
      cout << endl;
    }
  }
  
#ifdef cqmom_dbg
  std::vector< std::vector<double> > z (nEnv, std::vector<double> (nEnv, 0.0) );
  
  for ( int i = 0; i<(nEnv-1); i++) {
    z[i][i] = a[i];
    z[i][i+1] = sqrt( b[i+1] );
    z[i+1][i] = z[i][i+1];
  }
  z[nEnv-1][nEnv-1] = a[nEnv-1];
  cout << "made z matrix" << endl;
  for (int i=0; i<nEnv; i++) {
    for (int j = 0; j<nEnv; j++) {
      cout << "z[" << i << "][" << j << "] = " << z[i][j] << "  ";
    }
    cout << endl;
  }
#endif

  std::vector<double> z_(nEnv*nEnv,0.0);
  for (int i = 0; i<(nEnv-1); i++) {
    z_[i*nEnv + i] = a[i];
    z_[i*nEnv + i + 1] = sqrt( b[i+1] );
    z_[(i+1)*nEnv + i] = z_[i*nEnv + i + 1];
  }
  z_[nEnv*nEnv-1] = a[nEnv-1];

  std::vector<double> eigenVal (nEnv, 0.0);
  //_____________
  //solve eigenvalue problem with external package NOTE:try timing based on work definition?
  int  lda = nEnv, info, lwork;
  double wkopt;
  double* work;
  //  std::vector<double> work_;
  lwork = -1;
  char jobz='V';
  char matType = 'U';
  DSYEV( &jobz, &matType, &nEnv, &z_[0], &lda, &eigenVal[0], &wkopt, &lwork, &info ); //with -1 this finds work size
  lwork = (int)wkopt;
  //  work_.resize(lwork);
  work = new double[lwork];
  // Solve eigenproblem. eigenvectors are stored in the z_ matrix, columnwise
  //  dsyev_( &jobz, &matType, &n, &z_[0], &lda, &eigenVal[0], &work_[0], &lwork, &info );
  DSYEV( &jobz, &matType, &nEnv, &z_[0], &lda, &eigenVal[0], work, &lwork, &info );
  bool status = ( info>0 || info<0 )? false : true;
  
  if (!status) {
    stringstream err_msg;
    err_msg << "ERROR: Arches: CQMOMInversion: Solving Eigenvalue problem failed. Moment set: " << endl;
    for (int i = 0; i<nMom; i++) {
      err_msg << "m[" << i << "]=" << moments[i] << endl;
    }
    throw InvalidValue(err_msg.str(),__FILE__,__LINE__);
  }
  
  //_____________
  //Solve actual weights and abscissas
  for( int i = 0; i < nEnv; i++) {
    w[i] = moments[0] * z_[i*nEnv] * z_[i*nEnv];
    x[i] = eigenVal[i];
  }
  delete work;
}


/***************************
 N-D Adaptive Wheeler alogrithm
 ****************************/
void adaptiveWheelerAlgorithm(const std::vector<double>& moments, std::vector<double>& w, std::vector<double>& x,
                              const double& rMin, const double& eAbs)
{
  int nEnvMax = moments.size()/2;
  int nEnvOut = moments.size()/2;
  int nMom = moments.size();
#ifdef cqmom_dbg
  cout << "Wheeler Adaptive Start" << endl;
  for (int i = 0; i < nMom; i++) {
    cout << "m[" << i << "]=" << moments[i] << endl;
  }
#endif
  bool isRealizable = false;
  
  while ( !isRealizable ) {
    if (nEnvOut == 1) {
      w[0] = moments[0];
      x[0] = moments[1]/moments[0];
      double d_small = 1.0e-10;
      if ( fabs(x[0]) < d_small) {
        //prevent very small values from propagating junk later
        x[0] = 0.0;
      }
#ifdef cqmom_dbg
      cout << "Singular Point" << endl;
#endif
      isRealizable = true;
      continue;
    }
  
    std::vector<double> a (nEnvOut, 0.0);
    std::vector<double> b (nEnvOut, 0.0);
    std::vector< std::vector<double> > sig (2*nEnvOut+1, std::vector<double> (2*nEnvOut+1,0.0) );
  
    for ( int i = 1; i<(2*nEnvOut+1); i++) {
      sig[1][i] = moments[i-1];
    }
    a[0] = moments[1]/moments[0];
  
    for (int k = 2; k<(nEnvOut+1); k++) {
      for (int j = k; j<(2*nEnvOut-k+2); j++) {
        sig[k][j] = sig[k-1][j+1]-a[k-2]*sig[k-1][j]-b[k-2]*sig[k-2][j];
      }
      a[k-1] = sig[k][k+1]/sig[k][k] - sig[k-1][k]/sig[k-1][k-1];
      b[k-1] = sig[k][k]/sig[k-1][k-1];
    }
  
#ifdef cqmom_dbg
    for (int i=0; i<nEnvOut; i++) {
      cout << "a[" << i << "] = " << a[i] <<endl;
    }
    for (int i=0; i<nEnvOut; i++) {
      cout << "b[" << i << "] = " << b[i] <<endl;
    }
    for (int i=0; i<2*nEnvOut+1; i++) {
      for (int j = 0; j<2*nEnvOut+1; j++) {
        cout << "S[" << i << "][" << j << "] = " << sig[i][j] << "  ";
      }
      cout << endl;
    }
#endif
  
    bool nonrealCheck = false;
    //check a vector for a nan - occurs in point distribution
    for ( int i = 0; i<nEnvOut; i++ ) {
      if ( isnan(a[i]) || isinf(a[i]) ) {
#ifdef cqmom_dbg
        cout << "WARNING: Arches: CQMOMInversion: not-a-number in a vector encountered. " << endl;
#endif
        nEnvOut--;
        nonrealCheck = true;
        break;
      }
    }
    
    if (nonrealCheck)
      continue;
    
    double d_small = 1.0e-14;
    //check the b vector for realizable space
    for ( int i = 0; i<nEnvOut; i++ ) {
      if ( (b[i] != 0.0 && b[i]<d_small) || isnan(b[i]) ) { //clip if b is very close to zero
#ifdef cqmom_dbg
        cout << "WARNING: Arches: CQMOMInversion: Negative b vector encountered." << endl;
#endif
        nEnvOut--;
        nonrealCheck = true;
        break;
      }
    }
    
    if (nonrealCheck)
      continue;
  
#ifdef cqmom_dbg
    std::vector< std::vector<double> > z (nEnvOut, std::vector<double> (nEnvOut, 0.0) );
    for ( int i = 0; i<(nEnvOut-1); i++) {
      z[i][i] = a[i];
      z[i][i+1] = sqrt( b[i+1] );
      z[i+1][i] = z[i][i+1];
    }
    z[nEnvOut-1][nEnvOut-1] = a[nEnvOut-1];
    cout << "made z matrix" << endl;
    for (int i=0; i<nEnvOut; i++) {
      for (int j = 0; j<nEnvOut; j++) {
        cout << "z[" << i << "][" << j << "] = " << z[i][j] << "  ";
      }
      cout << endl;
    }
#endif
  
    std::vector<double> z_(nEnvOut*nEnvOut,0.0);
    for (int i = 0; i<(nEnvOut-1); i++) {
      z_[i*nEnvOut + i] = a[i];
      z_[i*nEnvOut + i + 1] = sqrt( b[i+1] );
      z_[(i+1)*nEnvOut + i] = z_[i*nEnvOut + i + 1];
    }
    z_[nEnvOut*nEnvOut-1] = a[nEnvOut-1];
  
    std::vector<double> eigenVal (nEnvOut, 0.0);
    //_____________
    //solve eigenvalue problem with external package
    int  lda = nEnvOut, info, lwork;
    double wkopt;
    double* work;
    lwork = -1;
    char jobz='V';
    char matType = 'U';
    DSYEV( &jobz, &matType, &nEnvOut, &z_[0], &lda, &eigenVal[0], &wkopt, &lwork, &info ); //with -1 this finds work size
    lwork = (int)wkopt;
    work = new double[lwork];
    // Solve eigenproblem. eigenvectors are stored in the z_ matrix, columnwise
    DSYEV( &jobz, &matType, &nEnvOut, &z_[0], &lda, &eigenVal[0], work, &lwork, &info );
    bool status = ( info>0 || info<0 )? false : true;
    delete work;
  
    if (!status) {
#ifdef cqmom_dbg
      cout << "WARNING: Arches: CQMOMInversion: Solving Eigenvalue problem failed. Moment set: " << endl;
#endif
      nEnvOut--;
      continue;
    }
    //_____________
    //Solve actual weights and abscissas
    for( int i = 0; i < nEnvOut; i++) {
      w[i] = moments[0] * z_[i*nEnvOut] * z_[i*nEnvOut];
      x[i] = eigenVal[i];
    }
    
    //____________
    //Check that the minimum spacing and weight ratios are met
    std::vector<double> dab (nEnvOut);
    std::vector<double> mab (nEnvOut);
    double mindab, maxmab;
    
    double minTest;
    for (int i = nEnvOut-1; i>=1; i--) {
      dab[i] = fabs(x[i] - x[0]);
      for (int j = 1; j<i; j++) {
        minTest = fabs(x[i]-x[j]);
        if (minTest < dab[i]) {
          dab[i] = minTest;
        }
      }
    }
    double maxTest;
    for (int i = nEnvOut-1; i>=1; i--) {
      mab[i] = fabs(x[i] - x[0]);
      for (int j = 1; j<i; j++) {
        maxTest = fabs(x[i]-x[j]);
        if (maxTest > mab[i]) {
          mab[i] = maxTest;
        }
      }
    }
    
    mindab = dab[1]; maxmab = mab[1];
    for (int i=1; i<nEnvOut; i++) {
      if (dab[i] < mindab) {
        mindab = dab[i];
      }
      if (mab[i] > maxmab) {
        maxmab = mab[i];
      }
    }
    
    //check that prescribed condtions are met
    double maxW, minW;
    maxW = w[0];
    minW = w[0];
    for (int i = 0; i <nEnvOut; i++) {
      if (w[i] < minW) {
        minW = w[i];
      }
      if (w[i] > maxW) {
        maxW = w[i];
      }
    }
    if (minW/maxW > rMin && mindab/maxmab > eAbs) {
      isRealizable = true;
    } else {
      nEnvOut--;
#ifdef cqmom_dbg
      cout << "Weight Ratio: " << minW/maxW << " Abscissa Ratio: " << mindab/maxmab << endl;
      cout << "Decreasing environments " << nEnvOut+1 << " to " << nEnvOut << endl;
#endif
    }

  }
  //fill in any extra weight/abscissa values as zero
  if ( nEnvOut < nEnvMax ) {
    for (int i = nEnvOut; i < nEnvMax; i++ ) {
      w[i] = 0.0;
      x[i] = 0.0;
    }
  }
}

/***************************
 CQMOMInversion Function
 ****************************/
void CQMOMInversion( const std::vector<double>& moments, const int& M, const std::vector<int>& N_i, const std::vector<int>& maxInd,
                     std::vector<double> & weights, std::vector<std::vector<double> >& abscissas, const bool& adapt, const bool& useLapack,
                    const double& rMin, const double& eAbs)
{
  //This is the actual cqmom inversion fucntion
  //moments should be a a flat array numerically ordered
  //insert 0s for unused moments to keep spacing right
  //maxInd vector of maxium moment# for flat array
  //For the time being this is only for M = 2 or 3
  //NOTE: to-do: expand to M = 4,5...N etc
  

  int nTot = 1;  //this could probably be input
  for (int i=0; i<M; i++) {
    nTot *= N_i[i];
  }
  
#ifdef cqmom_dbg
  for (unsigned int i = 0; i<moments.size(); i++) {
    cout << "M[" << i << "]= " << moments[i] << endl;
  }
#endif
  
  std::vector<double> tempMom ( N_i[0]*2,0.0);
  for (unsigned int i = 0; i<tempMom.size(); i++ ) {
    tempMom[i] = moments[i];
  }
  std::vector<double> x1 ( N_i[0], 0.0 );
  std::vector<double> w1 ( N_i[0], 0.0 );

  if (!adapt) {
    wheelerAlgorithm( tempMom, w1, x1);
  } else {
    adaptiveWheelerAlgorithm( tempMom, w1, x1, rMin, eAbs);
  }
  
#ifdef cqmom_dbg
  cout << "Completed Wheeler Algorithm x1" << endl;
  for ( int i = 0 ; i<x1.size(); i++ ) {
    cout << "w[" << i << "] = " << w1[i] << ", " << "x1[" << i << "] = " << x1[i] << endl;
  }
#endif
  
  if (M == 1) {
    for (int k1 = 0; k1 < N_i[0]; k1++) {
      weights[k1] = w1[k1];
      abscissas[0][k1] = x1[k1];
    }
    return;
  }
  
  std::vector<double> vanderMom (N_i[0], 0.0);    //moment matrix to solve conditionals
  std::vector<double> condMomStar(N_i[0], 0.0);   //temporary conditional moments
  std::vector< std::vector<double> > condMom (N_i[0], std::vector<double> (2*N_i[1], 1.0) ); //initialized to 1.0 to handle all 0th moments
  //loop over each k_2 1,...,2N2-1
  // 0th moment for all conditional = 1
  
  for (int k = 1; k<2*N_i[1]; k++) {   //this loops each column vector of conditional moments
    for (int i=0; i<N_i[0]; i++) {
      vanderMom[i] = moments[i+ k*maxInd[0] ];  //ith,kth moment, with j = 0
    }
    
#ifdef cqmom_dbg
  cout << "Calculating Conditional Moments" << endl;
  for (int i = 0; i<N_i[0]; i++) {
    cout << "vanderMom[" << k << "][" << i << "] = " << vanderMom[i] << endl;
  }
#endif
    
    if ( !useLapack ) {
    //vander solve (x,w,q) -> vandersolve sum x_i^k w_i = q_k
    // q contians known moments, x_i contains abscissas, w_i are conditional moments (unknown)
      vandermondeSolve ( x1, condMomStar, vanderMom);
    } else {
      int dim = N_i[0];
      int nRHS = 1;
      int info;
      vector<int> ipiv(dim);
      vector<double> b (dim);
      vector<double> a (dim*dim);
    
      for (int i = 0; i<dim; i++) {
        b[i] = vanderMom[i];
      }
    
      for (int i = 0; i<dim; i++) {
        for (int j = 0; j<dim; j++) {
          a[j + dim*i] = pow(x1[i],j);
        }
      }
    
      DGESV(&dim, &nRHS, &*a.begin(), &dim, &*ipiv.begin(), &*b.begin(), &dim, &info);
    
      for (int i = 0; i<dim; i++) {
        condMomStar[i] = b[i];
      }
    }
    
    for (int i = 0; i<N_i[0]; i++) {
      if ( w1[i] > 0.0 ) {
        condMom[i][k] = condMomStar[i]/w1[i];
      } else {
        condMom[i][k] = 0.0;
      }
    }
#ifdef cqmom_dbg
    for (int i = 0; i<N_i[0]; i++) {
      cout << "condMom*[" << k << "][" << i << "] = " << condMomStar[i] << endl;
    }
    for (int i = 0; i<N_i[0]; i++) {
      cout << "condMom[" << k << "][" << i << "] = " << condMom[i][k] << endl;
    }
#endif
  }
  
  std::vector<double> tempX (N_i[1], 0.0);
  std::vector<double> tempW (N_i[1], 0.0);
  std::vector< std::vector<double> > x2 (N_i[0], std::vector<double> (N_i[1], 0.0) );
  std::vector< std::vector<double> > w2 (N_i[0], std::vector<double> (N_i[1], 0.0) );
  tempMom.resize(2*N_i[1]);
  
  for (int i = 0; i<N_i[0]; i++) {
    for (int k = 0; k < 2*N_i[1]; k++) {
      tempMom[k] = condMom[i][k];
    }
    if (!adapt) {
      wheelerAlgorithm( tempMom, tempW, tempX);
    } else {
      adaptiveWheelerAlgorithm( tempMom, tempW, tempX, rMin, eAbs);
    }
    
    for (int k = 0; k < N_i[1]; k++) {
      x2[i][k] = tempX[k];
      w2[i][k] = tempW[k];
    }
  }
  
#ifdef cqmom_dbg
  cout << "2D Quadrature Nodes" << endl;
  cout << "________________" << endl;
  for (int i = 0 ; i<N_i[0]; i++) {
    cout << "x1[" << i << "] = " << x1[i] << " ";
    cout << "w1[" << i << "] = " << w1[i] << endl;
    for (int k = 0; k<N_i[1]; k++ ) {
      cout << "x2[" << i << "][" << k << "] = " << x2[i][k] << " ";
      cout << "w2[" << i << "][" << k << "] = " << w2[i][k] << endl;
    }
  }
#endif
  
  int ii = 0;
  if (M == 2) {
    //terminate the fucntion with if statement for now
    //NOTE: when this is expanded a i = 1...M loop should be around the algorithm
    
    //actually populate the weights/abscissas vector
    for (int k1 = 0; k1 < N_i[0]; k1++) {
      for (int k2 = 0; k2 < N_i[1]; k2++) {
        weights[ii] = w1[k1]*w2[k1][k2];
        abscissas[0][ii] = x1[k1];
        abscissas[1][ii] = x2[k1][k2];
        ii++;
      }
    }
    return;
  }
  
  //start the 3D quadrature method
  vanderMom.resize(N_i[0]);
  std::vector< std::vector< std::vector<double> > > Zeta (N_i[0], std::vector<std::vector<double> > (N_i[1], std::vector<double> (2*N_i[2],0.0)) );
  std::vector<double> tempStar (N_i[0],0.0);

  for (int k3 = 1; k3<2*N_i[2]; k3++) {
    for (int k2 = 0; k2<N_i[1]; k2++) { //loop thorugh all combinations of k_2/k_3 for zeta values
      // solve V_1R_1 * zeta^(k1,k2) = m for zeta 1 to N1
      //populate the moment vector
      for (int i = 0; i<N_i[0]; i++) {
        vanderMom[i] = moments[i + k2*maxInd[0] + k3*maxInd[0]*maxInd[1]];
#ifdef cqmom_dbg
        if ( k3 == 1 ) {
          cout << "RHS[" << i << "] = " << vanderMom[i] << endl;
        }
#endif
      }
      
      if ( !useLapack ) {
      //feed into vandermonde
        vandermondeSolve(x1, tempStar, vanderMom);
      } else {
        int dim = N_i[0];
        int nRHS = 1;
        int info;
        vector<int> ipiv(dim);
        vector<double> b (dim);
        vector<double> a (dim*dim);
      
        for (int i = 0; i<dim; i++) {
          b[i] = vanderMom[i];
        }
      
        for (int i = 0; i<dim; i++) {
          for (int j = 0; j<dim; j++) {
            a[j + dim*i] = pow(x1[i],j);
          }
        }
      
        DGESV(&dim, &nRHS, &*a.begin(), &dim, &*ipiv.begin(), &*b.begin(), &dim, &info);
      
        for (int i = 0; i<dim; i++) {
          tempStar[i] = b[i];
        }
      }
#ifdef cqmom_dbg
      if (k3 == 1) {
        for (int i = 0; i<N_i[0]; i++) {
          cout << "x[" << i << "] = " << x1[i] << endl;
          cout << "temp[" << i << "] = " << tempStar[i] << endl;
          cout << "vmom[" << i << "] = " << vanderMom[i] << endl;
        }
      }
#endif
      
      for (int i = 0; i<N_i[0]; i++) {
        if ( w1[i] > 0.0 ) {
          Zeta[i][k2][k3] = tempStar[i]/w1[i];
        } else {
          Zeta[i][k2][k3] = 0.0;
        }
#ifdef cqmom_dbg
        if ( k3 == 1 ) {
          cout << "Zeta[" << i << "] = " << Zeta[i][k2][k3] << endl;
        }
#endif
      }
    }
  }
  
  std::vector< std::vector<double> > condMom3 (N_i[1], std::vector<double> (2*N_i[2], 1.0) );
  std::vector<double> xTemp (N_i[1], 0.0);
  std::vector<double> wTemp (N_i[1], 0.0);
  std::vector<std::vector< std::vector<double> > > x3 (N_i[0], std::vector<std::vector<double> > (N_i[1], std::vector<double> (N_i[2],0.0)) );
  std::vector<std::vector< std::vector<double> > > w3 (N_i[0], std::vector<std::vector<double> > (N_i[1], std::vector<double> (N_i[2],0.0)) );
  
  condMomStar.resize(N_i[1]);
  vanderMom.resize(N_i[1]);
  
  //feed these zeta to solve for 3rd conditional moments
  for (int i = 0; i<N_i[0]; i++) {
    // make V_2, R_2 for each alpha_1 node of x1
    for (int j = 0; j<N_i[1]; j++) {
      xTemp[j] = x2[i][j];
      wTemp[j] = w2[i][j];
    }
    
    for (int k = 1; k<2*N_i[2]; k++) {
      for (int j = 0; j<N_i[1]; j++) {     //populate "moments" vector with zetas
        vanderMom[j] = Zeta[i][j][k];
#ifdef cqmom_dbg
        if ( k == 1 ) {
          cout << "rhs[" << j << "] = " << Zeta[i][j][k] << endl;
          cout << "rhs[" << j << "] = " << vanderMom[j]<< endl;
          cout << "x[" << j << "] = " << xTemp[j]<< endl;
        }
#endif
      }
      
      if ( !useLapack ) {
        vandermondeSolve(xTemp, condMomStar, vanderMom );
      } else {
        int dim = N_i[0];
        int nRHS = 1;
        int info;
        vector<int> ipiv(dim);
        vector<double> b (dim);
        vector<double> a (dim*dim);
      
        for (int i = 0; i<dim; i++) {
          b[i] = vanderMom[i];
        }
      
        for (int i = 0; i<dim; i++) {
          for (int j = 0; j<dim; j++) {
            a[j + dim*i] = pow(xTemp[i],j);
          }
        }
      
        DGESV(&dim, &nRHS, &*a.begin(), &dim, &*ipiv.begin(), &*b.begin(), &dim, &info);
      
        for (int i = 0; i<dim; i++) {
          condMomStar[i] = b[i];
        }
      }
      
      for (int j = 0; j<N_i[1]; j++) {
        if ( wTemp[j] > 0.0 ) {
          condMom3[j][k] = condMomStar[j]/wTemp[j]; //un-normalize the weights
        } else {
          condMom3[j][k] = 0.0;
        }
#ifdef cqmom_dbg
        if ( k == 1 ) {
          cout << "cmom*[" << j << "] = " << condMomStar[j] << endl;
          cout << "cmom[" << j << "] = " << condMom3[j][k] << endl;
        }
#endif
      }
      //now all conditional moments for node i are known
    }
    
    //solve the wheeler algorithm for each row of conditonal moment vector
    //here each row corresponds to node 2 jth
    tempMom.resize(N_i[2]*2); tempW.resize(N_i[2]); tempX.resize(N_i[2]);
    for (int j = 0; j<N_i[1]; j++) {
      for (int k = 0; k<2*N_i[2]; k++) {
        tempMom[k] = condMom3[j][k];
      }
      
      if (!adapt) {
        wheelerAlgorithm( tempMom, tempW, tempX);
      } else {
        adaptiveWheelerAlgorithm( tempMom, tempW, tempX, rMin, eAbs);
      }
      
      for (int k = 0; k<N_i[2]; k++) {
        x3[i][j][k] = tempX[k];
        w3[i][j][k] = tempW[k];
      }
    }
  }
  
#ifdef cqmom_dbg
  cout << "Final Quad Nodes" << endl;
  cout << "________________" << endl;
  for (int i = 0 ; i<N_i[0]; i++) {
    cout << "x1[" << i << "] = " << x1[i] << " ";
    cout << "w1[" << i << "] = " << w1[i] << endl;
    for (int j = 0; j<N_i[1]; j++ ) {
      cout << "x2[" << i << "][" << j << "] = " << x2[i][j] << " ";
      cout << "w2[" << i << "][" << j << "] = " << w2[i][j] << endl;
      for (int k = 0; k<N_i[2]; k++) {
        cout << "x3[" << i << "][" << j << "][" << k << "] = " << x3[i][j][k] << " ";
        cout << "w3[" << i << "][" << j << "][" << k << "] = " << w3[i][j][k] << " " << "w_n = " << w1[i]*w2[i][j]*w3[i][j][k] << endl;
        
      }
    }
  }
#endif
  
  //now populate the weights and abscissas vectors
  ii = 0;
  for (int k1 = 0; k1 < N_i[0]; k1++) {
    for (int k2 = 0; k2 < N_i[1]; k2++) {
      for (int k3 = 0; k3 < N_i[2]; k3++) {
        weights[ii] = w1[k1]*w2[k1][k2]*w3[k1][k2][k3];
        abscissas[0][ii] = x1[k1];
        abscissas[1][ii] = x2[k1][k2];
        abscissas[2][ii] = x3[k1][k2][k3];
        ii++;
      }
    }
  }
  
} //end CQMOMInversion

#endif

