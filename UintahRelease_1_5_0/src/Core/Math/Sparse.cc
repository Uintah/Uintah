/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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

#include <Core/Math/Sparse.h>
#include <Core/Math/Rand48.h>
#include <iostream>

using namespace std;
using namespace Uintah;


namespace Uintah {

valarray<double> cgSolve(SparseMatrix<double, int>& A, 
                         valarray<double>& b, int /*conflag*/)
{
  
  valarray<double> M(A.Rows());
  valarray<double>  z(A.Rows()),p(A.Rows());

  //  Use the Jacobian preconditioner,
  //  defined M(i,j) = A(i,j) for i=j else 0, which is just
  //  the diagonal of the A matrix.  But we take the inverse of
  //  it since we are gong to be solving for z in Mz = r.
  //
  
  for (int i = 0; i< A.Rows(); i++ ) {
    M[i] = 1./A[i][i];
  }


  // Make an initial guess for the solution, x, -- guess all ones.
  // valarrays initialize things to zero.
 
  valarray<double> x(0.,A.Rows()),residual(A.Rows());

  // compute the norm of b

//  double max = b.max();
//  cout << "max = " << max << endl;

  // Compute r^(0) = b - A x^(0) for some initial guess x^(0)

  residual = b - A*x;

  // Now start the procedure

  int max_iterations = 5000;
  double rho_i_1 = 0.;
  double rho_i_2 = 0.;


  for( int i = 1; i <= max_iterations; i++ ) {
    // Solve for Mz = r
    z = M*residual;

    // do the dot product of the residual and the z vector
    rho_i_1 = inner_product(&residual[0],&residual[residual.size()],&z[0],0.);

    if (i == 1) {
      p = z;
    }
    else {
      double beta = rho_i_1/rho_i_2;
      p = z + p*beta;
    }

    valarray<double> q(A.Rows());
    q = A*p;

    double temp = inner_product(&p[0],&p[p.size()],&q[0],0.);
    double alpha = rho_i_1/temp;
    
    x += alpha*p;

    residual -= alpha*q;

    // Check convergence
    valarray<double> res(residual.size());
    res = abs(residual);

    double t = accumulate(&res[0],&res[res.size()],0.);

    if (t <= 1.e-8) {
      cout << "number of iterations is " << i << endl;
      cout << "residual is " << t << endl;
      break;
    }

    if (i%1000 == 0){
      cout << "Iteration " << i << ", residual is " << t << endl;
    }

    if (i == max_iterations) {
      cout << "No convergence" << endl;
    }
    
    rho_i_2= rho_i_1;
    rho_i_1 = 0.;
  }

  return x;
    
}


double eigenvalue(SparseMatrix<double,int>& A, valarray<double>& eigenvector)
{

  //  This computes the eigenvalue for the matrix p_M.
  //  It is the largest unless there is a shift.  The
  //  eigenvector is used in the computation via the
  //   power method.
        
  
  valarray<double> y_new(A.Rows());

  y_new = A*eigenvector;

  // Normalize the vector

  double mag = inner_product(&y_new[0],&y_new[y_new.size()],&y_new[0],0.);
  mag = sqrt(mag);
  eigenvector = y_new/mag;

  y_new = 0.;
  
  y_new = A*eigenvector;

  double l_eigen = 0.;

  l_eigen = 
    inner_product(&eigenvector[0],&eigenvector[eigenvector.size()],&y_new[0],0.);

  return l_eigen;
  
}

    
double conditionNum(SparseMatrix<double,int>& A)
{
  // Determines the condition number of a matrix using the power method.
  // The condition number is defined to be
  // the largest eigenvalue divided by the smallest eigenvalue.
     
  

  valarray<double> y(A.Rows());
  y = drand48();
  y[0] = 1.;

  double l_eigen = Uintah::eigenvalue(A,y);
  double old_eigen;

  int k = 1;
  do {
    old_eigen = l_eigen;
    l_eigen = Uintah::eigenvalue(A,y);
    k++;
  } while (fabs(l_eigen-old_eigen) > 1e-8);

  cout << "Number of iteration: " << k << " large eigenvalue: " << l_eigen
       << endl;

  // Compute the smallst eigenvalue by shifting from the largest
  // do (A - shift I) x = (lambda - shift) x

  double shift = l_eigen;
  
  SparseMatrix<double,int> S(A);

  
  for (int i = 0; i<S.Rows(); i++) 
    S[i][i] = A[i][i] - shift;

  
  // input starting eigenvector

  y = .5;
  y[0] = 1.;
  

  // Compute the first time thru the routine

  double s_eigen = Uintah::eigenvalue(S,y);

  // end of pass

  k = 1;
  do {
    old_eigen = s_eigen;
    s_eigen = Uintah::eigenvalue(S,y);
    k++;
  } while (fabs(s_eigen - old_eigen) > 1e-10);

  s_eigen += shift;

  cout << "Number of iterations: " << k << " small eigenvalue: " 
       << s_eigen << endl;

  double cn = l_eigen/s_eigen;
  return cn;


}


}
