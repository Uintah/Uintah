/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

//  class Matrix3
//    Matrix3 data type -- holds components of a 3X3 matrix
//    Features:
//      1.  Nearly, all the overloaded operations of the Matrix class,
//      no dynamic memory allocation.

#include <Core/Math/Matrix3.h>
#include <Core/Math/CubeRoot.h>
#include <Core/Disclosure/TypeDescription.h>
#include <Core/Util/TypeDescription.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/Assert.h>
#include <Core/Util/Endian.h>
#include <Core/Math/MinMax.h>
#include "./TntJama/tnt.h"
#include "./TntJama/jama_eig.h"

#include <cstdlib>

#include <iostream>
#include <fstream>

#ifdef _WIN32
#define copysign _copysign
#endif

using namespace Uintah;

using std::cerr;
using std::endl;
using std::ostream;
using SCIRun::Vector;

const string& 
Matrix3::get_h_file_path() {
  static const string path(SCIRun::TypeDescription::cc_to_h(__FILE__));
  return path;
}

// Added for compatibility with Core types
namespace SCIRun {

  using std::string;

  template<> const string find_type_name(Matrix3*)
  {
    static const string name = "Matrix3";
    return name;
  }

  const TypeDescription* get_type_description(Matrix3*)
  {
    static TypeDescription* td = 0;
    if(!td){
      td = scinew TypeDescription("Matrix3", Matrix3::get_h_file_path(), "Uintah");
    }
    return td;
  }

  void
  Pio(Piostream& stream, Matrix3& mat)
  {
    stream.begin_cheap_delim();
    Pio(stream, mat(0,0)); Pio(stream, mat(0,1)); Pio(stream, mat(0,2));
    Pio(stream, mat(1,0)); Pio(stream, mat(1,1)); Pio(stream, mat(1,2));
    Pio(stream, mat(2,0)); Pio(stream, mat(2,1)); Pio(stream, mat(2,2));
    stream.end_cheap_delim();
  }

  // needed for bigEndian/littleEndian conversion
  void swapbytes( Uintah::Matrix3& m){
    double *p = (double *)(&m);
    SWAP_8(*p); SWAP_8(*++p); SWAP_8(*++p);
    SWAP_8(*++p); SWAP_8(*++p); SWAP_8(*++p);
    SWAP_8(*++p); SWAP_8(*++p); SWAP_8(*++p);
  }

} // namespace SCIRun



// Anything with absolute value < NEAR_ZERO may be considered
// zero, assuming error is caused by round-off.
#define NEAR_ZERO 1e-10

void Matrix3::set(int i, int j, double value)
{
  // Assign the Matrix3 the value components
  mat3[i][j] = value;
}

Matrix3 Matrix3::Inverse() const
{
  // Return the inverse of a 3x3 matrix
  // This looks ugly but it works -- just for 3x3

  double det;
  Matrix3 inv_matrix(0.0);

  det = this->Determinant();
  if ( det == 0.0 )
    {
      cerr << "Singular matrix in matrix inverse..." << endl;
      exit(1);
    }
  else
    {
      inv_matrix(0,0) = (*this)(1,1)*(*this)(2,2) - (*this)(1,2)*(*this)(2,1);
      inv_matrix(0,1) = -(*this)(0,1)*(*this)(2,2) + (*this)(2,1)*(*this)(0,2);
      inv_matrix(0,2) = (*this)(0,1)*(*this)(1,2) - (*this)(1,1)*(*this)(0,2);
      inv_matrix(1,0) = -(*this)(1,0)*(*this)(2,2) + (*this)(2,0)*(*this)(1,2);
      inv_matrix(1,1) = (*this)(0,0)*(*this)(2,2) - (*this)(0,2)*(*this)(2,0);
      inv_matrix(1,2) = -(*this)(0,0)*(*this)(1,2) + (*this)(1,0)*(*this)(0,2);
      inv_matrix(2,0) = (*this)(1,0)*(*this)(2,1) - (*this)(2,0)*(*this)(1,1);
      inv_matrix(2,1) = -(*this)(0,0)*(*this)(2,1) + (*this)(2,0)*(*this)(0,1);
      inv_matrix(2,2) = (*this)(0,0)*(*this)(1,1) - (*this)(0,1)*(*this)(1,0);
 
      inv_matrix = inv_matrix/det;

    }
  return inv_matrix;

} //end Inverse()

// A recursive Taylor series expansion (USE WITH CARE)
// **WARNING** Expansion may not be convergent in which case use
// eigenvalue expansion (not implemented)
// Based on Ortiz, Radovitzsky, Repetto (2001)
Matrix3 Matrix3::Exponential(int num_terms) const
{
  Matrix3 exp(0.0);

  // The k = 0 term
  Matrix3 exp_k_term; exp_k_term.Identity();
  exp += exp_k_term;

  for (int kk = 0; kk < num_terms; ++kk) {
    exp_k_term = (exp_k_term*(*this))*(1.0/(double)(kk+1));
    exp += exp_k_term;
  }
  return exp;
}

// A recursive Taylor series expansion (USE WITH CARE)
// **WARNING** Expansion may not be convergent in which case use
// eigenvalue expansion (not implemented)
// Based on Ortiz, Radovitzsky, Repetto (2001)
Matrix3 Matrix3::Logarithm(int num_terms) const
{
  Matrix3 log(0.0); 
  Matrix3 One; One.Identity();

  // The k = 0 term
  Matrix3 log_0_term(0.0), log_k_term(0.0); 
  log_0_term = *this - One;
  log_k_term = log_0_term;
  log += log_k_term;

  for (int ii = 1; ii <= num_terms; ++ii) {
    log_k_term = (log_k_term*log_0_term)*((double)ii/(double)(ii+1));
    log += log_k_term;
  }
  return log;
}

inline void swap(double& v1, double& v2)
{
  double tmp = v1;
  v1 = v2;
  v2 = tmp;
}

// Reduce the matrix and rhs, representing the equation system:
// A*x = y = rhs, to a matrix in upper triangular form with
// corresponding rhs for an equivalent equation systme.
// The guarantee for this new matrix is that the first non-zero
// column of a row is to the right (greater) of the first non-zero
// column in the row above it and the first non-zero column of
// any row has zeroes in every other row and a one in that row.
// If rhs == NULL, then the rhs is assumed to be
// the zero vector and thus will not need to change.
void Matrix3::triangularReduce(Matrix3& A, SCIRun::Vector* rhs, int& num_zero_rows,
                               double relative_scale)
{
  int i, j, k, pivot;
  int pivoting_row = 0;
  double tmp, mag;
  double relNearZero = relative_scale * NEAR_ZERO;

  // put the matrix in upper triangular form 
  for (i = 0; i < 3; i++) {
    // find pivot row (with greatest absolute ith column value)
    pivot = -1;
    // because of precision errors, consider anything smaller
    // than NEAR_ZERO as zero
    mag = relNearZero;
    for (j = pivoting_row; j < 3; j++) {
      tmp = fabs(A.mat3[j][i]);
      if (tmp > mag) {
        mag = tmp;
        pivot = j;
      }
    }

    if (pivot == -1) {
      for (j = pivoting_row; j < 3; j++)
        // < NEAR_ZERO absolute value is treated as zero, so set it to zero
        A.mat3[j][i] = 0;
      continue;
    }

    // swap rows and normalize
    double norm_multiplier = 1 / A.mat3[pivot][i];
    if (pivot != pivoting_row) {
      A.mat3[pivot][i] = A.mat3[pivoting_row][i];
      A.mat3[pivoting_row][i] = 1; // normalized    
      for (j = i+1 /* zeroes don't need to be swapped */; j < 3; j++) {
        tmp = A.mat3[pivot][j];
        A.mat3[pivot][j] = A.mat3[pivoting_row][j];
        A.mat3[pivoting_row][j] = tmp * norm_multiplier; // normalize
      }
      
      if (rhs != NULL) {
        // swap and normalize rhs in the same manner
        tmp = (*rhs)[pivot];
        (*rhs)[pivot] = (*rhs)[pivoting_row];
        (*rhs)[pivoting_row] = tmp * norm_multiplier; // normalizing
      }
    }
    else {
      // just normalize
      A.mat3[pivoting_row][i] = 1; 
      for (j = i+1 /* zeroes don't need to be normalized */; j < 3; j++)
        A.mat3[pivoting_row][j] *= norm_multiplier;
      if (rhs != NULL)
        (*rhs)[pivoting_row] *= norm_multiplier; // normalizing
    }

    // eliminate ith column from other rows via row reduction
    for (k = 0; k < 3; k++) {
      if (k == pivoting_row) continue;
      // row reduction (ignoring zeroed out columns of pivoting_row
      // (note that the pivoting row has been normalized)
      double mult = A.mat3[k][i]; // remember that pivoting_row is normalized
      for (j = i + 1; j < 3; j++)
        A.mat3[k][j] -= mult * A.mat3[pivoting_row][j]; 
      if (rhs != NULL)
        (*rhs)[k] -= mult * (*rhs)[pivoting_row];
      A.mat3[k][i] = 0;
    }

    pivoting_row++;
  }

  if (rhs != NULL) {
    for (i = 0; i < 3; i++)
      if (fabs((*rhs)[i]) <= relNearZero)
        (*rhs)[i] = 0; // set near zero's to zero to compensate for round-off
  }
  
  num_zero_rows = 3 - pivoting_row;
}

// solveHomogenous for a Matrix that has already by triangularReduced
// So the matrix is in one of the following forms:
// {{1, 0, 0} {0, 1, 0} {0, 0, 1}}, single solution
// {{1, 0, c} {0, 1, e} {0, 0, 0}}, line solution
// {{1, b, 0} {0, 0, 1} {0, 0, 0}}, line solution
// {{0, 1, 0} {0, 0, 1} {0, 0, 0}}, line solution
// {{1, b, c} {0, 0, 0} {0, 0, 0}}, plane solution 
// {{0, 1, c} {0, 0, 0} {0, 0, 0}}, plane solution 
// {{0, 0, 1} {0, 0, 0} {0, 0, 0}}, plane solution 
// {{0, 0, 0} {0, 0, 0} {0, 0, 0}} -> solution only iff rhs = {0, 0, 0} 
bool Matrix3::solveParticularReduced(const SCIRun::Vector& rhs, SCIRun::Vector& xp,
                                     int num_zero_rows) const
{
  double x, y, z;
  switch (num_zero_rows) {
  case 0:
    // x = rhs[0], y = rhs[1], z = rhs[2]
    xp = rhs;
    return true; // has solution
    
  case 1:
    // {{a, b, c} {0, d, e} {0, 0, 0}}, line solution
    if (rhs[2] != 0) // 0*x + 0*y + 0*z = rhs.z
      return false;
    
    if (mat3[1][1] == 0) {
      // {{1, b, 0} {0, 0, 1} {0, 0, 0}} or {{0, 1, 0} {0, 0, 1} {0, 0, 0}}

      // z = rhs[1]
      z = rhs[1];

      if (mat3[0][0] == 0) {
        // {{0, 1, 0} {0, 0, 1} {0, 0, 0}}
        y = rhs[0]; // y = rhs[0]
        x = 0; // arbitrary choice -- for simplicity
      }
      else {
        // {{1, b, 0} {0, 0, 1} {0, 0, 0}}
        y = 0; // arbitrary choice -- for simplicity
        x = rhs[0]; // x + b*(y=0) = rhs[0] -> x = rhs[0]
      }
    }
    else {
      // {{1, 0, c} {0, 1, e} {0, 0, 0}}, a,d nonzero -> line solution

      z = 0;  // since z can be anything, use z = 0 for simplicity
      y = rhs[1]; // y + e*(z=0) = rhs[1] -> y = rhs[1]
      x = rhs[0]; // x + c*(z=0) = rhs[0] -> x = rhs[0]
    }

    xp = SCIRun::Vector(x, y, z);
    return true;

  case 2:
    // {{1, b, c} {0, 0, 0} {0, 0, 0}} or
    // {{0, 1, c} {0, 0, 0} {0, 0, 0}} or
    // {{0, 0, 1} {0, 0, 0} {0, 0, 0}} or
    if ( rhs[1] != 0 || rhs[2] != 0) 
      return false; // 0*x + 0*y + 0*z = rhs.y = rhs.z
    
    // find the first non-zero element in row 0
    int i;
    for (i = 0; i < 2; i++)
      if (mat3[0][i] != 0) break;

    // make xp(i) (first non-zero column) non-zero and
    // the other two 0 because that is perfectly valid and
    // for simplicity.
    xp = SCIRun::Vector(0, 0, 0);
    xp[i] = rhs[0];

    return true;
  case 3:
    // solution only if rhs == 0 (in which case everywhere is a solution)
    if (rhs == SCIRun::Vector(0, 0, 0)) {
      xp = SCIRun::Vector(0, 0, 0); // arbitrarily choose {0, 0, 0}
      return true;
    }
    else
      return false;
  default:
    ASSERTFAIL("unexpected num_zero_rows in Matrix3::solveParticularReduced");
  }
}
 
// solveHomogenous for a Matrix that has already by triangularReduced.
std::vector<SCIRun::Vector> Matrix3::solveHomogenousReduced(int num_zero_rows) const
{
  std::vector<SCIRun::Vector> basis_vectors;

  basis_vectors.resize(num_zero_rows);
  
  switch (num_zero_rows) {
  case 3:
    // Solutions everywhere : A = 0 matrix
    basis_vectors[0] = SCIRun::Vector(1, 0, 0);
    basis_vectors[1] = SCIRun::Vector(0, 1, 0);
    basis_vectors[2] = SCIRun::Vector(0, 0, 1);
    break;

  case 1:
    {
      // line of solutions : A = {{a, b, c} {0, d, e} {0, 0, 0}}
      // do backwards substition, using value of 1 arbitrarily
      // if a variable is not constrained

      SCIRun::Vector v;
      if (mat3[1][1] == 0) {
        // A = {{a, b, c} {0, 0, 1} {0, 0, 0}
        v[2] = 0; // e*z = 0, e != 0 -> z = 0
        if (mat3[0][0] == 0) {
          // A = {{0, 1, c} {0, 0, 1} {0, 0, 0}
          // y + c*z = 0 -> y = 0
          v[1] = 0; // y = 0;
          v[0] = 1; // x is arbitrary (non-zero)
        }
        else {
          // A = {{1, b, c} {0, 0, 1} {0, 0, 0}
          v[1] = 1; // y is arbitrary (non-zer0)
          // x + b*y + c*0 = 0 -> x + b*y = 0 -> x = -b*y = -b
          v[0] = -mat3[0][1];
        }
      }
      else {
        // A = {{1, b, c} {0, 1, e} {0, 0, 0}
        v[2] = 1; // z is arbitrary (non-zero)

        // y + e*z = 0 -> y = -e*z = -e
        v[1] = -mat3[1][2];

        //  x + b*y + c*z -> x = -(b*y + c*z) = -(b*y + c) 
        v[0] = -(mat3[0][1] * v[1] + mat3[0][2]);
      }
      basis_vectors[0] = v;
      break;
    }

  case 2:
    // plane of solutions : A = {{a, b, c} {0, 0, 0} {0, 0, 0}}
    // ax + by + cz = 0
    // 3 line equations by taking
    // x=0 : line1 := {0, c, -b} * t
    // y=0 : line2 := {c, 0, -a} * t
    // z=0 : line3 := {b, -a, 0} * t
    // choose two that are unequal an non-zero

    // find the largest absolute value between a, b, c to choose
    // to two lines to use (assuming larger values are better,
    // i.e. non-zero for sure)
    int max_index = 0;
    double max = fabs(mat3[0][0]);
    if (fabs(mat3[0][1]) > max)
      max_index = 1;
    if (fabs(mat3[0][2]) > max)
      max_index = 2;

    if (max_index == 0) {
      // |a| largest, use line2 and line3
      basis_vectors[0] = SCIRun::Vector(mat3[0][2], 0, -mat3[0][0]); // {c, 0, -a}
      basis_vectors[1] = SCIRun::Vector(mat3[0][1], -mat3[0][0], 0); // {b, -a, 0}
    }
    else if (max_index == 1) {
      // |b| largest, use line1 and line3
      basis_vectors[0] = SCIRun::Vector(0, mat3[0][2], -mat3[0][1]); // {0, c, -b}
      basis_vectors[1] = SCIRun::Vector(mat3[0][1], -mat3[0][0], 0); // {b, -a, 0}
    }
    else {
      // |c| largest, use line1 and line2
      basis_vectors[0] = SCIRun::Vector(0, mat3[0][2], -mat3[0][1]); // {0, c, -b}
      basis_vectors[1] = SCIRun::Vector(mat3[0][2], 0, -mat3[0][0]); // {c, 0, -a}
    }
  }

  return basis_vectors;
}

void Matrix3::polarDecompositionRMB(Matrix3& U,
                                    Matrix3& R) const
{
  Matrix3 F = *this;

  // Get the rotation
  F.polarRotationRMB(R);

  // Stretch: U=R^T*F
  U=R.Transpose()*F;
}

void Matrix3::polarRotationRMB(Matrix3& R) const
{

//     PURPOSE: This routine computes the tensor [R] from the polar
//     decompositon (a special case of singular value decomposition),
//
//            F = RU = VR
//
//     for a 3x3 invertible matrix F. Here, R is an orthogonal matrix
//     and U and V are symmetric positive definite matrices.
//
//     This routine determines only [R].
//     After calling this routine, you can obtain [U] or [V] by
//       [U] = [R]^T [F]          and        [V] = [F] [R]^T
//     where "^T" denotes the transpose.
//
//     This routine is limited to 3x3 matrices specifically to
//     optimize its performance. You will notice below that all matrix
//     operations are written out long-hand because explicit
//     (no do-loop) programming avoids concerns about some compilers
//     not optimizing loop and other array operations as well as they
//     could for small arrays.
//
//     This routine returns a proper rotation if det[F]>0,
//     or an improper orthogonal tensor if det[F]<0.
//
//     This routine uses an iterative algorithm, but the iterations are
//     continued until the error is minimized relative to machine
//     precision. Therefore, this algorithm should be as accurate as
//     any purely analytical method. In fact, this algorithm has been
//     demonstrated to be MORE accurate than analytical solutions because
//     it is less vulnerable to round-off errors.
//
//     Reference for scaling method:
//     Brannon, R.M. (2004) "Rotation and Reflection" Sandia National
//     Laboratories Technical Report SAND2004-XXXX (in process).
//
//     Reference for fixed point iterator:
//     Bjorck, A. and Bowie, C. (1971) "An iterative algorithm for
//     computing the best estimate of an orthogonal matrix." SIAM J.
//     Numer. Anal., vol 8, pp. 358-364.

// input
// -----
//    F: the 3x3 matrix to be decomposed into the form F=RU
//
// output
// -----
//    R: polar rotation tensor (3x3 orthogonal matrix)

  Matrix3 F = *this;
  Matrix3 I;
  I.Identity();
  double det = F.Determinant();
  if ( det <= 0.0 ) {
    cerr << "Singular matrix in polar decomposition..." << endl;
    cerr << "F = " << F << endl;
    cerr << "det = " << det << endl;
    exit(1);
  }

//Step 1: Compute [C] = Transpose[F] . [F] (Save into E for now)
  Matrix3 E = F.Transpose()*F;


// Step 2: To guarantee convergence, scale [F] by multiplying it by
//         Sqrt[3]/magnitude[F]. This is allowable because this routine
//         finds ONLY the rotation tensor [R]. The rotation for any
//         positive multiple of [F] is the same as the rotation for [F]
//         Scaling [F] by a factor sqrt(3)/mag[F] requires replacing the
//         previously computed [C] matrix by a factor 3/squareMag[F],
//         where squareMag[F] is most efficiently computed by trace[C].
//         Complete computation of [E]=(1/2)([C]-[I]) with [C] now
//         being scaled.
  double S=3.0/E.Trace();
  E=(E*S-I)*0.5;

// Step 3: Replace S with Sqrt(S) and set the first guess for [R] equal
//         to the scaled [F] matrix,   [A]=Sqrt[3]F/magnitude[F]

  S=sqrt(S);
  Matrix3 A=F*S;

// Step 4. Compute error of this first guess.
//     The matrix [A] equals the rotation if and only if [E] equals [0]
  double ERRZ = E(0,0)*E(0,0) + E(1,1)*E(1,1) + E(2,2)*E(2,2) 
         + 2.0*(E(0,1)*E(0,1) + E(1,2)*E(1,2) + E(2,0)*E(2,0));

// Step 5.  Check if scaling ALONE was sufficient to get the rotation.
//     This occurs whenever the stretch tensor is isotropic.
//     A number X is zero to machine precision if (X+1.0)-1.0 evaluates
//     to zero. Typically, machine precision is around 1.e-16.
  bool converged=false;

  if(ERRZ+1.0 == 1.0){
    converged=true;
  }

  Matrix3 X;
  int num_iters=0;
  while(converged==false){

// Step 6: Improve the solution.
//
//     Compute a helper matrix, [X]=[A][I-E].
//     Do *not* overwrite [A] at this point.
    X=A*(I-E);

    A=X;

// Step 7: Using the improved solution compute the new
//         error tensor, [E] = (1/2) (Transpose[A].[A]-[I])

    E = (A.Transpose()*A-I)*.5;

// Step 8: compute new error
    double ERR  = E(0,0)*E(0,0) + E(1,1)*E(1,1) + E(2,2)*E(2,2) 
           + 2.0*(E(0,1)*E(0,1) + E(1,2)*E(1,2) + E(2,0)*E(2,0));

// Step 9:
// If new error is smaller than old error, then keep on iterating.
// If new error equals or exceeds old error, we have reached
// machine precision accuracy.
    if(ERR>=ERRZ || ERR+1.0 == 1.0){
      converged = true;
    }
    double old_ERRZ=ERRZ;
    ERRZ=ERR;

    if(num_iters==200){
      cerr.precision(15);
      cerr << "Matrix3::polarRotationRMB not converging with Matrix:" << endl;
      cerr << F << endl;
      cerr << "ERR = " << ERR << endl;
      cerr << "ERRZ = " << old_ERRZ << endl;
      exit(1);
    }
    num_iters++;

  }  // end while

// Step 10:
// Load converged rotation into R;
   R=A;
}

void Matrix3::polarDecompositionAFFinvTran(Matrix3& U,
                                           Matrix3& R) const
{
  Matrix3 F = *this;

  // Get the rotation
  F.polarRotationAFFinvTran(R);

  // Stretch: U=R^T*F
  U=R.Transpose()*F;
}

void Matrix3::polarRotationAFFinvTran(Matrix3& R) const
{

  bool converged = false;
  int num_iters=0;
  Matrix3 F = *this;
  R=F;
  while(!converged){
    // Compute inverse of R
    Matrix3 RI = R.Inverse();

    // Compute error as the norm of R-Inverse - R-Transpose
    double ERR = (RI - R.Transpose()).Norm();
    if(ERR/10. + 1.0 == 1.0){
      converged = true;
    }

    // Average R with its inverse-transpose to get a new estimate of R
    R = (R+RI.Transpose())*0.5;

    num_iters++;
    if(num_iters==200){
      cerr.precision(15);
      cerr << "Matrix3::polarRotationAFFinv not converging. Matrix = " << endl;
      cerr << F << endl;
      cerr << "ERR = " << ERR << endl;
      exit(1);
    }
  }
  return;
}


// Polar decomposition of a non-singular square matrix
// If rightFlag == true return right stretch and rotation
// else return left stretch and rotation
void Matrix3::polarDecomposition(Matrix3& U,
                                 Matrix3& R,
                                 double tolerance,
                                 bool rightFlag) const
{
  Matrix3 F = *this;
  double det = F.Determinant();
  if ( det <= 0.0 ) {
    cerr << "Singular matrix in polar decomposition..." << endl;
    exit(1);
  }
  Matrix3 C = F.Transpose()*F;

  Matrix3 Csq = C*C;
  double I1 = C.Trace();
  double I2 = .5*(I1*I1 - Csq.Trace());
  double I3 = C.Determinant();
  double I1_over_3 = I1/3.0;

  double b = I2 - I1*I1_over_3;
  double c = -2.0*I1_over_3*I1_over_3*I1_over_3 + I1_over_3*I2 - I3;

  double x1, x2, x3;
  if(fabs(b) <= tolerance){
    c = (c > 0.0) ? c : 0.0; 
    x1 = -pow(c,1./3.); x2 = x1; x3 = x1;
  } else {
    double m = 2.*sqrt(-b/3.0);
    double n = (3.*c)/(m*b);
    if (fabs(n) > 1.0) n = copysign(1.0,n);  // n = cos(theta) 
                                             // and cannot be greater than 1.0
    double t = atan2(sqrt(1-n*n),n)/3.0;
    x1 = m*cos(t);
    x2 = m*cos(t + 2.0/3.0*M_PI);
    x3 = m*cos(t + 4.0/3.0*M_PI);
  }

  double lam1sq = x1 + I1_over_3;
  double lam2sq = x2 + I1_over_3;
  double lam3sq = x3 + I1_over_3;
  lam1sq = (lam1sq > 0.0) ? lam1sq : 0.0;
  lam2sq = (lam2sq > 0.0) ? lam2sq : 0.0;
  lam3sq = (lam3sq > 0.0) ? lam3sq : 0.0;
  double lam1 = sqrt(lam1sq);
  double lam2 = sqrt(lam2sq);
  double lam3 = sqrt(lam3sq);

  double i1 = lam1 + lam2 + lam3;
  double i2 = lam1*lam2 + lam2*lam3 + lam3*lam1;
  double i3 = lam1*lam2*lam3;
  double D = i1*i2 - i3;

  Matrix3 One; One.Identity();
  U = (C*(i1*i1-i2) + One*i1*i3 - Csq)*(1./D);
  Matrix3 Uinv = (C - U*i1 + One*i2)*(1./i3);
  R = F*Uinv;
  if (!rightFlag) U = F*R.Transpose();

  // Set small values to zero
  //for (int i = 0 ; i < 3; ++i) {
  //  for (int j = 0 ; j < 3; ++j) {
  //    R(i,j) = (fabs(R(i,j)) > tolerance) ? R(i,j) : 0.0;
  //    U(i,j) = (fabs(U(i,j)) > tolerance) ? U(i,j) : 0.0;
  //  }
  //}
}


int Matrix3::getEigenValues(double& e1, double& e2, double& e3) const
{
  // eigen values will be roots of the following cubic polynomial
  // x^3 + b*x^2 + c*x + d
  double c[4];
  c[3] = 1;
  c[2] = -Trace();

  c[1] = 0;
  for (int i = 0; i < 2; i++) {
    for (int j = i+1; j < 3; j++)
      c[1] += mat3[i][i]*mat3[j][j] - mat3[i][j]*mat3[j][i];
  }
  
  c[0] = -Determinant();
  double r[3];

  // num_values will be either 1, 2, or 3
  int num_values = SolveCubic(c, r);

  // sort the eigen values
  if (num_values == 3) {
    int greatest = 1;
    e1 = r[0];    
    if (r[1] > e1) {
      e1 = r[1];
      greatest = 2;
    }
    if (r[2] > e1) {
      e1 = r[2];
      greatest = 3;
    }
    e2 = (greatest != 2) ? r[1] : r[0];
    e3 = (greatest != 3) ? r[2] : r[0];
    if (e2 < e3)
      swap(e2, e3);
  }
  else if (num_values == 2)
    if (r[0] > r[1]) {
      e1 = r[0];
      e2 = r[1];
    } else {
      e1 = r[1];
      e2 = r[0];
    }
  else // num_values == 1
    e1 = r[0];

  return num_values;
}

int SolveQuadratic(double b, double c, double& e1, double&e2)
{
  double disc = b*b - 4*c;
  if (disc < 0) return 0;

  if (disc == 0) {
    e1 = -b/2;
    return 1;
  }
  else {
    disc = sqrt(disc);
    e1 = (-b - disc) / 2;
    e2 = (-b + disc) / 2;
    return 2;
  }
}

int Matrix3::getXYEigenValues(double& e1, double& e2) const
{
  // eigen values will be roots of the following quadratic
  // a*x^2 + b*x + c
  double b = -(mat3[0][0] + mat3[1][1]);
  double c = mat3[0][0] * mat3[1][1] - mat3[0][1] * mat3[1][0];
  return SolveQuadratic(b, c, e2, e1);
}

int Matrix3::getYZEigenValues(double& e1, double& e2) const
{
  // eigen values will be roots of the following quadratic
  // a*x^2 + b*x + c
  double b = -(mat3[1][1] + mat3[2][2]);
  double c = mat3[1][1] * mat3[2][2] - mat3[1][2] * mat3[2][1];
  return SolveQuadratic(b, c, e2, e1);
}

int Matrix3::getXZEigenValues(double& e1, double& e2) const
{
  // eigen values will be roots of the following quadratic
  // a*x^2 + b*x + c
  double b = -(mat3[0][0] + mat3[2][2]);
  double c = mat3[0][0] * mat3[2][2] - mat3[0][2] * mat3[2][0];
  return SolveQuadratic(b, c, e2, e1);
}

void Matrix3::prettyPrint(std::ostream &out_file) const
{
    Matrix3 m3 = *this;
    out_file <<  m3(0,0) << ' ' << m3(0,1) << ' ' << m3(0,2) << endl;
    out_file <<  m3(1,0) << ' ' << m3(1,1) << ' ' << m3(1,2) << endl;
    out_file <<  m3(2,0) << ' ' << m3(2,1) << ' ' << m3(2,2) << endl;
}

namespace Uintah {
  ostream &
  operator << (ostream &out_file, const Matrix3 &m3)
  {
    // Overload the output stream << operator
    
    out_file <<  m3(0,0) << ' ' << m3(0,1) << ' ' << m3(0,2) << ' ';
    out_file <<  m3(1,0) << ' ' << m3(1,1) << ' ' << m3(1,2) << ' ';
    out_file <<  m3(2,0) << ' ' << m3(2,1) << ' ' << m3(2,2) ;
    
    return out_file;
  }
}

void
Matrix3::eigen(SCIRun::Vector& eval, Matrix3& evec)
{
  // Convert the current matrix into a 2x2 TNT Array
  TNT::Array2D<double> A = toTNTArray2D();

  // Compute the eigenvectors using JAMA
  JAMA::Eigenvalue<double> eig(A);
  TNT::Array1D<double> d(3);
  eig.getRealEigenvalues(d);
  TNT::Array2D<double> V(3,3);
  eig.getV(V);

  // Sort in descending order
  for (int ii = 0; ii < 2; ++ii) {
    int kk = ii;
    double valk = d[ii];
    for (int jj = ii+1; jj < 3; jj++) {
      double valj = d[jj];
      if (valj > valk) 
        {
          kk = jj;
          valk = valj; 
        }
    }
    if (kk != ii) {
      double temp = d[ii];
      d[ii] = d[kk];
      d[kk] = temp;
      for (int ll = 0; ll < 3; ++ll) {
        temp = V[ll][ii];
        V[ll][ii] = V[ll][kk];
        V[ll][kk] = temp;
      }
    }
  }

  // Store in eval and evec
  for (int ii = 0; ii < 3; ++ii) {
    eval[ii] = d[ii];
    for (int jj = 0; jj < 3; ++jj) {
      evec(ii,jj) = V[ii][jj];
    }
  }

}
namespace Uintah {
  MPI_Datatype makeMPI_Matrix3()
  {
    ASSERTEQ(sizeof(Matrix3), sizeof(double)*9);

    MPI_Datatype mpitype;
    MPI_Type_vector(1, 9, 9, MPI_DOUBLE, &mpitype);
    MPI_Type_commit(&mpitype);

    return mpitype;
  }

  const TypeDescription* fun_getTypeDescription(Matrix3*)
  {
    static TypeDescription* td = 0;
    if(!td){
      td = scinew TypeDescription(TypeDescription::Matrix3, "Matrix3", true,
                                  &makeMPI_Matrix3);
    }
    return td;
  }

} // End namespace Uintah

