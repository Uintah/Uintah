//  class Matrix3
//    Matrix3 data type -- holds components of a 3X3 matrix
//    Features:
//      1.  Nearly, all the overloaded operations of the Matrix class,
//      no dynamic memory allocation.

#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/Math/CubeRoot.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include <Core/Util/TypeDescription.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/Assert.h>
#include <Core/Util/Endian.h>

#include <stdlib.h>

#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <fstream>
#include <sgi_stl_warnings_on.h>

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1209
#endif

using namespace Uintah;

using std::cout;
using std::endl;
using std::ostream;

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
   mat3[i-1][j-1] = value;
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
    cout << "Singular matrix in matrix inverse..." << endl;
    exit(1);
  }
  else
  {
    inv_matrix(1,1) = (*this)(2,2)*(*this)(3,3) - (*this)(2,3)*(*this)(3,2);
    inv_matrix(1,2) = -(*this)(1,2)*(*this)(3,3) + (*this)(3,2)*(*this)(1,3);
    inv_matrix(1,3) = (*this)(1,2)*(*this)(2,3) - (*this)(2,2)*(*this)(1,3);
    inv_matrix(2,1) = -(*this)(2,1)*(*this)(3,3) + (*this)(3,1)*(*this)(2,3);
    inv_matrix(2,2) = (*this)(1,1)*(*this)(3,3) - (*this)(1,3)*(*this)(3,1);
    inv_matrix(2,3) = -(*this)(1,1)*(*this)(2,3) + (*this)(2,1)*(*this)(1,3);
    inv_matrix(3,1) = (*this)(2,1)*(*this)(3,2) - (*this)(3,1)*(*this)(2,2);
    inv_matrix(3,2) = -(*this)(1,1)*(*this)(3,2) + (*this)(3,1)*(*this)(1,2);
    inv_matrix(3,3) = (*this)(1,1)*(*this)(2,2) - (*this)(1,2)*(*this)(2,1);
 
    inv_matrix = inv_matrix/det;

  }
  return inv_matrix;

} //end Inverse()

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
void Matrix3::triangularReduce(Matrix3& A, Vector* rhs, int& num_zero_rows,
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
bool Matrix3::solveParticularReduced(const Vector& rhs, Vector& xp,
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

    xp = Vector(x, y, z);
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
    xp = Vector(0, 0, 0);
    xp[i] = rhs[0];

    return true;
  case 3:
    // solution only if rhs == 0 (in which case everywhere is a solution)
    if (rhs == Vector(0, 0, 0)) {
      xp = Vector(0, 0, 0); // arbitrarily choose {0, 0, 0}
      return true;
    }
    else
      return false;
  default:
    ASSERTFAIL("unexpected num_zero_rows in Matrix3::solveParticularReduced");
  }
}
 
// solveHomogenous for a Matrix that has already by triangularReduced.
std::vector<Vector> Matrix3::solveHomogenousReduced(int num_zero_rows) const
{
  std::vector<Vector> basis_vectors;

  basis_vectors.resize(num_zero_rows);
  
  switch (num_zero_rows) {
  case 3:
    // Solutions everywhere : A = 0 matrix
    basis_vectors[0] = Vector(1, 0, 0);
    basis_vectors[1] = Vector(0, 1, 0);
    basis_vectors[2] = Vector(0, 0, 1);
    break;

  case 1:
    {
    // line of solutions : A = {{a, b, c} {0, d, e} {0, 0, 0}}
    // do backwards substition, using value of 1 arbitrarily
    // if a variable is not constrained

    Vector v;
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
      basis_vectors[0] = Vector(mat3[0][2], 0, -mat3[0][0]); // {c, 0, -a}
      basis_vectors[1] = Vector(mat3[0][1], -mat3[0][0], 0); // {b, -a, 0}
    }
    else if (max_index == 1) {
      // |b| largest, use line1 and line3
      basis_vectors[0] = Vector(0, mat3[0][2], -mat3[0][1]); // {0, c, -b}
      basis_vectors[1] = Vector(mat3[0][1], -mat3[0][0], 0); // {b, -a, 0}
    }
    else {
      // |c| largest, use line1 and line2
      basis_vectors[0] = Vector(0, mat3[0][2], -mat3[0][1]); // {0, c, -b}
      basis_vectors[1] = Vector(mat3[0][2], 0, -mat3[0][0]); // {c, 0, -a}
    }
  }

  return basis_vectors;
}

// Polar decomposition of a non-singular square matrix
// If rightFlag == true return right stretch and rotation
// else return left stretch and rotation
void Matrix3::polarDecomposition(Matrix3& stretch,
                                 Matrix3& rotation,
                                 double tolerance,
                                 bool rightFlag) const
{
  double det = this->Determinant();
  if ( det == 0.0 ) {
    cout << "Singular matrix in polar decomposition..." << endl;
    exit(1);
  }

  // Calculate the right (C) Cauchy-Green tensor 
  // where C = Ftranspose*F 
  Matrix3 C = (this->Transpose())*(*this);

  // Find the principal invariants of the left or right Cauchy-Green tensor (b) 
  // where b = F*Ftransposeb
  double I1 = C.Trace();
  double I1Square = I1*I1;
  Matrix3 CSquare = C*C;
  double I2 = 0.5*(I1Square - CSquare.Trace());
  double I3 = C.Determinant();

  // Find the principal stretches lambdaA, lambdaB, lambdaC
  // or lambda(ii), ii = 1..3
  double lambda[4]; lambda[0] = 0.0;
  double oneThird = 1.0/3.0;
  double oneThirdI1 = oneThird*I1;
  double bb = I2 - oneThird*I1Square;
  double cc = -(2.0/27.0)*I1Square*I1 + oneThirdI1*I2 - I3;
  if (fabs(bb) > tolerance) {
    double mm = 2.0*sqrt(-oneThird*bb);
    double nn = 3.0*cc/(mm*bb);
    if (fabs(nn) > 1.0) nn /= fabs(nn);
    double tt = oneThird*atan(sqrt(1-(nn*nn))/nn);
    for (int ii = 1; ii < 4; ++ii) 
      lambda[ii] = sqrt(mm*cos(tt+2.0*(double)(ii-1)*oneThird*M_PI) + oneThirdI1); 
  } else {
    for (int ii = 1; ii < 4; ++ii) 
      lambda[ii] = sqrt(-pow(cc, oneThird) + oneThirdI1); 
  }

  // Find the stretch tensor 
  Matrix3 one;
  one.Identity();
  double lambda2p3 = lambda[2] + lambda[3];
  double lambda2m3 = lambda[2]*lambda[3];
  double i1 = lambda[1] + lambda2p3;
  double i2 = lambda[1]*lambda2p3 + lambda2m3;
  double i3 = lambda[1]*lambda2m3;
  double DD = i1*i2 - i3;
  if (fabs(DD) > 0.0 && fabs(i3) > 0.0) {
    stretch = (CSquare*(-1) + C*(i1*i1-i2) + one*(i1*i3))*(1.0/DD);
    Matrix3 stretchInv = (C - (stretch*i1 - one*i2))*(1.0/i3);

    // Calculate the rotation tensor
    rotation = (*this)*stretchInv;
    if (!rightFlag) stretch = (rotation*stretch)*(rotation.Transpose());
    for (int ii = 1; ii < 4; ++ii) {
      for (int jj = 1; jj < 4; ++jj) {
	if (fabs(stretch(ii,jj)) < tolerance) stretch(ii,jj) = 0.0;
	if (fabs(rotation(ii,jj)) < tolerance) rotation(ii,jj) = 0.0;
      }
    }
  }
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

ostream & operator << (ostream &out_file, const Matrix3 &m3)
{
  // Overload the output stream << operator

  out_file <<  m3(1,1) << ' ' << m3(1,2) << ' ' << m3(1,3) << endl;
  out_file <<  m3(2,1) << ' ' << m3(2,2) << ' ' << m3(2,3) << endl;
  out_file <<  m3(3,1) << ' ' << m3(3,2) << ' ' << m3(3,3) ;

  return out_file;
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

