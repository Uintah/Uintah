//  Matrix3.h
//
//  class Matrix3
//    Matrix3 data type -- holds components of a 3X3 matrix
//
//
//
//    Features:
//      1.  Nearly, all the overloaded operations of the Matrix class,
//	no dynamic memory allocation.

#ifndef __MATRIX3_H__
#define __MATRIX3_H__

#include <math.h>
#include <assert.h>
#include <iosfwd>
#include <SCICore/Geometry/Vector.h>
#include <vector>

namespace Uintah {
   class TypeDescription;
}

using namespace SCICore::Geometry;

class Matrix3 {
 private:
  // data areas
  double mat3[3][3];

 public:
  // constructors
  inline Matrix3();
  inline Matrix3(double value);
  inline Matrix3(double v00, double v01, double v02,
                 double v10, double v11, double v12,
                 double v20, double v21, double v22);
  inline Matrix3(double mat3[3][3]);
  inline Matrix3(const Matrix3&);

  // copy constructor
//  Matrix3(const Matrix3 &m3);

  // destructor
  inline ~Matrix3();

  // assign a value to all components of the Matrix3
  inline void set(double val);

  // assign a value to components of the Matrix3
  void set(int i, int j, double val);

  // assignment operator
  inline void operator = (const Matrix3 &m3);

  // access operator
  inline double operator() (int i,int j) const;
  inline double & operator() (int i,int j);

  // multiply Matrix3 by a constant
  inline Matrix3 operator * (const double value) const;

  // divide Matrix3 by a constant
  inline Matrix3 operator / (const double value) const;
 
  // modify by adding right hand side
  inline void operator += (const Matrix3 &m3);

  // modify by subtracting right hand side
  inline void operator -= (const Matrix3 &m3);

  // add two Matrix3s
  inline Matrix3 operator + (const Matrix3 &m3) const;

  // multiply two Matrix3s
  inline Matrix3 operator * (const Matrix3 &m3) const;

   // multiply Vector by Matrix3
  inline Vector operator * (const Vector& V) const;  
  
  // subtract two Matrix3s
  inline Matrix3 operator - (const Matrix3 &m3) const;

  // multiply by constant
  inline void operator *= (const double value);

  // divide by constant
  inline void operator /= (const double value);

  //Determinant
  inline double Determinant() const;

  //Inverse
  Matrix3 Inverse() const;

  //Trace
  inline double Trace() const;

  //Norm, sqrt(M:M)
  inline double Norm() const;

  //Identity
  inline void Identity();

  //Transpose
  inline Matrix3 Transpose() const;

  // Returns number of real eigen values and passes back
  // the values.  There will either be one eigenvalue
  // passed back via e1, or three eigenvalue in which case
  // they will be sorted such that e1 <= e2 <= e3.
  int getEigenValues(double& e1, double& e2, double& e3) const;

  // Returns an array of eigenvectors that form the basis
  // of eigenvectors corresponding to the given eigenvalue.
  // There may be 0 (if eigen_value is not a true eigen value),
  // 1, 2, or 3 ({1,0,0},{0,1,0},{0,0,1}) of these eigenvectors
  // (> 1 if eigen value has degeneracies I believe) and they
  // will NOT necessarily be normalized.
  inline std::vector<Vector> getEigenVectors(double eigen_value) const;

  // Solves for a single particular solution (possible arbitrary) to
  // the equation system: Ax = rhs where A is this Matrix.
  // Returns false if there is no solution, otherwise a particular
  // solution is passed back via xp.
  inline bool solveParticular(Vector rhs, Vector& xp) const;
  
  // Solves for the space of solutions for Ax = 0 where A is this Matrix.
  // The solution is any linear combination of the resulting array
  // of vectors.  That is, if result = {xg1, xg2, ...}, then
  // x = a*xg1 + b*xg2 + ... where a, b, etc. are arbitrary scalars.
  //
  // Result Possibilities:
  // Single Solution: result.size() = 0, {0,0,0} is only solution
  // Line of Solutions: result.size() = 1
  // Plane of Solutions: result.size() = 2
  // Solution everywhere: result = {{1,0,0},{0,1,0},{0,0,1}}
  inline std::vector<Vector> solveHomogenous() const;

  // Solves for the space of solutions for Ax = rhs where A is this Matrix.
  // This is a more efficient combination of solveParticular and
  // solveHomogenous (where the homogenous results are passed back via
  // xg_basis).
  inline bool solve(Vector rhs, Vector& xp,
		    std::vector<Vector>& xg_basis) const;
private:
  // Reduce the matrix and rhs, representing the equation system:
  // A*x = y = rhs, to a matrix in upper triangular form with
  // corresponding rhs for an equivalent equation systme.
  // The guarantee for this new matrix is that the first non-zero
  // column of a row is to the right (greater) of the first non-zero
  // column in the row above it and the first non-zero column of
  // any row has zeroes in every other row and a one in that row.
  // If rhs == NULL, then the rhs is assumed to be
  // the zero vector and thus will not need to change.
  static void triangularReduce(Matrix3& A, Vector* rhs, int& num_zero_rows);

  // solveHomogenous for a Matrix that has already by triangularReduced
  std::vector<Vector> solveHomogenousReduced(int num_zero_rows) const;

  // solveHomogenous for a Matrix that has already by triangularReduced
  bool solveParticularReduced(const Vector& rhs, Vector& xp,
			      int num_zero_rows) const;
};

std::ostream & operator << (std::ostream &out_file, const Matrix3 &m3);
namespace Uintah {
const TypeDescription* fun_getTypeDescription(Matrix3*);
}

inline double Matrix3::Trace() const
{
  // Return the trace of a 3x3 matrix

  double trace = 0.0;

  for (int i = 0; i< 3; i++) {
    trace += mat3[i][i];
  }

  return trace;

}

inline Matrix3::Matrix3()
{
  // Default Constructor
  // Initialization to 0.0
  for(int i=0;i<3;i++){
    for(int j=0;j<3;j++){
        mat3[i][j] = 0.0;
    }
  }
}

inline Matrix3::Matrix3(double value)
{
  // Constructor
  // With initialization to a single value

  for(int i=0;i<3;i++){
    for(int j=0;j<3;j++){
        mat3[i][j] = value;
    }
  }
}


inline Matrix3::Matrix3(double a00,double a01,double a02,
                        double a10,double a11,double a12,
                        double a20,double a21,double a22)
{
  // Constructor
  // With full initialization

  mat3[0][0] = a00; mat3[0][1] = a01; mat3[0][2] = a02;
  mat3[1][0] = a10; mat3[1][1] = a11; mat3[1][2] = a12;
  mat3[2][0] = a20; mat3[2][1] = a21; mat3[2][2] = a22;

}

inline Matrix3::Matrix3(const Matrix3& copy)
{
  // Copy Constructor

  mat3[0][0] = copy.mat3[0][0]; mat3[0][1] = copy.mat3[0][1]; mat3[0][2] = copy.mat3[0][2];
  mat3[1][0] = copy.mat3[1][0]; mat3[1][1] = copy.mat3[1][1]; mat3[1][2] = copy.mat3[1][2];
  mat3[2][0] = copy.mat3[2][0]; mat3[2][1] = copy.mat3[2][1]; mat3[2][2] = copy.mat3[2][2];

}

inline Matrix3::~Matrix3()
{
  // Destructor
  // Do nothing
}

inline double Matrix3::Norm() const
{
  // Return the norm of a 3x3 matrix

  double norm = 0.0;

  for (int i = 0; i< 3; i++) {
    for(int j=0;j<3;j++){
	norm += mat3[i][j]*mat3[i][j];
    }
  }

  return sqrt(norm);

}

inline Matrix3 Matrix3::Transpose() const
{
  // Return the transpose of a 3x3 matrix

  return Matrix3(mat3[0][0],mat3[1][0],mat3[2][0],
		 mat3[0][1],mat3[1][1],mat3[2][1],
		 mat3[0][2],mat3[1][2],mat3[2][2]);

}

inline void Matrix3::Identity()
{
  // Set a matrix3 to the identity

  mat3[0][0] = mat3[1][1] = mat3[2][2] = 1.0;
  mat3[0][1] = mat3[0][2] = mat3[1][0] = 0.0;
  mat3[1][2] = mat3[2][0] = mat3[2][1] = 0.0;

}

inline void Matrix3::operator = (const Matrix3 &m3)
{
  // Copy value from right hand side of assignment

  for(int i=0;i<3;i++){
    for(int j=0;j<3;j++){
        mat3[i][j] = m3(i+1,j+1);
    }
  }

}

inline void Matrix3::set(const double value)
{
  // Assign the Matrix3 the value components
  for(int i=0;i<3;i++){
    for(int j=0;j<3;j++){
        mat3[i][j] = value;
    }
  }
}

inline void Matrix3::operator *= (const double value)
{
  // Multiply each component of the Matrix3 by the value

  for(int i=0;i<3;i++){
    for(int j=0;j<3;j++){
        mat3[i][j] *= value;
    }
  }

}

inline void Matrix3::operator += (const Matrix3 &m3)
{
  // += operator 

  for(int i=0;i<3;i++){
    for(int j=0;j<3;j++){
        mat3[i][j] += m3(i+1,j+1);
    }
  }

}

inline void Matrix3::operator /= (const double value)
{
  // Divide each component of the Matrix3 by the value

  assert(value != 0.);
  double ivalue = 1./value;
  for(int i=0;i<3;i++){
    for(int j=0;j<3;j++){
        mat3[i][j] *= ivalue;
    }
  }
}

inline Matrix3 Matrix3::operator * (const double value) const
{
//   Multiply a Matrix3 by a constant

  return Matrix3(mat3[0][0]*value,mat3[0][1]*value,mat3[0][2]*value,
		 mat3[1][0]*value,mat3[1][1]*value,mat3[1][2]*value, 
		 mat3[2][0]*value,mat3[2][1]*value,mat3[2][2]*value); 
}

inline Matrix3 Matrix3::operator * (const Matrix3 &m3) const
{
//   Multiply a Matrix3 by a Matrix3

  return Matrix3(mat3[0][0]*m3(1,1)+mat3[0][1]*m3(2,1)+mat3[0][2]*m3(3,1),
		 mat3[0][0]*m3(1,2)+mat3[0][1]*m3(2,2)+mat3[0][2]*m3(3,2),
		 mat3[0][0]*m3(1,3)+mat3[0][1]*m3(2,3)+mat3[0][2]*m3(3,3),

		 mat3[1][0]*m3(1,1)+mat3[1][1]*m3(2,1)+mat3[1][2]*m3(3,1),
                 mat3[1][0]*m3(1,2)+mat3[1][1]*m3(2,2)+mat3[1][2]*m3(3,2),
                 mat3[1][0]*m3(1,3)+mat3[1][1]*m3(2,3)+mat3[1][2]*m3(3,3),

		 mat3[2][0]*m3(1,1)+mat3[2][1]*m3(2,1)+mat3[2][2]*m3(3,1),
                 mat3[2][0]*m3(1,2)+mat3[2][1]*m3(2,2)+mat3[2][2]*m3(3,2),
                 mat3[2][0]*m3(1,3)+mat3[2][1]*m3(2,3)+mat3[2][2]*m3(3,3));
}

inline Vector Matrix3::operator * (const Vector& V) const
{
  return Vector(mat3[0][0] * V(0) + mat3[0][1] * V(1) + mat3[0][2] * V(2),
		mat3[1][0] * V(0) + mat3[1][1] * V(1) + mat3[1][2] * V(2),
		mat3[2][0] * V(0) + mat3[2][1] * V(1) + mat3[2][2] * V(2));
}


inline Matrix3 Matrix3::operator + (const Matrix3 &m3) const
{
//   Add a Matrix3 to a Matrix3

  return Matrix3(mat3[0][0] + m3(1,1),mat3[0][1] + m3(1,2),mat3[0][2] + m3(1,3),
		 mat3[1][0] + m3(2,1),mat3[1][1] + m3(2,2),mat3[1][2] + m3(2,3),
		 mat3[2][0] + m3(3,1),mat3[2][1] + m3(3,2),mat3[2][2] + m3(3,3));
}

inline Matrix3 Matrix3::operator - (const Matrix3 &m3) const
{
//   Subtract a Matrix3 from a Matrix3

  return Matrix3(mat3[0][0] - m3(1,1),mat3[0][1] - m3(1,2),mat3[0][2] - m3(1,3),
		 mat3[1][0] - m3(2,1),mat3[1][1] - m3(2,2),mat3[1][2] - m3(2,3),
		 mat3[2][0] - m3(3,1),mat3[2][1] - m3(3,2),mat3[2][2] - m3(3,3));
}

inline Matrix3 Matrix3::operator / (const double value) const
{
//   Divide a Matrix3 by a constant

  assert(value != 0.);
  double ivalue = 1.0/value;

  return Matrix3(mat3[0][0]*ivalue,mat3[0][1]*ivalue,mat3[0][2]*ivalue,
		 mat3[1][0]*ivalue,mat3[1][1]*ivalue,mat3[1][2]*ivalue, 
		 mat3[2][0]*ivalue,mat3[2][1]*ivalue,mat3[2][2]*ivalue); 
}

inline double Matrix3::Determinant() const
{
  // Return the determinant of a 3x3 matrix

  double temp = 0.0;

  temp= mat3[0][0]*mat3[1][1]*mat3[2][2] +
        mat3[0][1]*mat3[1][2]*mat3[2][0] +
        mat3[0][2]*mat3[1][0]*mat3[2][1] -
        mat3[0][2]*mat3[1][1]*mat3[2][0] -
        mat3[0][1]*mat3[1][0]*mat3[2][2] -
        mat3[0][0]*mat3[1][2]*mat3[2][1];


  // return result

  return temp;
}

inline double Matrix3::operator () (int i, int j) const
{
  // Access the i,j component
  return mat3[i-1][j-1];
}

inline double &Matrix3::operator () (int i, int j)
{
  // Access the i,j component
  return mat3[i-1][j-1];
}

inline std::vector<Vector> Matrix3::getEigenVectors(double eigen_value) const
{
  // A*x = e*x
  // (A - e*I)*x = 0
  Matrix3 A_sub_eI(mat3[0][0] - eigen_value, mat3[0][1], mat3[0][2],
	       mat3[1][0], mat3[1][1] - eigen_value, mat3[1][2],
	       mat3[2][0], mat3[2][1], mat3[2][2] - eigen_value);
  int num_zero_rows;
  triangularReduce(A_sub_eI, NULL, num_zero_rows);
  return A_sub_eI.solveHomogenousReduced(num_zero_rows);
}

inline bool Matrix3::solveParticular(Vector rhs, Vector& xp) const
{
  int num_zero_rows;
  Matrix3 A(*this);
  triangularReduce(A, &rhs, num_zero_rows);
  return A.solveParticularReduced(rhs, xp, num_zero_rows);
}

inline std::vector<Vector> Matrix3::solveHomogenous() const
{
  int num_zero_rows;
  Matrix3 A(*this);
  triangularReduce(A, NULL, num_zero_rows);
  return A.solveHomogenousReduced(num_zero_rows);
}

inline bool Matrix3::solve(Vector rhs, Vector& xp,
		  std::vector<Vector>& xg_basis) const
{
  int num_zero_rows;
  Matrix3 A(*this);
  triangularReduce(A, &rhs, num_zero_rows);
  if (A.solveParticularReduced(rhs, xp, num_zero_rows)) {
    xg_basis = A.solveHomogenousReduced(num_zero_rows);
    return true;
  }
  xg_basis.resize(0);
  return false;
}

// This is backwards: if the vector comes first then it should
// multiply the matrix columnwise instead of rowwise.  For now,
// I won't fix it because changing it may break other code. -- witzel
#include <SCICore/Geometry/Vector.h>
inline SCICore::Geometry::Vector operator*(const SCICore::Geometry::Vector& v, const Matrix3& m3) {
  // Right multiply a Vector by a Matrix3

  double x = v.x()*m3(1,1)+v.y()*m3(1,2)+v.z()*m3(1,3);
  double y = v.x()*m3(2,1)+v.y()*m3(2,2)+v.z()*m3(2,3);
  double z = v.x()*m3(3,1)+v.y()*m3(3,2)+v.z()*m3(3,3);

  return SCICore::Geometry::Vector(x, y, z);
}


#endif  // __MATRIX3_H__

// $Log$
// Revision 1.7  2000/08/18 17:06:11  guilkey
// Fixed the += operator.
//
// Revision 1.6  2000/08/15 22:01:59  witzel
// Sorting eigenvalues e1, e2, e3.
//
// Revision 1.5  2000/08/15 19:15:19  witzel
// Added methods for finding eigenvalues, eigenvectors and solving
// equation systems of the form Ax=b and Ax=0.  Also added M*v
// operation (where M is a Matrix3 and v is a Vector).
//
// Revision 1.4  2000/05/20 08:09:12  sparker
// Improved TypeDescription
// Finished I/O
// Use new XML utility libraries
//
// Revision 1.3  2000/05/05 15:08:30  guilkey
// Added += operator to Matrix3.
//
// Revision 1.2  2000/05/05 06:55:04  dav
// put in dummy code for Matrix3 += operator so everything would compile
//
// Revision 1.1  2000/03/14 22:12:43  jas
// Initial creation of the utility directory that has old matrix routines
// that will eventually be replaced by the PSE library.
//
// Revision 1.1  2000/02/24 06:11:58  sparker
// Imported homebrew code
//
// Revision 1.2  2000/01/25 18:28:44  sparker
// Now links and kinda runs
//
// Revision 1.1  2000/01/24 22:48:54  sparker
// Stuff may actually work someday...
//
// Revision 1.7  1999/12/20 23:56:41  guilkey
// Worked over the Matrix3 class to do smarter things in methods that return
// a Matrix3.  This results in a dramatic improvement in performance for the
// + and * operators, which are used heavily in the constitutive models.
//
// Revision 1.6  1999/12/18 19:31:53  guilkey
// Fixed the Matrix3 class so that it now uses a double Mat3[3][3] for
// storage, rather than the 4X4 it used before.  This required adding offsets
// for the access operators.  A future improvement will be to fix all of the
// code so that this isn't necessary.
//
// Revision 1.5  1999/08/18 22:09:31  zhangr
// *** empty log message ***
//
// Revision 1.4  1999/08/17 20:41:02  zhangr
// // - Added two more functions:
// // - overloaded set() to assign the Matrix3 the value components
// // - Added Inverse() to inverse a 3x3 matrix.
// // - r. zhang.
//
// Revision 1.3  1999/05/24 21:11:18  guilkey
// Added Norm() function to return the norm of a Matrix3.
//
// Revision 1.2  1999/02/25 05:52:45  guilkey
// Inlined access operators.
//
// Revision 1.1  1999/02/18 21:13:45  guilkey
// Matrix3 is a fixed size matrix (3X3).
//
