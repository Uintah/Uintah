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
namespace Uintah {
   class TypeDescription;
}


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
  inline Matrix3 operator * (const double value);

  // divide Matrix3 by a constant
  inline Matrix3 operator / (const double value);

  // modify by adding right hand side
  inline void operator += (const Matrix3 &m3);

  // modify by subtracting right hand side
  inline void operator -= (const Matrix3 &m3);

  // add two Matrix3s
  inline Matrix3 operator + (const Matrix3 &m3);

  // multiply two Matrix3s
  inline Matrix3 operator * (const Matrix3 &m3);

  // subtract two Matrix3s
  inline Matrix3 operator - (const Matrix3 &m3);

  // multiply by constant
  inline void operator *= (const double value);

  // divide by constant
  inline void operator /= (const double value);

  //Determinant
  inline double Determinant();

  //Inverse
  Matrix3 Inverse();

  //Trace
  inline double Trace();

  //Norm, sqrt(M:M)
  inline double Norm();

  //Identity
  inline void Identity();

  //Transpose
  inline Matrix3 Transpose();

};

std::ostream & operator << (std::ostream &out_file, const Matrix3 &m3);
namespace Uintah {
const TypeDescription* fun_getTypeDescription(Matrix3*);
}

inline double Matrix3::Trace()
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

inline double Matrix3::Norm()
{
  // Return the trace of a 3x3 matrix

  double norm = 0.0;

  for (int i = 0; i< 3; i++) {
    for(int j=0;j<3;j++){
	norm += mat3[i][j]*mat3[i][j];
    }
  }

  return sqrt(norm);

}

inline Matrix3 Matrix3::Transpose()
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
  // Multiply each component of the Matrix3 by the value

  for(int i=0;i<3;i++){
    for(int j=0;j<3;j++){
        mat3[i][j] += m3(i,j);
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

inline Matrix3 Matrix3::operator * (const double value)
{
//   Multiply a Matrix3 by a constant

  return Matrix3(mat3[0][0]*value,mat3[0][1]*value,mat3[0][2]*value,
		 mat3[1][0]*value,mat3[1][1]*value,mat3[1][2]*value, 
		 mat3[2][0]*value,mat3[2][1]*value,mat3[2][2]*value); 
}

inline Matrix3 Matrix3::operator * (const Matrix3 &m3)
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

inline Matrix3 Matrix3::operator + (const Matrix3 &m3)
{
//   Add a Matrix3 to a Matrix3

  return Matrix3(mat3[0][0] + m3(1,1),mat3[0][1] + m3(1,2),mat3[0][2] + m3(1,3),
		 mat3[1][0] + m3(2,1),mat3[1][1] + m3(2,2),mat3[1][2] + m3(2,3),
		 mat3[2][0] + m3(3,1),mat3[2][1] + m3(3,2),mat3[2][2] + m3(3,3));
}

inline Matrix3 Matrix3::operator - (const Matrix3 &m3)
{
//   Subtract a Matrix3 from a Matrix3

  return Matrix3(mat3[0][0] - m3(1,1),mat3[0][1] - m3(1,2),mat3[0][2] - m3(1,3),
		 mat3[1][0] - m3(2,1),mat3[1][1] - m3(2,2),mat3[1][2] - m3(2,3),
		 mat3[2][0] - m3(3,1),mat3[2][1] - m3(3,2),mat3[2][2] - m3(3,3));
}

inline Matrix3 Matrix3::operator / (const double value)
{
//   Divide a Matrix3 by a constant

  assert(value != 0.);
  double ivalue = 1.0/value;

  return Matrix3(mat3[0][0]*ivalue,mat3[0][1]*ivalue,mat3[0][2]*ivalue,
		 mat3[1][0]*ivalue,mat3[1][1]*ivalue,mat3[1][2]*ivalue, 
		 mat3[2][0]*ivalue,mat3[2][1]*ivalue,mat3[2][2]*ivalue); 
}

inline double Matrix3::Determinant()
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
