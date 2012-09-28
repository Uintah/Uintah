/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
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


#ifndef __SYMM_MATRIX3_H__
#define __SYMM_MATRIX3_H__

#include <Core/Math/Matrix3.h>
#include <Core/Math/TntJama/tnt.h>

namespace Uintah {

  using SCIRun::Vector;
  using TNT::Array2D;

  /*! \class SymmMatrix3
    \brief Special matrix operations for a symmetric 3x3 matrix.
    \warning No copy constructors, operators etc.  Only a few 
    things needed for eigenvalue and eigenvector calculations
    have been implemented. 
   
    Compute eigenvalues/eigenvectors of real symmetric matrix using
    reduction to tridiagonal form, followed by QR iteration with
    implicit shifts.

    Order is : (1,1), (2,2), (3,3), (2,3), (3,1), (1,2)

    See Golub & Van Loan, "Matrix Computations" (3rd ed), Section 8.3
  */

  class SymmMatrix3 {

  public:
    // constructors
    inline SymmMatrix3();
    inline SymmMatrix3(const Matrix3&);
    inline SymmMatrix3(const SymmMatrix3&);

    // destructor
    virtual inline ~SymmMatrix3();

    /*! Calculate eigenvalues */
     void eigen(Vector& eval, Matrix3& evec);

    /*! Return a TNT Array2D */
    inline TNT::Array2D<double> toTNTArray2D() const;

    /*! Access operators */
    inline void operator=(const SymmMatrix3& mat);
    inline double operator[] (int i) const;
    inline double& operator[] (int i);

    /*! Compute Identity matrix */
    inline void Identity();

    /*! Compute trace of the matrix */
    inline double Trace() const;

    /*! Compute deviatoric part of the matrix */
     SymmMatrix3 Deviatoric() const;

    /*! Compute norm of a SymmMatrix3 */
     double Norm() const;

    /*! Compute Dyadic Product of two SymmMatrix3s */
     void Dyad(const SymmMatrix3& V, double dyad[6][6]) const;

    /*! Compute Dot Product of two SymmMatrix3s */
     Matrix3 Multiply(const SymmMatrix3& V) const; 

    /*! Compute square of the matrix */
     SymmMatrix3 Square() const;

    /*! Compute Inner Product of two SymmMatrix3s */
     double Contract(const SymmMatrix3& V) const; 

  private:
    double mat3[6];

  };




  inline SymmMatrix3::SymmMatrix3()
    {
      // Initialization to 0.0
      for(int i=0;i<6;i++) mat3[i] = 0.0;
    }

  inline SymmMatrix3::SymmMatrix3(const Matrix3& copy)
    {
      mat3[0] = copy(0,0);
      mat3[1] = copy(1,1);
      mat3[2] = copy(2,2);
      mat3[3] = copy(1,2);
      mat3[4] = copy(2,0);
      mat3[5] = copy(0,1);
    }

  inline SymmMatrix3::SymmMatrix3(const SymmMatrix3& copy)
    {
      mat3[0] = copy.mat3[0];
      mat3[1] = copy.mat3[1];
      mat3[2] = copy.mat3[2];
      mat3[3] = copy.mat3[3];
      mat3[4] = copy.mat3[4];
      mat3[5] = copy.mat3[5];
    }

  inline SymmMatrix3::~SymmMatrix3()
    {
    }

  inline TNT::Array2D<double> SymmMatrix3::toTNTArray2D() const
    {
      TNT::Array2D<double> mat(3,3);
      mat[0][0] = mat3[0];
      mat[1][1] = mat3[1];
      mat[2][2] = mat3[2];
      mat[1][2] = mat3[3];
      mat[0][2] = mat3[4];
      mat[0][1] = mat3[5];
      mat[1][0] = mat3[5];
      mat[2][0] = mat3[4];
      mat[2][1] = mat3[3];
      return mat;
    }

  inline void SymmMatrix3::operator=(const SymmMatrix3& mat)
    {
      mat3[0] = mat.mat3[0];
      mat3[1] = mat.mat3[1];
      mat3[2] = mat.mat3[2];
      mat3[3] = mat.mat3[3];
      mat3[4] = mat.mat3[4];
      mat3[5] = mat.mat3[5];
    }

  inline double SymmMatrix3::operator[](int i) const
    {
      return mat3[i]; 
    }

  inline double& SymmMatrix3::operator[](int i)
    {
      return mat3[i]; 
    }

  inline void SymmMatrix3::Identity()
    {
      mat3[0] = 1.0; mat3[1] = 1.0; mat3[2] = 1.0;
      mat3[3] = 0.0; mat3[4] = 0.0; mat3[5] = 0.0;
    }

  inline double SymmMatrix3::Trace() const
    {
      return (mat3[0] + mat3[1] + mat3[2]);
    }

} // End namespace Uintah

#endif  // __SYMM_MATRIX3_H__

