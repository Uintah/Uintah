#ifndef __SYMM_MATRIX3_H__
#define __SYMM_MATRIX3_H__

#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/Math/TntJama/tnt.h>

namespace Uintah {

  using SCIRun::Vector;
  using TNT::Array2D;

  /*! \class SymmMatrix3.h
    \brief Special matrix operations for a symmetric 3x3 matrix.
    \warning No copy constructors, operators etc.  Only a few 
    things needed for eigenvalue and eigenvector calculations
    have been implemented. 
   

    Compute eigenvalues/eigenvectors of real symmetric matrix using
    reduction to tridiagonal form, followed by QR iteration with
    implicit shifts.

    See Golub & Van Loan, "Matrix Computations" (3rd ed), Section 8.3
  */

  class SymmMatrix3 {

  public:
    // constructors
    inline SymmMatrix3();
    inline SymmMatrix3(const Matrix3&);

    // destructor
    virtual inline ~SymmMatrix3();

    /*! Calculate eigenvalues */
    void eigen(Vector& eval, Matrix3& evec);

    /*! Return a TNT Array2D */
    inline TNT::Array2D<double> toTNTArray2D() const;

  private:
    SymmMatrix3(const SymmMatrix3&);
    void operator=(const SymmMatrix3&);

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

} // End namespace Uintah

#endif  // __SYMM_MATRIX3_H__

