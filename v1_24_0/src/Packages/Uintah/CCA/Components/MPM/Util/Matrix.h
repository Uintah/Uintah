//  Matrix.h 
//  class Matrix
//    Type safe and access safe Matrix data type.
//    Dense Matrix class.
//    Uses templates.
//    Assumes all matrices starting index is 1 (one)
//    Features:
//      1.  Dense Matrix format, can resize.
//      2.  Simple LU decomposition and solve routine
//      3.  Transpose operator
//      4.  Create an Identity Matrix (square Matrix only)
//      5.  Matrix * Matrix operation
//      6.  Matrix * BoundedArray operation (and visa versa)
//      7.  Dot product of two BoundedArrays
//      8.  Add a scalar to a BoundedArray
//      9.  Multiply each element of a BoundedArray by a scalar
//     10.  Add two BoundedArrays
//     11.  Subtract two BoundedArrays
//     12.  Add two matrices together
//     13.  Subtract two matrices
//      Usage:
//        Matrix<double> a_Matrix();     // declare a double Matrix - no size 
//                                       // specified
//        Matrix<double> a_Matrix(10,10); // declare a 10x10 Matrix of doubles
//        Matrix<double> a_Matrix(10,10,2.0); // declare a 10x10 Matrix of 
//                                            // doubles initialized to 2.0   
//        Matrix<double> b_Matrix(a_Matrix); // copy the contents of a_Matrix
//                                         // to b_Matrix;
//        a_Matrix.~Matrix();           //  destroy a Matrix
//        Matrix<double> b_mat = a_mat; //  set b_mat equal to a_mat
//        a_Matrix.numerRows();       //  return number of rows of a_Matrix
//        a_Matrix.numberColumns();   //  return number of columns of a_Matrix //        double b = a_mat[i][j];    //  return the i,j entry of a_mat
//        a_mat[i][j] = 5.0;         //  set the i,j entry of a_mat to 5.0
//        a_Matrix.Identity();       //  set a_Matrix to the identity Matrix   //        a_Matrix.Zero();           //  set each element to be zero
//        a_Matrix.Transpose();      //  return the transpose of the Matrix
//        a_Matrix.LUdecomp(row_perm,determ);  // Perform the LUdecomposition
//        a_Matrix.LUbacksub(row_perm,b_vec); // Perform the LUbacksubstitution
//        a_Matrix.Solve(b_vec);  // Solve Ax = b_vec, b_vec holds solution
//        new_mat = a_mat * b_mat;  // Perform Matrix * Matrix multiplication
//        new_array = a_mat * b_array;  // Perform Matrix * array multiply
//        new_array = b_array * a_mat;  // Perform array * Matrix multiply
//        result = a_array * b_array;  // Dot product of two arrays
//        new_array = scalar + a_array; // add scalar to each value of array
//        new_array = a_array + scalar; // add scalar to each value of array
//        new_array = scalar * a_array; // multiply all array elems by scalar
//        new_array = a_array + b_array; // add two arrays together
//        new_array = a_array - b_array; // subtract two arrays
//        new_mat = a_mat + b_mat;  // add two matrices 
//        new_mat = a_mat - b_mat;  // subtract two matrices
//        new_mat = a_mat * scalar;  // multiply all elements by scalar


#ifndef __MATRIX_H__
#define __MATRIX_H__


#include <assert.h>
#include "BoundedArray.h"
namespace SCIRun {
  class Vector;
}

template <class T> class SymmetricMatrix;

template <class T> class Matrix {
 protected:
  // data areas
       BoundedArray<BoundedArray<T> *> rows;

 
 public:
  // constructors and destructor
       Matrix();
       Matrix( int numberOfRows,  int numberOfColumns);
       Matrix( int numberOfRows,  int numberOfColumn,
	      const T &initialValue);
  // copy constructor
       Matrix(const Matrix<T> &source); 
  // destructor 
       virtual ~Matrix(); 

  // Assignment operator
       Matrix<T>  & operator = (const Matrix<T> &source);
       Matrix<T>  & operator = (const SymmetricMatrix<T> &source);

  // access to elements via subscript
       inline BoundedArray<T> & operator[](int index);
       inline BoundedArray<T> operator[](int index) const;

  // Change the dimensions of the matrix
       void resize( int rows,  int columns);

  // Dimensions of Matrix
       int numberRows() const;
       int numberColumns() const;

  // Identity Matrix
       void Identity();

  // Zero out a Matrix
       void Zero();

  // Matrix Transpose();
       Matrix<T>  Transpose();

  // Matrix Trace();
       T  Trace();

  // LU Decomposition of a Matrix
       void LUdecomp(BoundedArray<int> &row_perm, T &determ);

  // LU Backsubstition of a Matrix with unknown b array
       void LUbacksub(BoundedArray<int> &row_perm, BoundedArray<T> &b);

  // Solve a Ax = b using LUdecomp and LUbacksub.  b is replaced with
  // the solution vector.  A is replaced with the LUdecomposition.

       void Solve(BoundedArray<T> &b);

  // Determinant of a matrix

  T Determinant();

  // Determinant of a 3x3 matrix
  T Determinant3();


};

  // inlined functions

template<class T> BoundedArray<T> & Matrix<T>::operator[] (int index)
{
  // subscript a Matrix value
  // leading subscript in a Matrix expression
  // check that the index is valid

  assert(rows[index] != 0);

  // return array value, use pointer dereference to get
  // reference to actual vector

  return  *rows[index];

}

template<class T> BoundedArray<T>  Matrix<T>::operator[] (int index) const
{
  // subscript a Matrix value
  // leading subscript in a Matrix expression
  // check that the index is valid

  assert(rows[index] != 0);

  // return array value, use pointer dereference to get
  // reference to actual vector

  return  *rows[index];

}


#include "SymmetricMatrix.h"

// Non-member function declarations					    

 template<class T>  inline Matrix<T>  operator * (const Matrix<T> &left,
                                         const Matrix<T> &right);

 template<class T>  inline Matrix<T>  operator * (const Matrix<T> &left, 
					 const Matrix<T> &right)
{
  // perform Matrix multiplication of left by right
  // first get dimensions of matrices

  int n = left.numberRows();
  int m = left.numberColumns();
  int p = right.numberColumns();

  // Check that they are compatible
  assert(m == right.numberRows());

  // allocate space for the result
  Matrix<T> result(n,p,0.0);
 

   
  // fill in the values
  for (int i = 1; i<= n; i++) {
    for (int j = 1; j<= p; j++) {
      (result)[i][j] = 0.;
      for (int k = 1; k <= m; k++) {
    	(result)[i][j] +=  left[i][k] * right[k][j];
      }
    }
  }

  // return the result
  return result;
}

template<class T> BoundedArray<T> operator * (const Matrix<T> &left,
						const BoundedArray<T> &right);

template<class T> BoundedArray<T> operator * (const BoundedArray<T> &left,
						const Matrix<T> &right);

#if 0
// Not need anymore?  -Steve
template<class T> BoundedArray<Vector> operator * (const Matrix<T> &left,
                                        const BoundedArray<Vector> &right);
#endif

template<class T> T operator * (const BoundedArray<T> &left,
				  const BoundedArray<T> &right);

template<class T> BoundedArray<T>  operator + (const T &value,
						const BoundedArray<T> &right);

template<class T> BoundedArray<T>  operator + (const BoundedArray<T> &left,
						const T &value);

template<class T> BoundedArray<T>  operator * (const T &value,
						const BoundedArray<T> &right);

template<class T> BoundedArray<T>  operator / (const BoundedArray<T> &right,
					       const T &value);

template<class T> BoundedArray<T>  operator + (const BoundedArray<T> &left,
						const BoundedArray<T> &right);

template<class T> BoundedArray<T>  operator - (const BoundedArray<T> &left,
						const BoundedArray<T> &right);

template<class T> Matrix<T>  operator + (const Matrix<T> &left,
					  const Matrix<T> &right);

template<class T> Matrix<T>  operator - (const Matrix<T> &left,
					  const Matrix<T> &right);

template<class T> Matrix<T>  operator * (const T &value,
						const Matrix<T> &right);

template<class T> Matrix<T>  operator * (const Matrix<T> &left,
					  const T &value);
   
  
#endif  // __MATRIX_H__ 

