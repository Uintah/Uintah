//  SymmetricMatrix.h 
//  class SymmetricMatrix
//    Type safe and access safe SymmetricMatrix data type.
//    Dense SymmetricMatrix class.
//    Uses templates.
//    Assumes all matrices starting index is 1 (one)
//    Features:
//      1.  Dense SymmetricMatrix format, can resize.
//      2.  Simple LU decomposition and solve routine
//      3.  Transpose operator
//      4.  Create an Identity SymmetricMatrix (square SymmetricMatrix only)
//      5.  SymmetricMatrix * SymmetricMatrix operation
//      6.  SymmetricMatrix * boundedArray operation (and visa versa)
//      7.  Dot product of two boundedArrays
//      8.  Add a scalar to a boundedArray
//      9.  Multiply each element of a boundedArray by a scalar
//     10.  Add two boundedArrays
//     11.  Subtract two boundedArrays
//     12.  Add two matrices together
//     13.  Subtract two matrices
//      Usage:
//   SymmetricMatrix<double> a_SymmetricMatrix();     // declare a double SymmetricMatrix - no size 
//                                       // specified
//   SymmetricMatrix<double> a_SymmetricMatrix(10,10); // declare a 10x10 SymmetricMatrix of doubles
//   SymmetricMatrix<double> a_SymmetricMatrix(10,10,2.0); // declare a 10x10 SymmetricMatrix of 
//                                            // doubles initialized to 2.0   
//   SymmetricMatrix<double> b_SymmetricMatrix(a_SymmetricMatrix); // copy the contents of a_SymmetricMatrix
//                                         // to b_SymmetricMatrix;
//   a_SymmetricMatrix.~SymmetricMatrix();           //  destroy a SymmetricMatrix
//   SymmetricMatrix<double> b_mat = a_mat; //  set b_mat equal to a_mat
//   a_SymmetricMatrix.numerRows();       //  return number of rows of a_SymmetricMatrix
//   a_SymmetricMatrix.numberColumns();   //  return number of columns of a_SymmetricMatrix //        double b = a_mat[i][j];    //  return the i,j entry of a_mat
//        a_mat[i][j] = 5.0;         //  set the i,j entry of a_mat to 5.0
//        a_SymmetricMatrix.Identity();       //  set a_SymmetricMatrix to the identity SymmetricMatrix   //        a_SymmetricMatrix.Zero();           //  set each element to be zero
//        a_SymmetricMatrix.Transpose();      //  return the transpose of the SymmetricMatrix
//        a_SymmetricMatrix.LUdecomp(row_perm,determ);  // Perform the LUdecomposition
//        a_SymmetricMatrix.LUbacksub(row_perm,b_vec); // Perform the LUbacksubstitution
//        a_SymmetricMatrix.Solve(b_vec);  // Solve Ax = b_vec, b_vec holds solution
//        new_mat = a_mat * b_mat;  // Perform SymmetricMatrix * SymmetricMatrix multiplication
//        new_array = a_mat * b_array;  // Perform SymmetricMatrix * array multiply
//        new_array = b_array * a_mat;  // Perform array * SymmetricMatrix multiply
//        result = a_array * b_array;  // Dot product of two arrays
//        new_array = scalar + a_array; // add scalar to each value of array
//        new_array = a_array + scalar; // add scalar to each value of array
//        new_array = scalar * a_array; // multiply all array elems by scalar
//        new_array = a_array + b_array; // add two arrays together
//        new_array = a_array - b_array; // subtract two arrays
//        new_mat = a_mat + b_mat;  // add two matrices 
//        new_mat = a_mat - b_mat;  // subtract two matrices


#ifndef __SYMMETRICMATRIX_H__
#define __SYMMETRICMATRIX_H__


#include <assert.h>
#include "BoundedArray.h"


template<class T> class Matrix;


template <class T> class SymmetricMatrix {
 protected:
  // data areas
      BoundedArray<BoundedArray<T> *> rows;
             
  
  
 public:
  // constructors and destructor
       SymmetricMatrix();
       SymmetricMatrix( int numberOfRows,  int numberOfColumns);
       SymmetricMatrix( int numberOfRows,  int numberOfColumn,
	      const T &initialValue);
  // copy constructor
       SymmetricMatrix(const SymmetricMatrix<T> &source); 
  // destructor 
       virtual ~SymmetricMatrix(); 

  // Assignment operator
       SymmetricMatrix<T>  & operator = (const SymmetricMatrix<T> &source);
       SymmetricMatrix<T>  & operator = (const Matrix<T> &source);

  // access to elements via subscript
       BoundedArray<T> & operator[](int index);
       BoundedArray<T> operator[](int index) const;
  
 
 
  // Dimensions of SymmetricMatrix
       int numberRows() const;
       int numberColumns() const;

  
  // Zero out a SymmetricMatrix
       void Zero();

  // SymmetricMatrix Transpose();
       SymmetricMatrix<T>  Transpose();

  // Trace of the Matrix - sum of diagonal terms
       T Trace();

 

};
					     
#include "Matrix.h"					    

// Non-member function declarations


template<class T> SymmetricMatrix<T>  operator * (const SymmetricMatrix<T> &left,
					  const T &value);

template<class T> SymmetricMatrix<T>  operator * (const T &value,
						const SymmetricMatrix<T> &right);


template<class T>  Matrix<T>  operator * (const SymmetricMatrix<T> &left, 
					const SymmetricMatrix<T> &right);


template<class T> BoundedArray<T>  operator * (const SymmetricMatrix<T> &left,
						const BoundedArray<T> &right);

template<class T> BoundedArray<T>  operator * (const BoundedArray<T> &left,
						const SymmetricMatrix<T> &right);

template<class T> SymmetricMatrix<T>  operator + 
(const SymmetricMatrix<T> &left, const SymmetricMatrix<T> &right);

template<class T> SymmetricMatrix<T>  operator - (const SymmetricMatrix<T> &left,
					  const SymmetricMatrix<T> &right);

#include "SymmetricMatrix.cc"
  
#endif  // __SYMMETRICMATRIX_H__ 

