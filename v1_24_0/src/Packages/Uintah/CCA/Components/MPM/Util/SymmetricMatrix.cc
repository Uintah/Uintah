#ifndef __SymmetricMatrix_cc__
#define __SymmetricMatrix_cc__

//  SymmetricMatrix.cc 
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
//      6.  SymmetricMatrix * BoundedArray operation (and visa versa)
//      7.  Dot product of two BoundedArrays
//      8.  Add a scalar to a BoundedArray
//      9.  Multiply each element of a BoundedArray by a scalar
//     10.  Add two BoundedArrays
//     11.  Subtract two BoundedArrays
//     12.  Add two matrices together
//     13.  Subtract two matrices
//      Usage:
//        SymmetricMatrix<double> a_SymmetricMatrix();     // declare a double SymmetricMatrix - no size 
//                                       // specified
//        SymmetricMatrix<double> a_SymmetricMatrix(10,10); // declare a 10x10 SymmetricMatrix of doubles
//        SymmetricMatrix<double> a_SymmetricMatrix(10,10,2.0); // declare a 10x10 SymmetricMatrix of 
//                                            // doubles initialized to 2.0   
//        SymmetricMatrix<double> b_SymmetricMatrix(a_SymmetricMatrix); // copy the contents of a_SymmetricMatrix
//                                         // to b_SymmetricMatrix;
//        a_SymmetricMatrix.~SymmetricMatrix();           //  destroy a SymmetricMatrix
//        SymmetricMatrix<double> b_mat = a_mat; //  set b_mat equal to a_mat
//        a_SymmetricMatrix.numerRows();       //  return number of rows of a_SymmetricMatrix
//        a_SymmetricMatrix.numberColumns();   //  return number of columns of a_SymmetricMatrix //        double b = a_mat[i][j];    //  return the i,j entry of a_mat
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


#include "SymmetricMatrix.h"
#include <Core/Malloc/Allocator.h>
				   
template<class T> SymmetricMatrix<T>::SymmetricMatrix()
{
  // Create a new SymmetricMatrix
  // No initialization

}

template<class T> SymmetricMatrix<T>::SymmetricMatrix( int numberOfRows,
				     int numberOfColumns,
				     const T &initialValue):
  rows(1,numberOfRows)
{
  // Create and initialize a new SymmetricMatrix
  // allocate the space for the elements and
  // set each element to initialValue

  // This only applies to square matrices, so check that numberOfRows
  // is equal to numberOfColumns.

  assert(numberOfRows == numberOfColumns);

 
  // now allocate each row of data

  for ( int i = 1; i <= numberOfRows; i++) {
    rows[i]= scinew BoundedArray<T>(i,numberOfColumns,initialValue);
    // Check that allocation was successful
    assert(rows[i] != 0);
  }
}

template<class T> SymmetricMatrix<T>::SymmetricMatrix( int numberOfRows,
				     int numberOfColumns):
  rows(1,numberOfRows)
{
  // Create and initialize a new SymmetricMatrix
  // allocate the space for the elements.
   
  // This only applies to square matrices, so check that numberOfRows
  // is equal to numberOfColumns.

  assert(numberOfRows == numberOfColumns);


  // now allocate each row of data

  for (int i = 1; i <= numberOfRows; i++) {
    rows[i]= scinew BoundedArray<T>(i,numberOfColumns);
    // Check that allocation was successful
    assert(rows[i] != 0);
  }
}

template<class T> SymmetricMatrix<T>::SymmetricMatrix(
  const SymmetricMatrix<T> &source)
{
  // copy constructor for the SymmetricMatrix class

  // create and initialize a new SymmetricMatrix
  // allocate the space for the elements

  rows = BoundedArray<BoundedArray<T> *>(1,source.numberRows());
 

  for ( int i = 1; i<= source.numberRows(); i++) {
    rows[i] = scinew BoundedArray<T>(i,source.numberColumns());
    assert(rows[i] != 0);
  }

  for (int i = 1; i<= source.numberRows(); i++) {
    for (int j = i; j<= source.numberColumns(); j++) {
      (*this)[i][j] = source[i][j];
    }
  }

   
}


template <class T> SymmetricMatrix<T>  & SymmetricMatrix<T>::operator = (const SymmetricMatrix<T> &source)
{
  // Assignment operator

 
  for (int i = 1; i<= this->numberRows(); i++) {
    BoundedArray<T> *p= rows[i];
    delete p;
    rows[i] = 0;
  }
  


  rows = BoundedArray<BoundedArray<T> *>(1,source.numberRows());

  
  for ( int i = 1; i<= source.numberRows(); i++) {
    rows[i] = scinew BoundedArray<T>(i,source.numberColumns());
    assert(rows[i] != 0);
  }
 
  for (int i = 1; i<= source.numberRows(); i++) {
    for (int j = i; j<= source.numberColumns(); j++) {
      (*this)[i][j] = source[i][j];
    }
  }


  return (*this);
}

template <class T> SymmetricMatrix<T>  & SymmetricMatrix<T>::operator = (const Matrix<T> &source)
{
  // Assignment operator, set a Symmetric Matrix = Matrix

 
  for (int i = 1; i<= this->numberRows(); i++) {
    BoundedArray<T> *p= rows[i];
    delete p;
    rows[i] = 0;
  }
  


  rows = BoundedArray<BoundedArray<T> *>(1,source.numberRows());

  
  for ( int i = 1; i<= source.numberRows(); i++) {
    rows[i] = scinew BoundedArray<T>(i,source.numberColumns());
    assert(rows[i] != 0);
  }
 
  for (int i = 1; i<= source.numberRows(); i++) {
    for (int j = i; j<= source.numberColumns(); j++) {
      (*this)[i][j] = source[i][j];
    }
  }


  return (*this);

}

template<class T> SymmetricMatrix<T>::~SymmetricMatrix()
{

  // destructor
  // delete all the row vectors


  unsigned int max = rows.length();
  
  
  for (unsigned int i = 1; i<=max; i++) {
    BoundedArray<T> *p = rows[i];
    delete p;
    rows[i] = 0;
  }
 
  rows.~BoundedArray();
}

template <class T> int SymmetricMatrix<T>::numberRows() const
{
  // return the number of rows in the SymmetricMatrix
  return rows.length();
}

template <class T> int SymmetricMatrix<T>::numberColumns() const
{
  // return the number of columns in the SymmetricMatrix
  // make sure there is a row 1
  assert(rows[1] != 0);

  // return the number of element in the row
  return rows[1]->length();
}



template<class T> BoundedArray<T> & SymmetricMatrix<T>::operator[] (int index)
{
  // subscript a SymmetricMatrix value
  // leading subscript in a SymmetricMatrix expression
  // check that the index is valid

  assert(rows[index] != 0);

  // return array value, use pointer dereference to get
  // reference to actual vector

  return  *rows[index];

}

template<class T> BoundedArray<T>  SymmetricMatrix<T>::operator[] (int index) const
{
  // subscript a SymmetricMatrix value
  // leading subscript in a SymmetricMatrix expression
  // check that the index is valid

  assert(rows[index] != 0);

  // return array value, use pointer dereference to get
  // reference to actual vector

  return  *rows[index];

}


    


template<class T> void SymmetricMatrix<T>::Zero()
{
  // Zero out a  SymmetricMatrix

  int nrows = this->numberRows();
  int ncols = this->numberColumns();

  for (int i = 1; i<= nrows; i++) {
    for (int j= i; j<= ncols; j++) {
      (*this)[i][j] = 0.;
    }
  }
}

template<class T> SymmetricMatrix<T>  SymmetricMatrix<T>::Transpose()
{
  // Return the transpose of a SymmetricMatrix

  int nrows = this->numberRows();
  int ncols = this->numberColumns();

  SymmetricMatrix<T> temp(ncols,nrows);

  for (int i = 1; i<= ncols; i++) {
    for (int j = i; j<= nrows; j++) {
      temp[i][j] = (*this)[j][i];
    }
  }

  // Return result
  
  return temp;
 

}

template<class T> T  SymmetricMatrix<T>::Trace()
{
  // Return the trace of a SymmetricMatrix

  int nrows = this->numberRows();
  

  T temp;
  temp = 0;

  for (int i = 1; i<= nrows; i++) {
    temp += (*this)[i][i];
    
  }

  // Return result
  
  return temp;
 

}

// Non-member functions

template<class T> SymmetricMatrix<T>  operator * (const SymmetricMatrix<T> &left,
					  const T &value)
{
  // Perform Matrix times scalar value.  Each component of the Matrix has
  // the scalar value multiplied by it.

  SymmetricMatrix<T> result(left.numberRows(),left.numberColumns(),0);
  
  for ( int i = 1; i<= left.numberRows(); i++) {
    result[i] = value * left[i];
  }
  
  // Return the result
  return result;
}

template<class T> SymmetricMatrix<T>  operator * (const T &value,
						const SymmetricMatrix<T> &right)
{
  // Perform scalar value times Matrix.  Each component of the Matrix has
  // the scalar value multiplied by it.

  SymmetricMatrix<T> result(right.numberRows(),right.numberColumns(),0);
  
  for (int i = 1; i<= right.numberRows(); i++) {
    result[i] = value * right[i];
  }
  
  // Return the result
  return result;
}


template<class T>  Matrix<T>  operator * (const SymmetricMatrix<T> &left, 
					const SymmetricMatrix<T> &right)
{
  // perform Symmetric Matrix multiplication of left by right
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
      result[i][j] = 0.;
      for (int k = 1; k <= m; k++) {
	if (k < j && k < i) {
	  result[i][j] +=  left[k][i] * right[k][j];
	}
	if (k < j && k >= i) {
	  result[i][j] +=  left[i][k] * right[k][j];
	}
	if (k >= j && k < i) {
	  result[i][j] +=  left[k][i] * right[j][k];
	}
	 
	if (k >= j && k >= i) {
	  result[i][j] +=  left[i][k] * right[j][k];
	}
      }

    }
  }

  // return the result
  return result;
}



template<class T> BoundedArray<T>  operator * (const SymmetricMatrix<T> &left,
						const BoundedArray<T> &right)
{
  // Perform SymmetricMatrix times vector multiplication to get 
  // another vector. Vectors are represented by BoundedArrays.  
  // The return is a BoundedArray.
 
  int mat_row = left.numberRows();
  int mat_col = left.numberColumns();
  int array_row = right.length();

  // Check that they can be multiplied
  assert(mat_col == array_row);

  // Create space for the result
  BoundedArray<T> result(1,mat_row,0.0);

  // Fill in the values
  for (int i = 1; i<= mat_row; i++) {
    for (int j = 1; j<= mat_col; j++) {
      if (j < i) {
	result[i] += left[j][i]*right[j];
      }
      else {
	result[i] += left[i][j]*right[j];
      }
    }
  }

  // return the result
  return result;

}

template<class T> BoundedArray<T>  operator * (const BoundedArray<T> &left,
						const SymmetricMatrix<T> &right)
{
  // Perform vector times SymmetricMatrix  multiplication to 
  // get another vector. Vectors are represented by BoundedArrays.  
  // The return is a BoundedArray.
 
  int mat_row = right.numberRows();
  int mat_col = right.numberColumns();
  int array_col = left.length();

  // Check that they can be multiplied
  assert(mat_row == array_col);

  // Create space for the result
  BoundedArray<T> result(1,mat_col,0.0);

  // Fill in the values
  for (int i = 1; i<= mat_col; i++) {
    for (int j = 1; j<= mat_row; j++) {
      if ( j > i ) {
	result[i] += left[j]*right[i][j];
      }
      else {
	result[i] += left[j]*right[j][i];
      }
    }
  }

  // return the result
  return result;

}



template<class T> SymmetricMatrix<T>  operator + 
(const SymmetricMatrix<T> &left, const SymmetricMatrix<T> &right)
{
  // Perform the addition of two Symmetric matrices.

  int left_row = left.numberRows();
  int left_col = left.numberColumns();
  int right_row = right.numberRows();
  int right_col = right.numberColumns();

  // Check that the dimensions are the same so they can be added.

  assert(left_row == right_row && left_col == right_col);
  
  // Create space for the result
  SymmetricMatrix<T> result(left_row,right_row,0.0);

  // Fill in the values
  for (int i = 1; i<= left_row; i++) {
    result[i] = left[i] + right[i];
  }

  // Return the result
  return result;
}

template<class T> SymmetricMatrix<T>  operator - (const SymmetricMatrix<T> &left,
					  const SymmetricMatrix<T> &right)
{
  // Perform the subtraction of two matrices.

  int left_row = left.numberRows();
  int left_col = left.numberColumns();
  int right_row = right.numberRows();
  int right_col = right.numberColumns();

  // Check that the dimensions are the same so they can be subtracted.

  assert(left_row == right_row && left_col == right_col);
  
  // Create space for the result
  SymmetricMatrix<T> result(left_row,right_row,0.0);

  // Fill in the values
  for (int i = 1; i<= left_row; i++) {
    result[i] = left[i] - right[i];
  }

  // Return the result
  return result;
}


#endif //__SymmetricMatrix_cc__
  
 
