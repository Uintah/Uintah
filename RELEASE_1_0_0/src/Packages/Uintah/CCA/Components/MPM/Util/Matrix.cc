#ifndef __Matrix_cc__
#define __Matrix_cc__

//  Matrix.cc
//  class matrix
//    Type safe and access safe matrix data type.
//    Dense matrix class.
//    Uses templates.
//    Assumes all matrices starting index is 1 (one)
//    Features:
//      1.  Dense matrix format, can resize.
//      2.  Simple LU decomposition and solve routine
//      3.  Transpose operator
//      4.  Create an Identity matrix (square matrix only)
//      5.  Matrix * matrix operation
//      6.  Matrix * boundedArray operation (and visa versa)
//      7.  Dot product of two boundedArrays
//      8.  Add a scalar to a boundedArray
//      9.  Multiply each element of a boundedArray by a scalar
//     10.  Add two boundedArrays
//     11.  Subtract two boundedArrays
//     12.  Add two matrices together
//     13.  Subtract two matrices
//      Usage:
//        matrix<double> a_matrix();     // declare a double matrix - no size 
//                                       // specified
//        matrix<double> a_matrix(10,10); // declare a 10x10 matrix of doubles
//        matrix<double> a_matrix(10,10,2.0); // declare a 10x10 matrix of 
//                                            // doubles initialized to 2.0   
//        matrix<double> b_matrix(a_matrix); // copy the contents of a_matrix
//                                         // to b_matrix;
//        a_matrix.~matrix();           //  destroy a matrix
//        matrix<double> b_mat = a_mat; //  set b_mat equal to a_mat
//        a_matrix.numerRows();       //  return number of rows of a_matrix
//        a_matrix.numberColumns();   //  return number of columns of a_matrix //        double b = a_mat[i][j];    //  return the i,j entry of a_mat
//        a_mat[i][j] = 5.0;         //  set the i,j entry of a_mat to 5.0
//        a_matrix.Identity();       //  set a_matrix to the identity matrix   //        a_matrix.Zero();           //  set each element to be zero
//        a_matrix.Transpose();      //  return the transpose of the matrix
//        a_matrix.LUdecomp(row_perm,determ);  // Perform the LUdecomposition
//        a_matrix.LUbacksub(row_perm,b_vec); // Perform the LUbacksubstitution
//        a_matrix.Solve(b_vec);  // Solve Ax = b_vec, b_vec holds solution
//        new_mat = a_mat * b_mat;  // Perform matrix * matrix multiplication
//        new_array = a_mat * b_array;  // Perform matrix * array multiply
//        new_array = b_array * a_mat;  // Perform array * matrix multiply
//        result = a_array * b_array;  // Dot product of two arrays
//        new_array = scalar + a_array; // add scalar to each value of array
//        new_array = a_array + scalar; // add scalar to each value of array
//        new_array = scalar * a_array; // multiply all array elems by scalar
//        new_array = a_array + b_array; // add two arrays together
//        new_array = a_array - b_array; // subtract two arrays
//        new_mat = a_mat + b_mat;  // add two matrices 
//        new_mat = a_mat - b_mat;  // subtract two matrices
//        new_mat = a_mat * scalar;  // multiply all elements by scalar



#include "Matrix.h"
#include <Core/Malloc/Allocator.h>

template<class T> Matrix<T>::Matrix()
{
  // Create a new Matrix
  // No initialization

}

template<class T> Matrix<T>::Matrix( int numberOfRows,
				     int numberOfColumns,
				     const T &initialValue):
  rows(1,numberOfRows)
{
  // Create and initialize a new Matrix
  // allocate the space for the elements and
  // set each element to initialValue

 
  // now allocate each row of data

  for ( int i = 1; i <= numberOfRows; i++) {
    rows[i]= scinew BoundedArray<T>(1,numberOfColumns,initialValue);
    // Check that allocation was successful
    assert(rows[i] != 0);
  }
}

template<class T> Matrix<T>::Matrix( int numberOfRows,
				     int numberOfColumns):
  rows(1,numberOfRows)
{
  // Create and initialize a new Matrix
  // allocate the space for the elements and
  // set each element to initialValue


  // now allocate each row of data

  for (int i = 1; i <= numberOfRows; i++) {
    rows[i]= scinew BoundedArray<T>(1,numberOfColumns);
    // Check that allocation was successful
    assert(rows[i] != 0);
  }
}

template<class T> Matrix<T>::Matrix(const Matrix<T> &source)
{
  // copy constructor for the Matrix class

  // create and initialize a new Matrix
  // allocate the space for the elements

  rows = BoundedArray<BoundedArray<T> *>(1,source.numberRows());
 

  for ( int i = 1; i<= source.numberRows(); i++) {
    rows[i] = scinew BoundedArray<T>(1,source.numberColumns());
    assert(rows[i] != 0);
  }

  for (int i = 1; i<= source.numberRows(); i++) {
    for (int j = 1; j<= source.numberColumns(); j++) {
      (*this)[i][j] = source[i][j];
    }
  }

   
}


template <class T> Matrix<T>  & Matrix<T>::operator = 
(const Matrix<T> &source)
{
  // Assignment operator

 
  for (int i = 1; i<= this->numberRows(); i++) {
    BoundedArray<T> *p= rows[i];
    delete p;
    rows[i] = 0;
  }
  


  rows = BoundedArray<BoundedArray<T> *>(1,source.numberRows());

  
  for ( int i = 1; i<= source.numberRows(); i++) {
    rows[i] = scinew BoundedArray<T>(1,source.numberColumns());
    assert(rows[i] != 0);
  }
 
  for (int i = 1; i<= source.numberRows(); i++) {
    for (int j = 1; j<= source.numberColumns(); j++) {
      (*this)[i][j] = source[i][j];
    }
  }


  return (*this);
}

template <class T> Matrix<T>  & Matrix<T>::operator = 
(const SymmetricMatrix<T> &source)
{
  // Assignment operator

 
  for (int i = 1; i<= this->numberRows(); i++) {
    BoundedArray<T> *p= rows[i];
    delete p;
    rows[i] = 0;
  }
  


  rows = BoundedArray<BoundedArray<T> *>(1,source.numberRows());

  
  for ( int i = 1; i<= source.numberRows(); i++) {
    rows[i] = scinew BoundedArray<T>(1,source.numberColumns());
    assert(rows[i] != 0);
  }
 
  for (int i = 1; i<= source.numberRows(); i++) {
    for (int j = 1; j<= source.numberColumns(); j++) {
      if (j >= i)
	(*this)[i][j] = source[i][j];
      else
	(*this)[i][j] = source[j][i];
    }
  }


  return (*this);
}

template<class T> Matrix<T>::~Matrix()
{

  // destructor
  // delete all the row vectors


  unsigned int max = rows.length();
  
  
  for (unsigned int i = 1; i<=max; i++) {
    BoundedArray<T> *p = rows[i];
    delete p;
    rows[i] = 0;
  }
 
}

template <class T> void Matrix<T>::resize( int num_rows,  int num_columns)
{

  assert(num_columns >= 1 && num_rows >= 1);

  // Old dimensions
  int o_rows = numberRows();
  //int o_cols = numberColumns();

  // Resize the matrix to size rows by columns

  rows.setSize(num_rows);
  
  // If added more elements, must allocate columns

  for ( int i = 1; i <= num_rows; i++) {
    if (i > o_rows) {
      rows[i] = scinew BoundedArray<T>(1,num_columns,0.0);
      assert(rows[i] != 0);
    }
    rows[i]->setSize(num_columns);
  }      


}

template <class T> int Matrix<T>::numberRows() const
{
  // return the number of rows in the Matrix
  return rows.length();
}

template <class T> int Matrix<T>::numberColumns() const
{
  // return the number of columns in the Matrix
  // make sure there is a row 1
  assert(rows[1] != 0);

  // return the number of element in the row
  return rows[1]->length();
}

//template<class T> BoundedArray<T> & Matrix<T>::operator[] (int index)
//{
  // subscript a Matrix value
  // leading subscript in a Matrix expression
  // check that the index is valid

//  assert(rows[index] != 0);

  // return array value, use pointer dereference to get
  // reference to actual vector

//  return  *rows[index];

//}

//template<class T> BoundedArray<T>  Matrix<T>::operator[] (int index) const
//{
  // subscript a Matrix value
  // leading subscript in a Matrix expression
  // check that the index is valid

//  assert(rows[index] != 0);

  // return array value, use pointer dereference to get
  // reference to actual vector

//  return  *rows[index];

//}

template<class T> void Matrix<T>::Identity()
{
  // Create an identity Matrix
  // Valid for square matrices

  int nrows = this->numberRows();
  int ncols = this->numberColumns();

  assert(nrows == ncols);

  for (int i = 1; i<= nrows; i++) {
    for (int j= 1; j<= ncols; j++) {
      (*this)[i][j] = 0.;
      if (i == j) {
	(*this)[i][j] = 1.0;
      }
    }
  }
}


template<class T> void Matrix<T>::Zero()
{
  // Zero out a  Matrix

  int nrows = this->numberRows();
  int ncols = this->numberColumns();

  for (int i = 1; i<= nrows; i++) {
    for (int j= 1; j<= ncols; j++) {
      (*this)[i][j] = 0.;
    }
  }
}

template<class T> Matrix<T>  Matrix<T>::Transpose()
{
  // Return the transpose of a Matrix

  int nrows = this->numberRows();
  int ncols = this->numberColumns();

  Matrix<T> temp(ncols,nrows);

  for (int i = 1; i<= ncols; i++) {
    for (int j = 1; j<= nrows; j++) {
      temp[i][j] = (*this)[j][i];
    }
  }

  // Return result
  
  return temp;
 

}

template<class T> T  Matrix<T>::Trace()
{
  // Return the trace of a Matrix

  int nrows = this->numberRows();
  

  T temp;
  temp = 0;

  for (int i = 1; i<= nrows; i++) {
    (temp) += (*this)[i][i];
    
  }

  // Return result
  
  return temp;
 

}


template<class T> void Matrix<T>::LUdecomp(BoundedArray<int> &row_perm, 
					     T &determ)
{
  // Performs the LU decomposition of a Matrix.
  // Original Matrix (this) is replaced by lower and upper triangular Matrix
  // row permuations are returned in array row_perm.  determ is either
  // +- 1 depending on the number of row interchanges.  It is used for
  // computing the determinant in other routines.
  // Uses Crout's method with partitial pivoting described in NR.
  // Use with LUbacksubstition

  // Check that the Matrix is square


  int num_rows = this->numberRows();
  int num_cols = this->numberColumns();

  assert(num_rows == num_cols);

  
  // No row interchanges yet
  determ = 1.0; 
  
  // Scale values for each row
  BoundedArray<T> row_scale(1,num_rows); 

 
  // Determine the scaling factors
  T big;
  for (int i = 1; i<=num_rows; i++) {
    big = ((*this)[i]).largest_mag();
    assert(big != 0.0);
    row_scale[i] = 1.0/big;
  }
 
  // Loop over the columns in Crout's method
  int i,j,k,imax = 0;
  T sum,abs_sum,fig_merit;
  for ( j = 1; j <= num_rows; j++) {
    // Do beta[i][j] = a[i][j] - Sum(k = 1,i - 1)(alpha[i][k]*beta[k][j])
    for ( i = 1; i < j; i++) {
      sum = (*this)[i][j];
      for ( k = 1; k < i; k++) {
	sum -= (*this)[i][k]*(*this)[k][j];
      }
      (*this)[i][j] = sum;
    }

    // Search for the largest pivot element
    big = 0.0;
    for ( i = j; i<=num_rows; i++) {
      sum = (*this)[i][j];
      for ( k = 1; k < j; k++) {
	sum -= (*this)[i][k]*(*this)[k][j];
      }
      (*this)[i][j] = sum;
      // find the absolute value of sum
      if (sum >= 0) {
	abs_sum = sum;
      }
      else {
	abs_sum = -(sum);
      }  // Done with absolute value of sum
      if ( (fig_merit = row_scale[i]*abs_sum) >= big) {
	big = fig_merit;
	 imax = i;
      }
    } // Done with i loop

    if (j != imax) {
      // Interchange rows
      for ( k = 1; k <= num_rows; k++) {
	fig_merit = (*this)[imax][k];
	(*this)[imax][k] = (*this)[j][k];
	(*this)[j][k] = fig_merit;
      }
      determ = -(determ);  // Change the parity of determinate
      row_scale[imax] = row_scale[j];  // Interchange scale factor;
    
    }
    
      row_perm[j] = imax;
      assert((*this)[j][j] != 0.0);
     
      if (j != num_rows) {  // Divide by the pivot element
	fig_merit = 1.0/((*this)[j][j]);
	for ( i = j+1; i<=num_rows; i++) {
	  (*this)[i][j] *= fig_merit;
	}
      }

  } // End of looping over columns

 

}

template<class T> void Matrix<T>::LUbacksub(BoundedArray<int> &row_perm,
					    BoundedArray<T> &b)
{
  // Solves the set of linear equations A x = b, given the LUdecomp of A
  // and the permutation array row_perm.  The solution array is returned
  // in the right hand side array, b.

  int num_rows = this->numberRows();
  int i,ii=0,ip,j;
  T sum;

  for (i = 1; i<= num_rows; i++) {
    ip = row_perm[i];
    sum = b[ip];

    b[ip] = b[i];
    if (ii) {
      for (j = ii; j <= i-1; j++) {
	sum -= (*this)[i][j] * b[j];
      }
    }
    else if (sum) {
      ii = i;
    }
    b[i] = sum;
  }
  
  for (i = num_rows; i>=1; i--) {
    sum = b[i];
    for (j = i+1; j<=num_rows; j++) {
      sum -= (*this)[i][j] * b[j];
    }
    b[i] = sum/(*this)[i][i];
  }
  

}

template<class T> void Matrix<T>::Solve(BoundedArray<T> &b)
{
  // Solves the Matrix equation Ax = b using LUdecomp and LUbacksub
  // and returns the solution vector in the b vector.
  // The A Matrix is modified to have the LU decomposition in it.


  BoundedArray<int> row_perm(1,this->numberRows());
  T determ;
  
  (*this).LUdecomp(row_perm,determ);
  (*this).LUbacksub(row_perm,b);

}

template<class T> T Matrix<T>::Determinant()
{
  // Computes the determinant of the matrix using the LUdecomposition.
  // First copies the matrix and operates on the copy.


  if (this->numberRows() == 3)
    return Determinant3();

  Matrix<T> copy = *this;

  BoundedArray<int> row_perm(1,copy.numberRows());
  T determ;
  T determinant = 1.;

  copy.LUdecomp(row_perm,determ);

  for (int i = 1; i<= copy.numberRows(); i++) {
    determinant *=  copy[i][i];
  }

  return determinant;
}

template<class T> T Matrix<T>::Determinant3()
{
  // Return the determinant of a 3x3 matrix

  int nrows = this->numberRows();
  assert(nrows == 3);
  
  T temp;
  temp = 0;

  temp = (*this)[1][1]*(*this)[2][2]*(*this)[3][3] + 
    (*this)[1][2]*(*this)[2][3]*(*this)[3][1] +
    (*this)[1][3]*(*this)[2][1]*(*this)[3][2] -
    (*this)[1][3]*(*this)[2][2]*(*this)[3][1] -
    (*this)[1][2]*(*this)[2][1]*(*this)[3][3] -
    (*this)[1][1]*(*this)[2][3]*(*this)[3][2];

  // return result

  return temp;
}



// Non-member functions for Matrix, SymmetricMatrix, BoundedArray

// template<class T>  Matrix<T>  operator * (const Matrix<T> &left, 
// 					 const Matrix<T> &right)
// {
//   // perform Matrix multiplication of left by right
//   // first get dimensions of matrices

//   int n = left.numberRows();
//   int m = left.numberColumns();
//   int p = right.numberColumns();

//   // Check that they are compatible
//   assert(m == right.numberRows());

//   // allocate space for the result
//   Matrix<T> result(n,p,0.0);
 

   
//   // fill in the values
//   for (int i = 1; i<= n; i++) {
//     for (int j = 1; j<= p; j++) {
//       (result)[i][j] = 0.;
//       for (int k = 1; k <= m; k++) {
//     	(result)[i][j] +=  left[i][k] * right[k][j];
//       }
//     }
//   }

//   // return the result
//   return result;
// }


template<class T> BoundedArray<T> operator * (const Matrix<T> &left,
						const BoundedArray<T> &right)
{
  // Perform Matrix times vector multiplication to get another vector.
  // Vectors are represented by BoundedArrays.  The return is a BoundedArray.
 
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
      result[i] += left[i][j]*right[j];
    }
  }

  // return the result
  return result;

}

using namespace SCIRun;
template<class T> BoundedArray<Vector> operator * (const Matrix<T> &left,
                                              const BoundedArray<Vector> &right)
{
  // Perform Matrix times BoundedArray of vectors multiplication to get
  // another BoundedArray of vectors.

  int mat_row = left.numberRows();
  int mat_col = left.numberColumns();
  int array_row = right.length();

  // Check that they can be multiplied
  assert(mat_col == array_row);

  Vector zero(0.0,0.0,0.0);

  // Create space for the result
  BoundedArray<Vector> result(1,mat_row,zero);

  // Fill in the values
  for (int i = 1; i<= mat_row; i++) {
    for (int j = 1; j<= mat_col; j++) {
      result[i] += right[j]*left[i][j];
    }
  }

  // return the result
  return result;

}


template<class T> BoundedArray<T> operator * (const BoundedArray<T> &left,
						const Matrix<T> &right)
{
  // Perform vector times Matrix  multiplication to get another vector.
  // Vectors are represented by BoundedArrays.  The return is a BoundedArray.
 
  int mat_row = right.numberRows();
  int mat_col = right.numberColumns();
  int array_col = left.length();

  // Check that they can be multiplied
  assert(mat_row == array_col);

  // Create space for the result
  BoundedArray<T> result (1,mat_col,0.0);

  // Fill in the values
  for (int i = 1; i<= mat_col; i++) {
    for (int j = 1; j<= mat_row; j++) {
      result[i] += left[j]*right[j][i];
    }
  }

  // return the result
  return result;

}

template<class T> T operator * (const BoundedArray<T> &left,
				  const BoundedArray<T> &right)
{
  // Perform the dot product of two vectors represented as
  // BoundedArrays, returns a scalar value.

  // Check that the dimensions are the same

  assert(left.length() == right.length() );
  T result ;
  result = 0;
  
  for (unsigned int i = 1; i<= left.length(); i++) {
    result += left[i] * right[i];
  }

  // Return the result

  return result;
}
    
template<class T> BoundedArray<T>  operator + (const T &value,
						const BoundedArray<T> &right)
{
  // Perform scalar value + vector.  Each component of the vector has
  // the scalar value added to it.

  BoundedArray<T> result(right.lowerBound(),right.upperBound(),0);
  
  for (int i = right.lowerBound(); i<= right.upperBound(); i++) {
    result[i] = value + right[i];
  }
  
  // Return the result
  return result;
}

template<class T> BoundedArray<T>  operator + (const BoundedArray<T> &left,
						const T &value)
{
  // Perform scalar value + vector.  Each component of the vector has
  // the scalar value added to it.

  BoundedArray<T> result(left.lowerBound(),left.upperBound(),0);
  
  for (int i = left.lowerBound(); i<= left.upperBound(); i++) {
    result[i] = value + left[i];
  }
  
  // Return the result
  return result;
}

template<class T> BoundedArray<T>  operator * (const T &value,
						const BoundedArray<T> &right)
{
  // Perform scalar value times vector.  Each component of the vector has
  // the scalar value multiplied by it.

  BoundedArray<T> result(right.lowerBound(),right.upperBound(),0);
  
  for (int i = right.lowerBound(); i<= right.upperBound(); i++) {
    result[i] = value * right[i];
  }
  
  // Return the result
  return result;
}

template<class T> BoundedArray<T>  operator / (const BoundedArray<T> &right,
					       const T &value)
{
  // Perform vector divided by scalar scalar value.  Each component of 
  // the vector has the scalar value divided by it.

  BoundedArray<T> result(right.lowerBound(),right.upperBound(),0);
  
  for (int i = right.lowerBound(); i<= right.upperBound(); i++) {
    result[i] = right[i]/value;
  }
  
  // Return the result
  return result;
}
    

template<class T> BoundedArray<T>  operator + (const BoundedArray<T> &left,
						const BoundedArray<T> &right)
{
  // Perform the addition of two vectors represented as BoundedArrays.

  // Check that the dimensions are the same so they can be added and there
  // lower bounds are the same.
  
  assert(left.upperBound() == right.upperBound());
  assert(left.lowerBound() == right.lowerBound());
  
  // Create space for the result
  BoundedArray<T> result(left.lowerBound(),left.upperBound(),0.0);

  // Fill in the values
  for (int i = left.lowerBound(); i<= left.upperBound(); i++) {
    result[i] = left[i] + right[i];
  }

  // Return the result
  return result;
}

template<class T> BoundedArray<T>  operator - (const BoundedArray<T> &left,
						const BoundedArray<T> &right)
{
  // Perform the subtraction of two vectors represented as BoundedArrays.

 

  // Check that the dimensions are the same so they can be added.
  assert(left.upperBound() == right.upperBound());
  assert(left.lowerBound() == right.lowerBound());
    
  // Create space for the result
  BoundedArray<T> result(left.lowerBound(),left.upperBound(),0.0);

  // Fill in the values
  for (int i = left.lowerBound(); i<= left.upperBound(); i++) {
    result[i] = left[i] - right[i];
  }

  // Return the result
  return result;
}

template<class T> Matrix<T>  operator + (const Matrix<T> &left,
					  const Matrix<T> &right)
{
  // Perform the addition of two matrices.

  int left_row = left.numberRows();
  int left_col = left.numberColumns();
  int right_row = right.numberRows();
  int right_col = right.numberColumns();

  // Check that the dimensions are the same so they can be added.

  assert(left_row == right_row && left_col == right_col);
  
  // Create space for the result
  Matrix<T> result(left_row,right_row,0.0);

  // Fill in the values
  for (int i = 1; i<= left_row; i++) {
    result[i] = left[i] + right[i];
  }

  // Return the result
  return result;
}

template<class T> Matrix<T>  operator - (const Matrix<T> &left,
					  const Matrix<T> &right)
{
  // Perform the subtraction of two matrices.

  int left_row = left.numberRows();
  int left_col = left.numberColumns();
  int right_row = right.numberRows();
  int right_col = right.numberColumns();

  // Check that the dimensions are the same so they can be subtracted.

  assert(left_row == right_row && left_col == right_col);
  
  // Create space for the result
  Matrix<T> result(left_row,right_row,0.0);

  // Fill in the values
  for (int i = 1; i<= left_row; i++) {
    result[i] = left[i] - right[i];
  }

  // Return the result
  return result;
}

template<class T> Matrix<T>  operator * (const T &value,
						const Matrix<T> &right)
{
  // Perform scalar value times Matrix.  Each component of the Matrix has
  // the scalar value multiplied by it.

  Matrix<T> result(right.numberRows(),right.numberColumns(),0.);
  
  for ( int i = 1; i<= right.numberRows(); i++) {
    result[i] = value * right[i];
  }
  
  // Return the result
  return result;
}

template<class T> Matrix<T>  operator * (const Matrix<T> &left,
					  const T &value)
{
  // Perform Matrix times scalar value.  Each component of the Matrix has
  // the scalar value multiplied by it.

  Matrix<T> result(left.numberRows(),left.numberColumns(),0);
  
  for ( int i = 1; i<= left.numberRows(); i++) {
    result[i] = value * left[i];
  }
  
  // Return the result
  return result;
}

#endif // __Matrix_cc__

