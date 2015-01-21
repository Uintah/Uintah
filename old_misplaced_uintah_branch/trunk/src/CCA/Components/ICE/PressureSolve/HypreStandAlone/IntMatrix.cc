#include "IntMatrix.h"
#include <iostream>
using namespace std;

IntMatrix::IntMatrix(const Counter rows, 
                     const Counter cols)
  : rows(rows), cols(cols)
{
  assert(rows>0);
  assert(cols>0);
  mat = scinew int[rows*cols];
}

IntMatrix::~IntMatrix(void)
{
  if (mat) delete[] mat;
}

void
IntMatrix::transpose(const IntMatrix& a)
{
  assert(cols == a.rows);
  assert(rows == a.cols);
  // Cache-coherent on writes.  We could do better by blocking, but
  // This should be good enough for small matrices, which is what this
  // class is intended for
  for(Counter row=0;row<rows; row++){
    for(Counter col=0;col<cols; col++){
      this->operator()(row,col) = a(col,row);
    }
  }
}

void
IntMatrix::identity(void)
{
  assert(rows == cols);
  zero();
  for(Counter i=0;i<rows;i++)
    this->operator()(i,i) = 1;
}

void
IntMatrix::zero(void)
{
  for(Counter i = 0; i < rows*cols; i++) {
    mat[i] = 0;
  }
}

void
IntMatrix::copy(const IntMatrix& a)
{
  assert(rows == a.rows);
  assert(cols == a.cols);
  //Counter size=rows*cols;
  for(Counter i=0;i<rows;i++){
    for(Counter j=0;j<cols;j++)
      this->operator()(i,j) = a(i,j);
  }
}

std::ostream&
operator << (std::ostream& os, const IntMatrix& a)
  // Write the matrix to the stream os.
{
  for(Counter i=0;i<a.rows;i++){
    os << i << ":";
    for(Counter j=0;j<a.cols;j++){
      os << "\t" << a(i,j);
    }
    os << "\n";
  }
  return os;
}

IntMatrix&
IntMatrix::operator = (const IntMatrix& other)
{
  rows = other.rows;
  cols = other.cols;
  //Counter size=rows*cols;
  if (mat) {
    delete[] mat;
    mat = scinew int[rows*cols];
  }
  for(Counter i = 0; i < rows*cols; i++) {
    mat[i] = other.mat[i];
  }
  return *this;
}

IntMatrix::IntMatrix(const IntMatrix& other)
{
  rows = other.rows;
  cols = other.cols;
  //Counter size=rows*cols;
  if (mat) {
    delete[] mat;
    mat = scinew int[rows*cols];
  }
  for(Counter i = 0; i < rows*cols; i++) {
    mat[i] = other.mat[i];
  }
}
