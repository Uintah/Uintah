#ifndef __INTMATRIX_H__
#define __INTMATRIX_H__

#include "mydriver.h"
#include <vector>

using namespace std;
class IntMatrix {
 public:
  IntMatrix(int rows, int cols);
  ~IntMatrix(void);
  
  int numRows(void) const {
    return rows;
  }
  int numCols(void) const {
    return cols;
  }
  int& operator()(int r, int c) {
    return mat[r*cols + c];
  }
  int operator()(int r, int c) const {
    return mat[r*cols + c];
  }

  void resize(int r, int c) {
    if (mat) delete[] mat;
    rows = r;
    cols = c;
    mat = new int[rows*cols];
    zero();
  }

  void transpose(const IntMatrix& transpose);
  void multiply(const vector<int>& b, vector<int>& X) const;
  void multiply(const int* b, int* X) const;
  void multiply(const IntMatrix& a, const IntMatrix& b);
  void multiply(int s);
  void identity(void);
  void zero(void);
  void copy(const IntMatrix& copy);
  void print(std::ostream& out);
  IntMatrix& operator = (const IntMatrix&);
  IntMatrix(const IntMatrix&);

 private:    
  int* mat;
  int  rows, cols;
};

#endif // __INTMATRIX_H__
