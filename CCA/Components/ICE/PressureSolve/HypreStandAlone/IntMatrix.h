#ifndef __INTMATRIX_H__
#define __INTMATRIX_H__

#include "mydriver.h"
#include <vector>

using namespace std;

class IntMatrix {
 public:
  IntMatrix(const Counter rows, const Counter cols);
  ~IntMatrix(void);
  
  Counter numRows(void) const {
    return rows;
  }
  Counter numCols(void) const {
    return cols;
  }
  int& operator()(const Counter r, const Counter c) {
    return mat[r*cols + c];
  }
  Counter operator()(const Counter r, const Counter c) const {
    return mat[r*cols + c];
  }

  void resize(const Counter r, const Counter c) {
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
  void multiply(const int s);
  void identity(void);
  void zero(void);
  void copy(const IntMatrix& copy);
  void print(std::ostream& out);
  IntMatrix& operator = (const IntMatrix&);
  IntMatrix(const IntMatrix&);

 private:    
  int*    mat;
  Counter rows;
  Counter cols;
};

#endif // __INTMATRIX_H__
