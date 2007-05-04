#ifndef __INTMATRIX_H__
#define __INTMATRIX_H__

#include "Macros.h"

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
  void identity(void);
  void zero(void);
  void copy(const IntMatrix& copy);
  IntMatrix& operator = (const IntMatrix&);
  IntMatrix(const IntMatrix&);

  friend std::ostream& operator << (std::ostream& os, const IntMatrix& a);

 private:    
  int*    mat;
  Counter rows;
  Counter cols;
};

#endif // __INTMATRIX_H__
