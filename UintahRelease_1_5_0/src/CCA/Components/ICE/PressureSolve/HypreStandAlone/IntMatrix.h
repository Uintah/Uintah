/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

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
    mat = scinew int[rows*cols];
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
