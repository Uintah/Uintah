/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
 *  FastMatrix.h: 
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   October 2001
 *
 */

#ifndef Uintah_Core_Math_FastMatrix_h
#define Uintah_Core_Math_FastMatrix_h

#include <sgi_stl_warnings_off.h>
#include <iosfwd>
#include <vector>
#include <sgi_stl_warnings_on.h>

namespace Uintah {
  using namespace std;
  class FastMatrix {
  public:
    FastMatrix(int rows, int cols);
    ~FastMatrix();

    int numRows() const {
      return rows;
    }
    int numCols() const {
      return cols;
    }
    double& operator()(int r, int c) {
      return mat[r][c];
    }
    double operator()(int r, int c) const {
      return mat[r][c];
    }

    void destructiveInvert(FastMatrix& inverse);
    void destructiveSolve(const vector<double>& b, vector<double>& X);
    void transpose(const FastMatrix& transpose);
    void multiply(const vector<double>& b, vector<double>& X) const;
    void multiply(const FastMatrix& a, const FastMatrix& b);
    void multiply(double s);
    double conditionNumber() const;
    void identity();
    void zero();
    void copy(const FastMatrix& copy);
    void print(std::ostream& out);

  private:
    static const int smallSize;
    static const int smallRows;
    double smallMat[16];    // Make sure this equals smallSize
    double* smallMatPtr[4]; // Make sure this equals smallRows
    double** mat;
    int rows, cols;

    void big_destructiveInvert(FastMatrix& inverse);
    void big_destructiveSolve(const vector<double>& b, vector<double>& X);

    FastMatrix(const FastMatrix&);
    FastMatrix& operator=(const FastMatrix&);
  };

}

#endif
