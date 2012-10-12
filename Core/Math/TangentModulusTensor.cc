/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

// Implementation of TangentModulusTensor
#include <Core/Math/TangentModulusTensor.h>

using namespace std;
using namespace Uintah;

// Default constructor 
TangentModulusTensor::TangentModulusTensor(): Tensor4D<double>(3,3,3,3)
{
}

// Standard constructor 
TangentModulusTensor::TangentModulusTensor(const FastMatrix& C_6x6):
  Tensor4D<double>(3,3,3,3)
{
  ASSERT(!(C_6x6.numRows() != 6 || C_6x6.numCols() != 6));
  int index[3][3];
  for (int ii = 0; ii < 3; ++ii) index[ii][ii] = ii;
  index[0][1] = 5; index[1][0] = 5;
  index[0][2] = 4; index[2][0] = 4;
  index[1][2] = 3; index[2][1] = 3;

  for (int ii = 0; ii < 3; ++ii) {
    for (int jj = 0; jj < 3; ++jj) {
      int row = index[ii][jj];
      for (int kk = 0; kk < 3; ++kk) {
	for (int ll = 0; ll < 3; ++ll) {
          int col = index[kk][ll];
          (*this)(ii,jj,kk,ll) = C_6x6(row,col);
	}
      }
    }
  }
}

// Copy constructor : Create tensor from given tensor
TangentModulusTensor::TangentModulusTensor(const TangentModulusTensor& tensor):
  Tensor4D<double>(tensor)
                                                 
{
}

// Destructor
TangentModulusTensor::~TangentModulusTensor()
{
}

// Assignment operator
inline TangentModulusTensor& 
TangentModulusTensor::operator=(const TangentModulusTensor& tensor)
{
  Tensor4D<double>::operator=(tensor);
  return *this;   
}

// Convert to 6x6 matrix (Voigt form)
void
TangentModulusTensor::convertToVoigtForm(FastMatrix& C_6x6) const
{
  int index[6][2];
  ASSERT(!(C_6x6.numRows() != 6 || C_6x6.numCols() != 6));

  for (int ii = 0; ii < 3; ++ii) {
    for (int jj = 0; jj < 2; ++jj) {
      index[ii][jj] = ii;
    }
  }
  index[3][0] = 1; index[3][1] = 2;
  index[4][0] = 2; index[4][1] = 0;
  index[5][0] = 0; index[5][1] = 1;

  for (int ii = 0; ii < 6; ++ii) {
    for (int jj = 0; jj < 6; ++jj) {
      C_6x6(ii,jj) = (*this)(index[ii][0], index[ii][1], 
			     index[jj][0], index[jj][1]);
    }
  }
}

// Convert to 3x3x3x3 matrix 
void
TangentModulusTensor::convertToTensorForm(const FastMatrix& C_6x6) 
{
  ASSERT(!(C_6x6.numRows() != 6 || C_6x6.numCols() != 6));
  int index[3][3];
  for (int ii = 0; ii < 3; ++ii) index[ii][ii] = ii;
  index[0][1] = 5; index[1][0] = 5;
  index[0][2] = 4; index[2][0] = 4;
  index[1][2] = 3; index[2][1] = 3;

  for (int ii = 0; ii < 3; ++ii) {
    for (int jj = 0; jj < 3; ++jj) {
      int row = index[ii][jj];
      for (int kk = 0; kk < 3; ++kk) {
	for (int ll = 0; ll < 3; ++ll) {
          int col = index[kk][ll];
          (*this)(ii,jj,kk,ll) = C_6x6(row,col);
	}
      }
    }
  }
}

void 
TangentModulusTensor::contract(const Matrix3& D, Matrix3& sigrate) const
{
  for (int ii = 0; ii < 3; ++ii) {
    for (int jj = 0; jj < 3; ++jj) {
      sigrate(ii,jj) = 0.0;
      for (int kk = 0; kk < 3; ++kk) {
	for (int ll = 0; ll < 3; ++ll) {
          sigrate(ii,jj) += (*this)(ii,jj,kk,ll)*D(kk,ll);
	}
      }
    }
  }
}

void TangentModulusTensor::transformBy2ndOrderTensor(const Matrix3& F, double J){
/*      Computes c_ijkl = (1/J)*F_iI*F_jJ*F_kK*F_lL*C_IJKL    */
  TangentModulusTensor c_ijkl;
  for (int ii = 0; ii < 3; ++ii) {
   for (int jj = 0; jj < 3; ++jj) {
    for (int kk = 0; kk < 3; ++kk) {
     for (int ll = 0; ll < 3; ++ll) {
       c_ijkl(ii,jj,kk,ll) = 0.;
     }
    }
   }
  }

  for (int ii = 0; ii < 3; ++ii) {
   for (int jj = 0; jj < 3; ++jj) {
    for (int kk = 0; kk < 3; ++kk) {
     for (int ll = 0; ll < 3; ++ll) {
      for (int II = 0; II < 3; ++II) {
       for (int JJ = 0; JJ < 3; ++JJ) {
        for (int KK = 0; KK < 3; ++KK) {
         for (int LL = 0; LL < 3; ++LL) {
          c_ijkl(ii,jj,kk,ll) += F(ii,II)*F(jj,JJ)
                                *F(kk,KK)*F(ll,LL)*(*this)(II,JJ,KK,LL);
        }
       }
      }
     }
    }
   }
  }
 }

 (*this) = c_ijkl;
}

ostream& operator<<(ostream& out, const TangentModulusTensor& C)
{
  int index[6][2];

  for (int ii = 0; ii < 3; ++ii) {
    for (int jj = 0; jj < 2; ++jj) {
      index[ii][jj] = ii;
    }
  }
  index[3][0] = 1; index[3][1] = 2;
  index[4][0] = 2; index[4][1] = 0;
  index[5][0] = 0; index[5][1] = 1;

  for (int ii = 0; ii < 6; ++ii) {
    for (int jj = 0; jj < 6; ++jj) {
      out << C(index[ii][0], index[ii][1], index[jj][0], index[jj][1])
          << ' ';
    }
  }
  out << endl;

  return out;
}
