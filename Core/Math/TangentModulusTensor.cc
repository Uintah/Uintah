// Implementation of TangentModulusTensor
#include <Packages/Uintah/Core/Math/TangentModulusTensor.h>

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
TangentModulusTensor::convertToVoigtForm(FastMatrix& C_6x6) 
{
  int index[6][2];
  ASSERT(!(C_6x6.numRows() != 6 || C_6x6.numCols() != 6));

  for (int ii = 0; ii < 3; ++ii) {
    for (int jj = 0; jj < 3; ++jj) {
      index[ii][jj] = ii;
    }
  }
  index[3][0] = 1; index[3][1] = 2;
  index[4][0] = 0; index[4][1] = 2;
  index[5][0] = 1; index[5][1] = 2;

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
