#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <string>
#include <iostream>
#include <sstream>
#include <vector>

using namespace::std;

#if defined(ABQ_LINUX)
#include <aba_for_c.h>
#endif // ABQ_LINUX


namespace {

  std::ofstream _logFile;
  bool _isInitialized = false;
  
  class matrix {
    
  public:
    
    // Constructors
    matrix() 
    { double values[3][3] = {{1.0,0.0,0.0},{0.0,1.0,0.0},{0.0,0.0,1.0}};
      memcpy(&_data, &values, sizeof(_data)); return; }
    
    matrix(const double (&values)[3][3])
    { memcpy(&_data, &values, sizeof(_data)); return; }
    
    matrix(const matrix & inputMatrix)
    { 
      for (int i=0; i<3; ++i) 
	for (int j=0; j<3; ++j) 
	  _data[i][j] = inputMatrix(i,j);
      return; 
    }
  
    // Mathematical operatations
    
    double trace() { return (_data[0][0] + _data[1][1] + _data[2][2]); }

    matrix
    transpose() {
      matrix returnValue(*this);
      returnValue(0,1) = (*this)(1,0); returnValue(1,0) = (*this)(0,1);
      returnValue(0,2) = (*this)(2,0); returnValue(2,0) = (*this)(0,2);
      returnValue(1,2) = (*this)(2,1); returnValue(2,1) = (*this)(1,2);
      return returnValue;
    }
    
    double
    determinant() {
      double J =(_data[0][0]*(_data[1][1]*_data[2][2]-_data[1][2]*_data[2][1])-
		 _data[0][1]*(_data[1][0]*_data[2][2]-_data[1][2]*_data[2][0])+
		 _data[0][2]*(_data[1][0]*_data[2][1]-_data[1][1]*_data[2][0]));
      return J;
      
    }
    
    // Index operators
    double operator()(int i, int j) const { return _data[i][j]; }
    double & operator()(int i, int j) { return _data[i][j]; }
    
    //Print matrix
    void print(const std::string & label) {
      std::cout << label << ":" << std::endl;
      std::cout << "  " << _data[0][0] << ", " << _data[0][1] << ", " 
		<< _data[0][2] << std::endl;
      std::cout << "  " << _data[1][0] << ", " << _data[1][1] << ", " 
		<< _data[1][2] << std::endl;
      std::cout << "  " << _data[2][0] << ", " << _data[2][1] << ", " 
		<< _data[2][2] << std::endl;
      return;
    }

    void printSymm(std::ofstream & oFile) {
      oFile << _data[0][0] << " " << _data[1][1] << " " << _data[2][2] 
	    << " " << _data[0][1] << " " << _data[1][2] << " " 
	    << _data[2][0] << std::endl;
      return;
    }    

    // Data member
  private:
    double _data[3][3];
    
  };
  
  // Global operators
  matrix 
  operator*(double a, const matrix & M) {
    matrix returnValue(M);
    for (int i=0; i<3; ++i)
      for (int j=0; j<3; ++j)
        returnValue(i,j) *= a;
    return returnValue; }

  matrix operator*(const matrix & M, double a) {
    matrix returnValue(M);
    for (int i=0; i<3; ++i)
      for (int j=0; j<3; ++j)
	returnValue(i,j) *= a;
    return returnValue; }

  matrix operator*(const matrix & A, const matrix & B) {
    matrix returnValue;
    for (int i=0; i<3; ++i) {
      for (int j=0; j<3; ++j) {
	double sum = 0.0;
	for (int k=0; k<3; ++k) {
	  sum += A(i,k)*B(k,j);
	}
	returnValue(i,j) = sum;
      }
    }
    return returnValue; }

  matrix operator+(const matrix & A, const matrix & B) {
    matrix returnValue;
    for (int i=0; i<3; ++i)
      for (int j=0; j<3; ++j)
	returnValue(i,j) = A(i,j) + B(i,j);
    return returnValue; }

  matrix operator-(const matrix & A, const matrix & B) {
    matrix returnValue;
    for (int i=0; i<3; ++i)
      for (int j=0; j<3; ++j)
	returnValue(i,j) = A(i,j) - B(i,j);
    return returnValue; }
}

extern "C" void vumat (
// Read only
  const int *blockInfo, const int &ndir, const int &nshr, const int &nstatev, 
  const int &nfieldv, const int &nprops, const int &lanneal, 
  const double &stepTime, const double &totalTime, const double &dt, 
  const char *cmname, const double *coordMp, const double *charLength,
  const double *props, const double *density, const double *strainInc, 
  const double *relSpinInc, const double *tempOld, const double *stretchOld, 
  const double *defgradOld, const double *fieldOld, const double *stressOld, 
  const double *stateOld, const double *enerInternOld, 
  const double *enerInelasOld, const double *tempNew, const double *stretchNew,
  const double *defgradNew, const double *fieldNew,
// Write only
  double *stressNew, double *stateNew, double *enerInternNew, 
  double *enerInelasNew )
{

  //
 //
  // Perform initialization on the first step
  //
  if (!_isInitialized) {

    //
    // Create logging file
    //
    _logFile.open("vumat_c++.log",std::ofstream::app);
    _logFile.precision(18);

    _isInitialized = true;
    
  } /* End initialization */

  // Get elastic properties
  const double youngMod = props[0];
  const double nu = props[1];
  const double twoMu  = youngMod / (1.0 + nu);
  const double lambda = twoMu*nu/(1.0-2.0*nu);

  // loop through the blocks
  matrix U;
  matrix I;
  int nblock = blockInfo[0];

  for (int i = 0; i  < nblock; ++i) {

    // Put stretchNew into matrix
    U(0,0) = stretchNew[i];
    U(1,1) = stretchNew[nblock+i];
    U(2,2) = stretchNew[2*nblock+i];
    U(0,1) = stretchNew[3*nblock+i];
    U(1,2) = stretchNew[4*nblock+i];
    U(2,0) = stretchNew[5*nblock+i];
    U(1,0) = U(0,1);
    U(2,1) = U(1,2);
    U(0,2) = U(2,0);

    // Compute stress
    matrix E = 0.5*(U.transpose()*U - I);
    matrix S = lambda*E.trace()*I + twoMu*E;
    matrix sigma = U*S*U.transpose()*(1.0/U.determinant());

    // Store stress in stressNew
    stressNew[i] = sigma(0,0);
    stressNew[nblock+i] = sigma(1,1);
    stressNew[2*nblock+i] = sigma(2,2);
    stressNew[3*nblock+i] = sigma(0,1);
    stressNew[4*nblock+i] = sigma(1,2);
    stressNew[5*nblock+i] = sigma(2,0);
  } /* loop over blocks */

  return;
}
