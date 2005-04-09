#ifndef cSMATRIX_H
#define ccSMatrix_H 1

#include <iostream.h>
#include <fstream.h>
#include <Datatypes/cMatrix.h>
#include <Math/Complex.h>
#include <Datatypes/cVector.h>

class cSMatrix:public cMatrix{

  
private:
  Complex *a;
  int *row_ptr;
  int *col;
  int nrows;
  int ncols;
  int nnz;

  
public:   
 
  cSMatrix(int nrows, int ncols,int nnz,Complex *a, int * row, int *col );
  ~cSMatrix();
    
  friend ostream &operator<< (ostream &output, cSMatrix &B);
  
 
 cVector  operator*(cVector &V);

 void mult(cVector& V,cVector& tmp);
 virtual Complex& get(int row, int col);

  
};


#endif





