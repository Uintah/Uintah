#ifndef cDMATRIX_H
#define ccDMatrix_H 1

#include <iostream.h>
#include <fstream.h>
#include <Datatypes/cMatrix.h>
#include <Math/Complex.h>
#include <Datatypes/cVector.h>

class cDMatrix:public cMatrix{
  
private:
  Complex **a;
  int Size;
    
public:

  cDMatrix(int N); //constructor
  cDMatrix(const cDMatrix &B); //copy constructor;
  cDMatrix &operator=(const cDMatrix &B); //assigment operator  
  ~cDMatrix(); //Destructor;
  
  int size() {return Size;};
  Complex &operator()(int i, int j);
  int load(char* filename);
  
  cDMatrix operator+(const cDMatrix& B) const;
  cDMatrix operator-(const cDMatrix& B) const;
  cDMatrix operator*(const cDMatrix& B) const;

  friend cDMatrix operator*(const cDMatrix& B,Complex x);
  friend cDMatrix operator*(Complex x, const cDMatrix& B);
  
  friend cDMatrix operator*(const cDMatrix& B,double x);
  friend cDMatrix operator*(double x, const cDMatrix& B);

  
  friend ostream &operator<< (ostream &output, cDMatrix &B);
  
  cVector  operator*(cVector &V);

  void mult(cVector& V,cVector& tmp);
  virtual Complex& get(int row, int col);
  
 };

#endif
