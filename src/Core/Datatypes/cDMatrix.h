
/*
 *  cDMatrix.h : ?
 *
 *  Written by:
 *   Author: ?
 *   Department of Computer Science
 *   University of Utah
 *   Date: ?
 *
 *  Copyright (C) 199? SCI Group
 */

#ifndef CDMATRIX_H
#define CDMATRIX_H 1

#include <Core/share/share.h>
#include <Core/Datatypes/cMatrix.h>
#include <Core/Datatypes/cVector.h>

namespace SCIRun {

class SCICORESHARE cDMatrix:public cMatrix{
  
public:
    typedef std::complex<double> Complex;
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

  friend SCICORESHARE cDMatrix operator*(const cDMatrix& B,Complex x);
  friend SCICORESHARE cDMatrix operator*(Complex x, const cDMatrix& B);
  
  friend SCICORESHARE cDMatrix operator*(const cDMatrix& B,double x);
  friend SCICORESHARE cDMatrix operator*(double x, const cDMatrix& B);

  
  friend SCICORESHARE std::ostream &operator<< (std::ostream &output, cDMatrix &B);
  
  cVector  operator*(cVector &V);

  void mult(cVector& V,cVector& tmp);
  virtual Complex& get(int row, int col);
  
 };

} // End namespace SCIRun


#endif
