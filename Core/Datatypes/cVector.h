/*
 *  cVector.h : ?
 *
 *  Written by:
 *   Author: ?
 *   Department of Computer Science
 *   University of Utah
 *   Date: ?
 *
 *  Copyright (C) 199? SCI Group
 */

#ifndef cVECTOR_H
#define cVECTOR_H 1

#include <Core/share/share.h>
#include <iosfwd>
#include <math.h>
#include <complex>

#include <Core/Datatypes/Datatype.h>
#include <Core/Containers/LockingHandle.h>

namespace SCIRun {


class cVector;
typedef LockingHandle<cVector> cVectorHandle;

class SCICORESHARE cVector :public Datatype{

  friend class cDMatrix;
  friend class cSMatrix;

public:  
    typedef std::complex<double> Complex;
private:
    Complex *a;
    int Size;
  
public:

// Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;

  
  cVector (int N); //constructor
  cVector (const cVector &B); //copy constructor;
  cVector &operator=(const cVector &B); //assigment operator  
  ~cVector(); //Destructor;
  
  int size() {return Size;};
  double norm();
  Complex &operator()(int i);
  void conj();
  int load(char* filename);
  
  
  void set(const cVector& B);
  void add(const cVector& B);
  void subtr(const cVector& B);
  void mult(const Complex x); 
  
  cVector operator+(const cVector& B) const;
  cVector operator-(const cVector& B) const;
  
  friend SCICORESHARE Complex operator*(cVector& A, cVector& B);
  friend SCICORESHARE cVector  operator*(const cVector& B, Complex x);
  friend SCICORESHARE cVector  operator*(Complex x, const cVector &B);
  friend SCICORESHARE cVector  operator*(const cVector& B,double x);
  friend SCICORESHARE cVector  operator*(double x, const cVector &B);
  
  friend SCICORESHARE std::ostream &operator<< (std::ostream &output, cVector &B);
  
};

} // End namespace SCIRun


#endif



