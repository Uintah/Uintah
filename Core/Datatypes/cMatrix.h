#ifndef cMATRIX_H
#define cMATRIX_H 1

/*
 *  cMatrix.h : ?
 *
 *  Written by:
 *   Author: ?
 *   Department of Computer Science
 *   University of Utah
 *   Date: ?
 *
 *  Copyright (C) 199? SCI Group
 */

#include <Core/Datatypes/Datatype.h>
#include <Core/Containers/LockingHandle.h>
#include <complex>

namespace SCIRun {


class cMatrix;
typedef LockingHandle<cMatrix> cMatrixHandle;

class cVector;
class SCICORESHARE cMatrix:public Datatype{
  
public:
    typedef std::complex<double> Complex;
 virtual cVector operator*( cVector &V)=0;

 virtual void mult(cVector& V,cVector& tmp)=0;
 virtual Complex& get(int row, int col)=0;

// Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
 
  
};

} // End namespace SCIRun


#endif
