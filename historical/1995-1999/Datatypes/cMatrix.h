#ifndef cMATRIX_H
#define cMATRIX_H 1


#include <Datatypes/Datatype.h>
#include <Classlib/LockingHandle.h>

class cMatrix;
typedef LockingHandle<cMatrix> cMatrixHandle;
class Complex;


class cVector;
class cMatrix:public Datatype{
  
  
public:
 virtual cVector operator*( cVector &V)=0;

 virtual void mult(cVector& V,cVector& tmp)=0;
 virtual Complex& get(int row, int col)=0;
    virtual cMatrix* clone()=0;

// Persistent representation...
    virtual void io(Piostream&);
    static PersistentTypeID type_id;
 
  
};

#endif
