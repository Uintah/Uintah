

#ifndef DPVec_H
#define DPVec_H 1

class doubleVec;

#include <Classlib/LockingHandle.h>
#include <Datatypes/Datatype.h>

class DPVec : public Datatype {
public:
    DPVec(doubleVec*);
    virtual ~DPVec();
    doubleVec* vec;

    virtual void io(Piostream&);
};

typedef LockingHandle<DPVec> DPVecHandle;

#endif DPVec_H
