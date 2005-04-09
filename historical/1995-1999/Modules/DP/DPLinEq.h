

#ifndef DPLinEq_H
#define DPLinEq_H 1

class LinEqAdm;
class doubleVec;

#include <Classlib/LockingHandle.h>
#include <Datatypes/Datatype.h>

class DPLinEq : public Datatype {
public:
    DPLinEq(LinEqAdm*);
    virtual ~DPLinEq();
    LinEqAdm* lineq;
    doubleVec* sol;

    virtual void io(Piostream&);
};

typedef LockingHandle<DPLinEq> DPLinEqHandle;

#endif DPLinEq_H
