

#ifndef DPProblem_H
#define DPProblem_H 1

class DPNavierStokes;

#include <Classlib/LockingHandle.h>
#include <Datatypes/Datatype.h>

class DPProblem : public Datatype {
public:
    DPProblem(DPNavierStokes*);
    virtual ~DPProblem();
    DPNavierStokes* problem;

    virtual void io(Piostream&);
};

typedef LockingHandle<DPProblem> DPProblemHandle;

#endif DPProblem_H
