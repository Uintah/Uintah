

#ifndef DPGRID_H
#define DPGRID_H 1

class GridFE;

#include <Classlib/LockingHandle.h>
#include <Datatypes/Datatype.h>

class DPGrid : public Datatype {
public:
    DPGrid();
    virtual ~DPGrid();
    GridFE* grid;

    virtual void io(Piostream&);
};

typedef LockingHandle<DPGrid> DPGridHandle;

#endif DPGRID_H
