
inline double abs(double x)
{
    return x<0?-x:x;
}

#define NUMT double
#define ARRAY_RANGECHECK
#define HANDLE_0PTR_CHECK
#define F77HANDLING 1


#include <FEM.h>                // finite element toolbox
#include <readOrMakeGrid.h>

#undef Handle
#undef Vector

#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Modules/DP/DPGridPort.h>

class DPGenGridM : public Module {
public:
    DPGridOPort* outgrid;

    /* SCIRUN STUFF */
    DPGenGridM(const clString& id);
    virtual void execute();
    virtual Module* clone(int deep);
};

extern "C" {
Module* make_DPGenGrid(const clString& id)
{
    return new DPGenGridM(id);
}
};

Module* DPGenGridM::clone(int deep)
{
    NOT_FINISHED("DPGenGrid::clone");
    return 0;
}

DPGenGridM::DPGenGridM(const clString& id)
: Module("DPGenGrid", id, Filter)
{
    outgrid=new DPGridOPort(this, "DPGrid", DPGridIPort::Atomic);
    add_oport(outgrid);
}

void DPGenGridM::execute()
{
    DPGrid* grid=new DPGrid;
    grid->grid=new GridFE();
    String gridfile = "PREPROCESSOR=PreproBox/d=3 [0,0.05]x[0,0.01]x[0,0.008]/d=3 elm_tp=ElmTensorProd1 div=[6,6,6], grading=[1,2,2]";
    readOrMakeGrid(*grid->grid, gridfile);
    outgrid->send(grid);
}
