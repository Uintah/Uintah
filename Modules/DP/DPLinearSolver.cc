
inline double abs(double x)
{
    return x<0?-x:x;
}

#define NUMT double
#define ARRAY_RANGECHECK
#define HANDLE_0PTR_CHECK
#define F77HANDLING 1

#include <LinEqAdm.h>

#undef Handle
#undef Vector

#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Modules/DP/DPLinEqPort.h>
#include <Modules/DP/DPVecPort.h>
#include <Modules/DP/DPLinEq.h>

class DPLinearSolverM : public Module {
public:
    DPLinEqIPort* lineq;
    DPVecOPort* sol_out;
    /* SCIRUN STUFF */
    DPLinearSolverM(const clString& id);
    virtual void execute();
    virtual Module* clone(int deep);
};

extern "C" {
Module* make_DPLinearSolver(const clString& id)
{
    return new DPLinearSolverM(id);
}
};

Module* DPLinearSolverM::clone(int deep)
{
    NOT_FINISHED("DPLinearSolver::clone");
    return 0;
}

DPLinearSolverM::DPLinearSolverM(const clString& id)
: Module("DPLinearSolver", id, Filter)
{
    lineq=new DPLinEqIPort(this, "system", DPLinEqIPort::Atomic);
    add_iport(lineq);
    sol_out=new DPVecOPort(this, "out_sol", DPVecIPort::Atomic);
    add_oport(sol_out);
}

void DPLinearSolverM::execute()
{
    DPLinEqHandle lin;
    if(!lineq->get(lin)){
	cerr << "Get failed!\n";
	return;
    }
    lin->lineq->solve(); 
    lin->lineq->scan(global_menu);
    cerr << "Solve done!\n";
    lin->sol->print(s_o,"linear_solution after solve");
    sol_out->send(new DPVec(lin->sol));
}
