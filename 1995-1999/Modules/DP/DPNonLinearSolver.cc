
inline double abs(double x)
{
    return x<0?-x:x;
}

#define NUMT double
#define ARRAY_RANGECHECK
#define HANDLE_0PTR_CHECK
#define F77HANDLING 1


#include <NonLinEqSolverUDC.h>  // user's definition of nonlinear PDEs
#include <NonLinEqSolver_prm.h> // parameters for nonlinear solvers
#include <NonLinEqSolver.h>     // nonlinear solver interface
#include <createNonLinEqSolver.h>

#undef Handle
#undef Vector

#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Modules/DP/DPVecPort.h>

class DPNonLinearSolverM : public Module, public NonLinEqSolverUDC {
public:
    DPVecIPort* initial_solution;
    DPVecOPort* linear_out;
    DPVecIPort* linear_in;
    DPVecOPort* final_solution;
    doubleVec* lin_solution;
    doubleVec* nlvec;
    NonLinEqSolver* nlsolver;
    /* SCIRUN STUFF */
    DPNonLinearSolverM(const clString& id);
    virtual void execute();
    virtual Module* clone(int deep);
    virtual void makeAndSolveLinearSystem();
};

extern "C" {
Module* make_DPNonLinearSolver(const clString& id)
{
    return new DPNonLinearSolverM(id);
}
};

Module* DPNonLinearSolverM::clone(int deep)
{
    NOT_FINISHED("DPNonLinearSolver::clone");
    return 0;
}

DPNonLinearSolverM::DPNonLinearSolverM(const clString& id)
: Module("DPNonLinearSolver", id, Filter)
{
    initial_solution=new DPVecIPort(this, "init", DPVecIPort::Atomic);
    add_iport(initial_solution);
    linear_in=new DPVecIPort(this, "in_linear", DPVecIPort::Atomic);
    add_iport(linear_in);
    linear_out=new DPVecOPort(this, "out_linear", DPVecIPort::Atomic);
    add_oport(linear_out);
    final_solution=new DPVecOPort(this, "final", DPVecIPort::Atomic);
    add_oport(final_solution);
}

void DPNonLinearSolverM::execute()
{
    DPVecHandle nl_solution;
    if(!initial_solution->get(nl_solution))
	return;

    prm(NonLinEqSolver)    nlsolver_prm;    // init prm for nlsolver
    nlsolver=createNonLinEqSolver (nlsolver_prm);
    nlsolver->attachUserCode (*this);
    nlsolver->attachNonLinSol (*nl_solution->vec);
    nlvec=nl_solution->vec;
    lin_solution=new doubleVec;
    lin_solution->redim(nl_solution->vec->size());
    nlsolver->attachLinSol (*lin_solution);
nl_solution->vec->print(s_o,"nonlin_solution before nonlinear iteration");

    cerr << "nlsolver->solve...\n";
    if (!nlsolver->solve ())
	errorFP("DPNavierStokes::driver",
		"The nlsolver.solve call: divergence of solver \"%s\"",
		nlsolver_prm.method.chars());
    // in each iteration in the loop in nlsolver->solve,
    // makeAndSolveLinearSystem is called
    final_solution->send(new DPVec(nl_solution->vec));
}

void DPNonLinearSolverM::makeAndSolveLinearSystem()
{
    nlvec->print(s_o,"nonlin_solution before nonlinear iteration");
    linear_out->send_intermediate(new DPVec(nlvec));
    DPVecHandle lin;
    if(!linear_in->get(lin)){
	cerr << "Linear system loop failed!\n";
    }
    lin->vec->print(s_o,"linear solution back on nlsolver...");
    cerr << "Attaching linear solution...\n";
    nlsolver->attachLinSol(*lin->vec);
    cerr << "Done\n";
}
