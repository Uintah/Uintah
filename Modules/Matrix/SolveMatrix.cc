/*
 *  SolveMatrix.cc:  Unfinished modules
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Datatypes/ColumnMatrixPort.h>
#include <Datatypes/MatrixPort.h>
#include <Datatypes/SurfacePort.h>
#include <Geometry/Point.h>
#include <Malloc/Allocator.h>
#include <strstream.h>

class SolveMatrix : public Module {
    MatrixIPort* matrixport;
    ColumnMatrixIPort* rhsport;
    ColumnMatrixOPort* solport;
    ColumnMatrixHandle solution;
public:
    SolveMatrix(const clString& id);
    SolveMatrix(const SolveMatrix&, int deep);
    virtual ~SolveMatrix();
    virtual Module* clone(int deep);
    virtual void execute();
};

class SolveMatrixUpdater : public MatrixUpdater {
public:
    clString solverid;
    ColumnMatrixOPort* solport;
    Module* module;
    virtual void update(int, double, double, double,
			const ColumnMatrix&);
};

extern "C" {
Module* make_SolveMatrix(const clString& id)
{
    return scinew SolveMatrix(id);
}
};

SolveMatrix::SolveMatrix(const clString& id)
: Module("SolveMatrix", id, Filter)
{
    matrixport=scinew MatrixIPort(this, "Matrix", MatrixIPort::Atomic);
    add_iport(matrixport);
    rhsport=scinew ColumnMatrixIPort(this, "RHS", ColumnMatrixIPort::Atomic);
    add_iport(rhsport);

    solport=scinew ColumnMatrixOPort(this, "Solution", ColumnMatrixIPort::Atomic);
    add_oport(solport);
}

SolveMatrix::SolveMatrix(const SolveMatrix& copy, int deep)
: Module(copy, deep)
{
    NOT_FINISHED("SolveMatrix::SolveMatrix");
}

SolveMatrix::~SolveMatrix()
{
}

Module* SolveMatrix::clone(int deep)
{
    return scinew SolveMatrix(*this, deep);
}

void SolveMatrix::execute()
{
    MatrixHandle matrix;
    if(!matrixport->get(matrix))
	return;	
    ColumnMatrixHandle rhs;
    if(!rhsport->get(rhs))
	return;
    if(!solution.get_rep()){
	solution=scinew ColumnMatrix(rhs->nrows());
	solution->zero();
    } else {
	solution.detach();
    }
    SolveMatrixUpdater updater;
    updater.solverid=id;
    updater.solport=solport;
    updater.module=this;
    matrix->isolve(*solution.get_rep(), *rhs.get_rep(), 1.e-4,
		   &updater);
    solport->send(solution);
}

void SolveMatrixUpdater::update(int iter, double first_error,
				double current_error, double final_error,
				const ColumnMatrix& solution)
{
    char buf[1000];
    ostrstream str(buf, 1000);
    str << solverid << " update_iter " << iter << " " << first_error << " " << current_error << " " << final_error << '\0';
    TCL::execute(str.str());
#if 0
    solport->send(ColumnMatrixHandle(solution.clone()));
    module->multisend(solport);
#endif
}
