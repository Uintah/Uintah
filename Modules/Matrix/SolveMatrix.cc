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
#include <Dataflow/ModuleList.h>
#include <Datatypes/ColumnMatrixPort.h>
#include <Datatypes/MatrixPort.h>
#include <Datatypes/SurfacePort.h>
#include <Geometry/Point.h>

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

static Module* make_SolveMatrix(const clString& id)
{
    return new SolveMatrix(id);
}

static RegisterModule db1("Unfinished", "SolveMatrix", make_SolveMatrix);

SolveMatrix::SolveMatrix(const clString& id)
: Module("SolveMatrix", id, Filter)
{
    matrixport=new MatrixIPort(this, "Matrix", MatrixIPort::Atomic);
    add_iport(matrixport);
    rhsport=new ColumnMatrixIPort(this, "RHS", ColumnMatrixIPort::Atomic);
    add_iport(rhsport);

    solport=new ColumnMatrixOPort(this, "Solution", ColumnMatrixIPort::Atomic);
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
    return new SolveMatrix(*this, deep);
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
	solution=new ColumnMatrix(rhs->nrows());
	solution->zero();
    } else {
	solution.detach();
    }
    matrix->isolve(*solution.get_rep(), *rhs.get_rep(), 1.e-4);
    solport->send(solution);
}
