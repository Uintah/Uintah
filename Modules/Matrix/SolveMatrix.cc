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

#include <SolveMatrix/SolveMatrix.h>
#include <MatrixPort.h>
#include <ModuleList.h>
#include <NotFinished.h>
#include <SurfacePort.h>
#include <Geometry/Point.h>
#include <iostream.h>
#include <fstream.h>

static Module* make_SolveMatrix(const clString& id)
{
    return new SolveMatrix(id);
}

static RegisterModule db1("Unfinished", "SolveMatrix", make_SolveMatrix);

SolveMatrix::SolveMatrix(const clString& id)
: Module("SolveMatrix", id, Filter)
{
    add_iport(new MatrixIPort(this, "Geometry", MatrixIPort::Atomic));
    // Create the output port
    add_oport(new MatrixOPort(this, "Geometry", MatrixIPort::Atomic));
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
    NOT_FINISHED("SolveMatrix::execute");
}
