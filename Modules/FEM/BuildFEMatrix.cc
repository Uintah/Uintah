/*
 *  BuildFEMatrix.cc:  Unfinished modules
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <Modules/FEM/BuildFEMatrix.h>

#include <Classlib/NotFinished.h>
#include <Dataflow/ModuleList.h>
#include <Datatypes/MatrixPort.h>
#include <Datatypes/MeshPort.h>
#include <Datatypes/SurfacePort.h>
#include <Geometry/Point.h>

#include <iostream.h>
#include <fstream.h>

static Module* make_BuildFEMatrix(const clString& id)
{
    return new BuildFEMatrix(id);
}

static RegisterModule db1("Unfinished", "BuildFEMatrix", make_BuildFEMatrix);

BuildFEMatrix::BuildFEMatrix(const clString& id)
: Module("BuildFEMatrix", id, Filter)
{
    add_iport(new MeshIPort(this, "Geometry", MeshIPort::Atomic));
    // Create the output port
    add_oport(new MatrixOPort(this, "Geometry", MatrixIPort::Atomic));
}

BuildFEMatrix::BuildFEMatrix(const BuildFEMatrix& copy, int deep)
: Module(copy, deep)
{
    NOT_FINISHED("BuildFEMatrix::BuildFEMatrix");
}

BuildFEMatrix::~BuildFEMatrix()
{
}

Module* BuildFEMatrix::clone(int deep)
{
    return new BuildFEMatrix(*this, deep);
}

void BuildFEMatrix::execute()
{
    NOT_FINISHED("BuildFEMatrix::execute");
}
