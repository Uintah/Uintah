/*
 *  MakeScalarField.cc:  Unfinished modules
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <MakeScalarField/MakeScalarField.h>
#include <MatrixPort.h>
#include <MeshPort.h>
#include <ModuleList.h>
#include <NotFinished.h>
#include <ScalarFieldPort.h>
#include <Geometry/Point.h>
#include <iostream.h>
#include <fstream.h>

static Module* make_MakeScalarField(const clString& id)
{
    return new MakeScalarField(id);
}

static RegisterModule db1("Unfinished", "MakeScalarField", make_MakeScalarField);

MakeScalarField::MakeScalarField(const clString& id)
: Module("MakeScalarField", id, Filter)
{
    add_iport(new MeshIPort(this, "Mesh", MeshIPort::Atomic));
    add_iport(new MatrixIPort(this, "Geometry", MatrixIPort::Atomic));
    // Create the output port
    add_oport(new ScalarFieldOPort(this, "Geometry", ScalarFieldIPort::Atomic));
}

MakeScalarField::MakeScalarField(const MakeScalarField& copy, int deep)
: Module(copy, deep)
{
    NOT_FINISHED("MakeScalarField::MakeScalarField");
}

MakeScalarField::~MakeScalarField()
{
}

Module* MakeScalarField::clone(int deep)
{
    return new MakeScalarField(*this, deep);
}

void MakeScalarField::execute()
{
    NOT_FINISHED("MakeScalarField::execute");
}
