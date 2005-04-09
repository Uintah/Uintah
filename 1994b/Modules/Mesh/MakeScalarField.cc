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

#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Dataflow/ModuleList.h>
#include <Datatypes/ColumnMatrixPort.h>
#include <Datatypes/MeshPort.h>
#include <Datatypes/ScalarFieldPort.h>
#include <Datatypes/ScalarFieldUG.h>
#include <Geometry/Point.h>

class MakeScalarField : public Module {
    MeshIPort* inmesh;
    ColumnMatrixIPort* inrhs;
    ScalarFieldOPort* ofield;
public:
    MakeScalarField(const clString& id);
    MakeScalarField(const MakeScalarField&, int deep);
    virtual ~MakeScalarField();
    virtual Module* clone(int deep);
    virtual void execute();
};

static Module* make_MakeScalarField(const clString& id)
{
    return new MakeScalarField(id);
}

static RegisterModule db1("Unfinished", "MakeScalarField", make_MakeScalarField);

MakeScalarField::MakeScalarField(const clString& id)
: Module("MakeScalarField", id, Filter)
{
    inmesh=new MeshIPort(this, "Mesh", MeshIPort::Atomic);
    add_iport(inmesh);
    inrhs=new ColumnMatrixIPort(this, "RHS", ColumnMatrixIPort::Atomic);
    add_iport(inrhs);
    // Create the output port
    ofield=new ScalarFieldOPort(this, "ScalarField", ScalarFieldIPort::Atomic);
    add_oport(ofield);
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
    MeshHandle mesh;
    if(!inmesh->get(mesh))
	return;
    ColumnMatrixHandle rhshandle;
    if(!inrhs->get(rhshandle))
	return;
    ScalarFieldUG* sf=new ScalarFieldUG;
    sf->mesh=mesh;
    ColumnMatrix& rhs=*rhshandle.get_rep();
    sf->data.resize(rhs.nrows());
    for(int i=0;i<rhs.nrows();i++){
	if(mesh->nodes[i]->ndof>0)
	    sf->data[i]=rhs[i];
	else
	    sf->data[i]=mesh->nodes[i]->value;
    }
    ofield->send(sf);
}
