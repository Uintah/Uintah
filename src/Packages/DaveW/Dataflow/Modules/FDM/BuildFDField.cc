
/*
 *  BuildFDField.cc:  Unfinished modules
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   January 1996
 *
 *  Copyright (C) 1996 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/ColumnMatrixPort.h>
#include <Dataflow/Ports/ScalarFieldPort.h>
#include <Core/Datatypes/ScalarFieldRG.h>
#include <Core/Geometry/Point.h>
#include <Core/Malloc/Allocator.h>
#include <iostream>
using std::cerr;

namespace DaveW {
using namespace SCIRun;

class BuildFDField : public Module {
    ScalarFieldIPort *ifield;
    ColumnMatrixIPort* inrhs;
    ScalarFieldOPort* ofield;
public:
    BuildFDField(const clString& id);
    virtual ~BuildFDField();
    virtual void execute();
};

extern "C" Module* make_BuildFDField(const clString& id)
{
    return scinew BuildFDField(id);
}

BuildFDField::BuildFDField(const clString& id)
: Module("BuildFDField", id, Filter)
{
    ifield=scinew ScalarFieldIPort(this, "ScalarField", ScalarFieldIPort::Atomic);
    add_iport(ifield);
    inrhs=scinew ColumnMatrixIPort(this, "RHS", ColumnMatrixIPort::Atomic);
    add_iport(inrhs);
    // Create the output port
    ofield=scinew ScalarFieldOPort(this, "ScalarField", ScalarFieldIPort::Atomic);
    add_oport(ofield);
}

BuildFDField::~BuildFDField()
{
}

void BuildFDField::execute()
{
    ScalarFieldHandle ifld;
    if(!ifield->get(ifld))
	return;
    ScalarFieldRGBase* sfrg = ifld->getRGBase();
    if (!sfrg) return;
    ColumnMatrixHandle rhshandle;
    if(!inrhs->get(rhshandle))
	return;

    int nx=sfrg->nx;
    int ny=sfrg->ny;
    int nz=sfrg->nz;
    Point min;
    Point max;
    sfrg->get_bounds(min, max);

    ScalarFieldRG* osf=scinew ScalarFieldRG;
    osf->resize(nx, ny, nz);
    osf->set_bounds(min, max);

    ColumnMatrix& rhs=*rhshandle.get_rep();
    if (rhs.nrows() != nx*ny*nz) {
	cerr << "BuildFDField failed -- field size mismatch!\n";
	return;
    }
    int idx=0;
    for (int x=0; x<nx; x++)
	for (int y=0; y<ny; y++)
	   for (int z=0; z<nz; z++, idx++)
	       osf->grid(x,y,z)=rhs[idx];
    ofield->send(ScalarFieldHandle(osf));
}
} // End namespace DaveW


