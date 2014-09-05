//static char *id="@(#) $Id$";

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

#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/ColumnMatrixPort.h>
#include <PSECore/Datatypes/ScalarFieldPort.h>
#include <SCICore/Datatypes/ScalarFieldRG.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Malloc/Allocator.h>
#include <iostream>
using std::cerr;

namespace DaveW {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;

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

} // End namespace Modules
} // End namespace DaveW

//
// $Log$
// Revision 1.5  2000/03/17 09:25:40  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.4  1999/10/07 02:06:33  sparker
// use standard iostreams and complex type
//
// Revision 1.3  1999/09/08 02:26:26  sparker
// Various #include cleanups
//
// Revision 1.2  1999/08/25 03:47:40  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.1  1999/08/24 06:23:04  dmw
// Added in everything for the DaveW branch
//
// Revision 1.2  1999/05/03 04:52:15  dmw
// Added and updated DaveW Datatypes/Modules
//
// Revision 1.1  1999/04/27 23:44:08  dav
// moved FDM to DaveW
//
// Revision 1.2  1999/04/27 22:57:47  dav
// updates in Modules for Datatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:30  dav
// Import sources
//
//
