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

#include <PSECore/Datatypes/ColumnMatrixPort.h>
#include <PSECore/Datatypes/MeshPort.h>
#include <PSECore/Datatypes/ScalarFieldPort.h>
#include <SCICore/Datatypes/ScalarFieldUG.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Malloc/Allocator.h>

namespace SCIRun {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::Geometry;

class MakeScalarField : public Module {
    MeshIPort* inmesh;
    ColumnMatrixIPort* inrhs;
    ScalarFieldOPort* ofield;
public:
    MakeScalarField(const clString& id);
    virtual ~MakeScalarField();
    virtual void execute();
};

extern "C" Module* make_MakeScalarField(const clString& id)
{
    return scinew MakeScalarField(id);
}

MakeScalarField::MakeScalarField(const clString& id)
: Module("MakeScalarField", id, Filter)
{
    inmesh=scinew MeshIPort(this, "Mesh", MeshIPort::Atomic);
    add_iport(inmesh);
    inrhs=scinew ColumnMatrixIPort(this, "RHS", ColumnMatrixIPort::Atomic);
    add_iport(inrhs);
    // Create the output port
    ofield=scinew ScalarFieldOPort(this, "ScalarField", ScalarFieldIPort::Atomic);
    add_oport(ofield);
}

MakeScalarField::~MakeScalarField()
{
}

void MakeScalarField::execute()
{
    MeshHandle mesh;
    if(!inmesh->get(mesh))
	return;
    ColumnMatrixHandle rhshandle;
    ScalarFieldUG* sf;
    if(!inrhs->get(rhshandle))
	return;
    ColumnMatrix& rhs=*rhshandle.get_rep();
    if (rhs.nrows() == mesh->nodes.size()) {
	cerr << "Using nodal values";
	sf=scinew ScalarFieldUG(ScalarFieldUG::NodalValues);
    }
    else
	if (rhs.nrows() == mesh->elems.size()) {
	    cerr << "Using element values";
	    sf=scinew ScalarFieldUG(ScalarFieldUG::ElementValues);
	}
	else {
	    cerr << "The ColumnMatrix size, " << rhs.nrows() << ", does not match the number of nodes, " << mesh->nodes.size() << ", nor does it match the number of elements, " << mesh->elems.size();
	    return;
	}
    sf->mesh=mesh;
    sf->data.resize(rhs.nrows());
    for(int i=0;i<rhs.nrows();i++){
	if (rhs.nrows() == mesh->nodes.size())
	    if(mesh->nodes[i]->bc)
		sf->data[i]=mesh->nodes[i]->bc->value;
	    else	
		sf->data[i]=rhs[i];
	else
	    sf->data[i]=rhs[i];
    }
    ofield->send(ScalarFieldHandle(sf));
}

} // End namespace Modules
} // End namespace SCIRun


//
// $Log$
// Revision 1.4  2000/09/07 00:12:19  zyp
// MakeScalarField.cc
//
// Revision 1.3  2000/03/17 09:29:12  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.2  2000/03/11 00:41:55  dahart
// Replaced all instances of HashTable<class X, class Y> with the
// Standard Template Library's std::map<class X, class Y, less<class X>>
//
// Revision 1.1  1999/09/05 01:15:27  dmw
// added all of the old SCIRun mesh modules
//
