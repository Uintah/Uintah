/*
 *  ExtractMesh.cc:  Unfinished modules
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   March 1994
 *
 *  Copyright (C) 1994 SCI Group
 */

#include <SCICore/Datatypes/ColumnMatrix.h>
#include <SCICore/Datatypes/DenseMatrix.h>
#include <SCICore/Datatypes/ScalarFieldRGBase.h>
#include <SCICore/Datatypes/ScalarFieldUG.h>
#include <SCICore/Datatypes/VectorFieldRG.h>
#include <SCICore/Datatypes/VectorFieldUG.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Malloc/Allocator.h>
#include <PSECore/Datatypes/ColumnMatrixPort.h>
#include <PSECore/Datatypes/MatrixPort.h>
#include <PSECore/Datatypes/MeshPort.h>
#include <PSECore/Datatypes/ScalarFieldPort.h>
#include <PSECore/Datatypes/VectorFieldPort.h>

namespace SCIRun {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::Geometry;

class ExtractMesh : public Module {
    ScalarFieldIPort* inports;
    VectorFieldIPort* inportv;
    MeshOPort* outport;
    ColumnMatrixOPort* ocol;
    MatrixOPort* omat;
public:
    ExtractMesh(const clString& id);
    virtual ~ExtractMesh();
    virtual void execute();
};

Module* make_ExtractMesh(const clString& id)
{
    return scinew ExtractMesh(id);
}

ExtractMesh::ExtractMesh(const clString& id)
: Module("ExtractMesh", id, Filter)
{
    inports=scinew ScalarFieldIPort(this, "Scalars", ScalarFieldIPort::Atomic);
    add_iport(inports);
    inportv=scinew VectorFieldIPort(this, "Vectors", VectorFieldIPort::Atomic);
    add_iport(inportv);

    // Create the output ports
    outport=scinew MeshOPort(this, "Geometry", MeshIPort::Atomic);
    add_oport(outport);
    ocol=scinew ColumnMatrixOPort(this, "Scalar Values", ColumnMatrixIPort::Atomic);
    add_oport(ocol);
    omat=scinew MatrixOPort(this, "Vector Values", MatrixIPort::Atomic);
    add_oport(omat);
}

ExtractMesh::~ExtractMesh()
{
}

void ExtractMesh::execute()
{
    ScalarFieldHandle sf;
    VectorFieldHandle vf;
    if(inports->get(sf)) {
	ColumnMatrixHandle cmh;
	ScalarFieldUG* ugfield=sf->getUG();
	if(ugfield) {
	    outport->send(ugfield->mesh);
	    ColumnMatrix *cmat=scinew ColumnMatrix(ugfield->data.size());
	    for (int i=0; i<ugfield->data.size(); i++)
		(*cmat)[i]=ugfield->data[i];
	    cmh=cmat;
	} else {
	    ScalarFieldRGBase* rgfield=sf->getRGBase();
	    ColumnMatrix *cmat=scinew ColumnMatrix(rgfield->nx*rgfield->ny*rgfield->nz);
	    int count=0;
	    for (int i=0; i<rgfield->nx; i++)
		for (int j=0; j<rgfield->ny; j++) 
		    for (int k=0; k<rgfield->nz; k++, count++) 
			(*cmat)[count]=rgfield->get_value(i,j,k);
	    cmh=cmat;
	}
	ocol->send(cmh);
    } else if (inportv->get(vf)) {
	MatrixHandle dmh;
	VectorFieldUG* ugfield=vf->getUG();
	if (ugfield) {
	    outport->send(ugfield->mesh);
	    DenseMatrix *dmat=scinew DenseMatrix(ugfield->data.size(), 3);
	    for (int i=0; i<ugfield->data.size(); i++) {
		Vector v=ugfield->data[i];
		(*dmat)[i][0]=v.x(); (*dmat)[i][1]=v.y(); (*dmat)[i][2]=v.z();
	    }
	    dmh=dmat;
	} else {
	    VectorFieldRG* rgfield=vf->getRG();
	    DenseMatrix *dmat=scinew DenseMatrix(rgfield->nx*rgfield->ny*rgfield->nz, 3);
	    int count=0;
	    for (int i=0; i<rgfield->nx; i++)
		for (int j=0; j<rgfield->ny; j++) 
		    for (int k=0; k<rgfield->nz; k++, count++) {
			Vector v=rgfield->grid(i,j,k);
			(*dmat)[count][0]=v.x();
			(*dmat)[count][1]=v.y();
			(*dmat)[count][2]=v.z();
		    }
	    dmh=dmat;
	}
	omat->send(dmh);
    }
}
} // End namespace Modules
} // End namespace SCIRun


//
// $Log$
// Revision 1.2  1999/09/05 23:15:32  dmw
// output values through a second/third port when extracting mesh from a field
//
// Revision 1.1  1999/09/05 01:15:26  dmw
// added all of the old SCIRun mesh modules
//
