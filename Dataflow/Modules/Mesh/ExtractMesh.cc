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

#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/ScalarFieldRGBase.h>
#include <Core/Datatypes/ScalarFieldUG.h>
#include <Core/Datatypes/VectorFieldRG.h>
#include <Core/Datatypes/VectorFieldUG.h>
#include <Core/Geometry/Point.h>
#include <Core/Malloc/Allocator.h>
#include <Dataflow/Ports/ColumnMatrixPort.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/Ports/MeshPort.h>
#include <Dataflow/Ports/ScalarFieldPort.h>
#include <Dataflow/Ports/VectorFieldPort.h>

namespace SCIRun {


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

extern "C" Module* make_ExtractMesh(const clString& id)
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
} // End namespace SCIRun


