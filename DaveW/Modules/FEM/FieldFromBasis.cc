/*
 *  FieldFromBasis.cc: Build the Reciprocity Basis
 *
 *  Written by:
 *   David Weinstein
 *   University of Utah
 *   September 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/ColumnMatrixPort.h>
#include <PSECore/Datatypes/MatrixPort.h>
#include <PSECore/Datatypes/MeshPort.h>
#include <PSECore/Datatypes/ScalarFieldPort.h>
#include <SCICore/Datatypes/DenseMatrix.h>
#include <SCICore/Datatypes/ScalarFieldUG.h>

#include <iostream>
using std::cerr;
using std::endl;
#include <stdio.h>
#include <math.h>

namespace DaveW {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::Containers;

class FieldFromBasis : public Module {    
    MeshIPort* mesh_iport;
    MatrixIPort* basis_iport;
    int matrixGen;
    ColumnMatrixIPort* rms_iport;
    MatrixOPort* elem_oport;
    ScalarFieldOPort* field_oport;
    ScalarFieldHandle sfh;
public:
    FieldFromBasis(const clString& id);
    virtual ~FieldFromBasis();
    virtual void execute();
};


Module* make_FieldFromBasis(const clString& id) {
    return new FieldFromBasis(id);
}

//---------------------------------------------------------------
FieldFromBasis::FieldFromBasis(const clString& id)
: Module("FieldFromBasis", id, Filter)
{
    mesh_iport = new MeshIPort(this, "Domain Mesh",
				MeshIPort::Atomic);
    add_iport(mesh_iport);
    basis_iport = new MatrixIPort(this,"Basis Matrix",
				  MatrixIPort::Atomic);
    add_iport(basis_iport);
    rms_iport = new ColumnMatrixIPort(this, "Element Min Error",
				       ColumnMatrixIPort::Atomic);
    add_iport(rms_iport);
    elem_oport = new MatrixOPort(this,"Element Vectors",
				 MatrixIPort::Atomic);
    add_oport(elem_oport);
    field_oport = new ScalarFieldOPort(this, "Error Field",
				       ScalarFieldIPort::Atomic);
    add_oport(field_oport);
    matrixGen=-1;
}

FieldFromBasis::~FieldFromBasis(){}

void FieldFromBasis::execute() {
    MeshHandle mesh_in;
    if (!mesh_iport->get(mesh_in)) {
	cerr << "FieldFromBasis -- couldn't get mesh.  Returning.\n";
	return;
    }
    MatrixHandle basisH;
    Matrix* basis;
    if (!basis_iport->get(basisH) || !(basis=basisH.get_rep())) {
	cerr << "FieldFromBasis -- couldn't get basis matrix.  Returning.\n";
	return;
    }
    int nelecs=basis->ncols()/3;
    int nelems=mesh_in->elems.size();
    if (nelems != basis->nrows()) {
	cerr << "FieldFromBasis -- error, basis doesn't have the same number of elements \nas the mesh: basis->nrows()="<<basis->nrows()<<" mesh->elems.size()="<<nelems<<"\n";
	return;
    }

    if (matrixGen == basisH->generation && sfh.get_rep()) {
	field_oport->send(sfh);
	return;
    }
    matrixGen=basisH->generation;

    int counter=0;
    Array1<double> errors(nelems);
    while (counter<nelems) {

	if (counter && counter%100 == 0)
	    cerr << "FieldFromBasis: "<<counter<<"/"<<nelems<<"\n";
	// send element's basis matrix
	DenseMatrix* bas=new DenseMatrix(nelecs,3);
	int i;
	for (i=0; i<nelecs; i++) 
	    for (int j=0; j<3; j++)
		(*bas)[i][j]=(*basis)[counter][i*3+j];
	if (counter<(nelems-1)) elem_oport->send_intermediate(bas);
	else elem_oport->send(bas);

	// read error
	ColumnMatrixHandle err_in;
	if (!rms_iport->get(err_in) || !(err_in.get_rep()) || 
	    (err_in->nrows() != 1)) {
	    cerr <<"FieldFromBasis -- couldn't get error vector.\n";
	    return;
	}
	errors[counter]=(*err_in.get_rep())[0];
	counter++;
    }

    ScalarFieldUG *sfug=new ScalarFieldUG(mesh_in, 
					  ScalarFieldUG::ElementValues);
    sfug->data=errors;
    sfh=sfug;
    field_oport->send(sfh);
    cerr << "Done with the Module!"<<endl;
}
} // End namespace Modules
} // End namespace DaveW

//
// $Log$
// Revision 1.3  1999/10/07 02:06:35  sparker
// use standard iostreams and complex type
//
// Revision 1.2  1999/09/22 18:43:25  dmw
// added new GUI
//
// Revision 1.1  1999/09/05 23:16:19  dmw
// build scalar field of error values from Basis Matrix
//
