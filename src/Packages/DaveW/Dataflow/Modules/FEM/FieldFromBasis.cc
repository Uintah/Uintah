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
#include <SCICore/Datatypes/DenseMatrix.h>

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
    ColumnMatrixOPort* vec_oport;
    ColumnMatrixHandle vecH;
public:
    FieldFromBasis(const clString& id);
    virtual ~FieldFromBasis();
    virtual void execute();
};


extern "C" Module* make_FieldFromBasis(const clString& id) {
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
    vec_oport = new ColumnMatrixOPort(this, "Error Vector",
				      ColumnMatrixIPort::Atomic);
    add_oport(vec_oport);
    matrixGen=-1;
}

FieldFromBasis::~FieldFromBasis(){}

void FieldFromBasis::execute() {
    MeshHandle mesh;
    if (!mesh_iport->get(mesh)) {
	cerr << "FieldFromBasis -- couldn't get mesh.  Returning.\n";
	return;
    }
    MatrixHandle basisH;
    Matrix* basis;
    if (!basis_iport->get(basisH) || !(basis=basisH.get_rep())) {
	cerr << "FieldFromBasis -- couldn't get basis matrix.  Returning.\n";
	return;
    }
    int nelems=mesh->elems.size();
    int nnodes=mesh->nodes.size();
    cerr << "basisH->nrows()="<<basisH->nrows()<<" cols="<<basisH->ncols()<<"\n";
    cerr << "     mesh->elems.size()="<<nelems<<" mesh->nodes.size()="<<nnodes<<"\n";
    if (matrixGen == basisH->generation && vecH.get_rep()) {
	vec_oport->send(vecH);
	return;
    }
    matrixGen=basisH->generation;
    int counter=0;

    ColumnMatrix *vec=new ColumnMatrix(nelems);
    double *errors = vec->get_rhs();

    while (counter<nelems) {
	if (counter && counter%100 == 0)
	    cerr << "FieldFromBasis: "<<counter<<"/"<<nelems<<"\n";
	// send element's basis matrix
	if (nelems == basis->nrows()) {
	    int nelecs=basis->ncols()/3;
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
	} else if (nelems == basis->ncols()/3) {
	    int nelecs=basis->nrows();
	    DenseMatrix* bas=new DenseMatrix(nelecs,3);
	    int i;
	    for (i=0; i<nelecs; i++) 
		for (int j=0; j<3; j++)
		    (*bas)[i][j]=(*basis)[i][counter*3+j];
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
	} else if (nnodes == basis->ncols()) {
	    int nelecs=basis->nrows();
	    DenseMatrix* bas=new DenseMatrix(nelecs,4);
	    Element *e=mesh->elems[counter];
	    int i;
	    for (i=0; i<nelecs; i++) 
		for (int j=0; j<4; j++)
		    (*bas)[i][j]=(*basis)[i][e->n[j]];
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
	} else {
	    cerr << "FieldFromBasis -- error, basis doesn't have the same number of rows \as mesh elements or mesh nodes: basis->nrows()="<<basis->nrows()<<" mesh->elems.size()="<<nelems<<" mesh->nodes.size()="<<nnodes<<"\n";
	    return;
	}
    }

    vecH=vec;
    vec_oport->send(vecH);
    cerr << "Done with the Module!"<<endl;
}
} // End namespace Modules
} // End namespace DaveW

//
// $Log$
// Revision 1.6  2000/08/01 18:03:03  dmw
// fixed errors
//
// Revision 1.5  2000/03/17 09:25:44  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.4  1999/12/10 07:00:08  dmw
// build field from either reciprocity or RA-1 basis
//
// Revision 1.3  1999/10/07 02:06:35  sparker
// use standard iostreams and complex type
//
// Revision 1.2  1999/09/22 18:43:25  dmw
// added new GUI
//
// Revision 1.1  1999/09/05 23:16:19  dmw
// build scalar field of error values from Basis Matrix
//
