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

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/Ports/MeshPort.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/DenseMatrix.h>

#include <iostream>
using std::cerr;
using std::endl;
#include <stdio.h>
#include <math.h>

namespace BioPSE {
using namespace SCIRun;

class FieldFromBasis : public Module {    
    MeshIPort* mesh_iport;
    MatrixIPort* basis_iport;
    int matrixGen;
    MatrixIPort* rms_iport;
    MatrixOPort* elem_oport;
    MatrixOPort* vec_oport;
    MatrixHandle vecH;
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
    rms_iport = new MatrixIPort(this, "Element Min Error",
				MatrixIPort::Atomic);
    add_iport(rms_iport);
    elem_oport = new MatrixOPort(this,"Element Vectors",
				 MatrixIPort::Atomic);
    add_oport(elem_oport);
    vec_oport = new MatrixOPort(this, "Error Vector",
				      MatrixIPort::Atomic);
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
	    MatrixHandle err_in;
	    ColumnMatrix* err_inp;
	    if (!rms_iport->get(err_in) || 
		!(err_inp=dynamic_cast<ColumnMatrix*>(err_in.get_rep())) || 
		(err_in->nrows() != 1)) {
		cerr <<"FieldFromBasis -- couldn't get error vector.\n";
		return;
	    }
	    errors[counter]=(*err_inp)[0];
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
	    MatrixHandle err_in;
	    ColumnMatrix* err_inp;
	    if (!rms_iport->get(err_in) || 
		!(err_inp=dynamic_cast<ColumnMatrix*>(err_in.get_rep())) || 
		(err_in->nrows() != 1)) {
		cerr <<"FieldFromBasis -- couldn't get error vector.\n";
		return;
	    }
	    errors[counter]=(*err_inp)[0];
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
	    MatrixHandle err_in;
	    ColumnMatrix* err_inp;
	    if (!rms_iport->get(err_in) || 
		!(err_inp=dynamic_cast<ColumnMatrix*>(err_in.get_rep())) || 
		(err_in->nrows() != 1)) {
		cerr <<"FieldFromBasis -- couldn't get error vector.\n";
		return;
	    }
	    errors[counter]=(*err_inp)[0];
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
} // End namespace BioPSE
