/*
 *  RecipBasis.cc: Build the Reciprocity Basis
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
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/ColumnMatrix.h>

#include <iostream>
using std::cerr;
using std::endl;
#include <stdio.h>
#include <math.h>

namespace BioPSE {
using namespace SCIRun;

class RecipBasis : public Module {    
    MeshIPort* mesh_iport;
    MatrixIPort* idx_iport;
    MatrixIPort* sol_iport;
    MatrixOPort* rhs_oport;
    MatrixOPort* basis_oport;
    MatrixOPort* basis2_oport;
public:
    RecipBasis(const clString& id);
    virtual ~RecipBasis();
    virtual void execute();
};


extern "C" Module* make_RecipBasis(const clString& id) {
    return new RecipBasis(id);
}

//---------------------------------------------------------------
RecipBasis::RecipBasis(const clString& id)
: Module("RecipBasis", id, Filter)
{
    mesh_iport = new MeshIPort(this, "Domain Mesh",
				MeshIPort::Atomic);
    add_iport(mesh_iport);
    idx_iport = new MatrixIPort(this, "Electrode Indices",
				       MatrixIPort::Atomic);
    add_iport(idx_iport);
    sol_iport = new MatrixIPort(this,"Solution Vectors",
				      MatrixIPort::Atomic);
    add_iport(sol_iport);
    rhs_oport = new MatrixOPort(this,"RHS Vector",
				       MatrixIPort::Atomic);
    add_oport(rhs_oport);
    basis_oport = new MatrixOPort(this, "EBasis (nelems x (nelecs-1)x3)",
				  MatrixIPort::Atomic);
    add_oport(basis_oport);
    basis2_oport = new MatrixOPort(this, "EBasis (nelecs x nelemsx3)",
				  MatrixIPort::Atomic);
    add_oport(basis2_oport);
}

RecipBasis::~RecipBasis(){}

void RecipBasis::execute() {
    MeshHandle mesh_in;
    if (!mesh_iport->get(mesh_in)) {
	cerr << "RecipBasis -- couldn't get mesh.  Returning.\n";
	return;
    }
    MatrixHandle idx_inH;
    ColumnMatrix *idx_in;
    if (!idx_iport->get(idx_inH) || !(idx_in = dynamic_cast<ColumnMatrix *>(idx_inH.get_rep()))) {
	cerr << "RecipBasis -- couldn't get index vector.  Returning.\n";
	return;
    }
    int nelecs=idx_in->nrows();
    int nnodes=mesh_in->nodes.size();
    int nelems=mesh_in->elems.size();
    int counter=0;
    DenseMatrix *bmat=new DenseMatrix(nelems, (nelecs-1)*3);
    DenseMatrix *bmat2=new DenseMatrix(nelecs, nelems*3);
    bmat2->zero();

    while (counter<(nelecs-1)) {
	// send rhs
	ColumnMatrix* rhs=new ColumnMatrix(nnodes);
	int i;
	for (i=0; i<nnodes; i++) (*rhs)[i]=0;
	(*rhs)[(*idx_in)[0]]=1;
	(*rhs)[(*idx_in)[counter+1]]=-1;
	if (counter<(nelecs-2)) rhs_oport->send_intermediate(rhs);
	else rhs_oport->send(rhs);

	// read sol'n
	MatrixHandle sol_in;
	if (!sol_iport->get(sol_in)) {
	    cerr <<"RecipBasis -- couldn't get solution vector.  Returning.\n";
	    return;
	}
	for (i=0; i<nelems; i++)
	    for (int j=0; j<3; j++) {
		(*bmat)[i][counter*3+j]=-(*sol_in.get_rep())[i][j];
		(*bmat2)[counter+1][i*3+j]=-(*sol_in.get_rep())[i][j];
	    }
	cerr << "RecipBasis: "<<counter<<"/"<<nelecs-1<<"\n";
	counter++;

    }
    basis_oport->send(bmat);
    basis2_oport->send(bmat2);
    cerr << "Done with the Module!"<<endl;
} 
} // End namespace BioPSE
