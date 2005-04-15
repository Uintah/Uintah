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

class RecipBasis : public Module {    
    MeshIPort* mesh_iport;
    ColumnMatrixIPort* idx_iport;
    MatrixIPort* sol_iport;
    ColumnMatrixOPort* rhs_oport;
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
    idx_iport = new ColumnMatrixIPort(this, "Electrode Indices",
				       ColumnMatrixIPort::Atomic);
    add_iport(idx_iport);
    sol_iport = new MatrixIPort(this,"Solution Vectors",
				      MatrixIPort::Atomic);
    add_iport(sol_iport);
    rhs_oport = new ColumnMatrixOPort(this,"RHS Vector",
				       ColumnMatrixIPort::Atomic);
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
    ColumnMatrixHandle idx_in;
    if (!idx_iport->get(idx_in)) {
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
	(*rhs)[(*idx_in.get_rep())[0]]=1;
	(*rhs)[(*idx_in.get_rep())[counter+1]]=-1;
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
} // End namespace Modules
} // End namespace DaveW

//
// $Log$
// Revision 1.8.2.4  2000/11/01 23:02:25  mcole
// Fix for previous merge from trunk
//
// Revision 1.8.2.2  2000/10/26 14:02:08  moulding
// merge HEAD into FIELD_REDESIGN
//
// Revision 1.9  2000/08/01 18:03:04  dmw
// fixed errors
//
// Revision 1.8  2000/03/17 09:25:44  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.7  1999/12/09 00:05:25  dmw
// new modules
//
// Revision 1.6  1999/12/02 21:57:32  dmw
// new camera path datatypes and modules
//
// Revision 1.5  1999/10/07 02:06:35  sparker
// use standard iostreams and complex type
//
// Revision 1.4  1999/09/22 18:43:25  dmw
// added new GUI
//
// Revision 1.3  1999/09/16 00:36:55  dmw
// added new Module that Chris Butson will work on (DipoleInSphere) and fixed SRCDIR references in DaveW makefiles
//
// Revision 1.2  1999/09/05 23:16:19  dmw
// build scalar field of error values from Basis Matrix
//
// Revision 1.1  1999/09/05 01:14:02  dmw
// new module for source localization
//
//
