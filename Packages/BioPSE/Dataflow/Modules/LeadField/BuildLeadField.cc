/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
 *  BuildLeadField.cc: Build the Reciprocity Basis
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
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/TetVol.h>

#include <iostream>
using std::cerr;
using std::endl;
#include <stdio.h>
#include <math.h>

namespace BioPSE {
using namespace SCIRun;

class BuildLeadField : public Module {    
  FieldIPort* mesh_iport;
  FieldIPort* interp_iport;
  MatrixIPort* sol_iport;
  MatrixOPort* rhs_oport;
  MatrixOPort* basis_oport;
  MatrixOPort* basis2_oport;
public:
  BuildLeadField(const clString& id);
  virtual ~BuildLeadField();
  virtual void execute();
};


extern "C" Module* make_BuildLeadField(const clString& id) {
  return new BuildLeadField(id);
}

//---------------------------------------------------------------
BuildLeadField::BuildLeadField(const clString& id)
  : Module("BuildLeadField", id, Filter)
{
  mesh_iport = new FieldIPort(this, "Domain Mesh",
			      FieldIPort::Atomic);
  add_iport(mesh_iport);
  interp_iport = new FieldIPort(this, "Electrode Interpolant",
				FieldIPort::Atomic);
  add_iport(interp_iport);
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

BuildLeadField::~BuildLeadField(){}

void BuildLeadField::execute() {
  FieldHandle mesh_in;
  if (!mesh_iport->get(mesh_in)) {
    cerr << "BuildLeadField -- couldn't get mesh.  Returning.\n";
    return;
  }
#if 0 // FIX_ME mesh to TetVol
  MatrixHandle idx_inH;
  ColumnMatrix *idx_in;
  if (!idx_iport->get(idx_inH) || !(idx_in = dynamic_cast<ColumnMatrix *>(idx_inH.get_rep()))) {
    cerr << "BuildLeadField -- couldn't get index vector.  Returning.\n";
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
      cerr <<"BuildLeadField -- couldn't get solution vector.  Returning.\n";
      return;
    }
    for (i=0; i<nelems; i++)
      for (int j=0; j<3; j++) {
	(*bmat)[i][counter*3+j]=-(*sol_in.get_rep())[i][j];
	(*bmat2)[counter+1][i*3+j]=-(*sol_in.get_rep())[i][j];
      }
    cerr << "BuildLeadField: "<<counter<<"/"<<nelecs-1<<"\n";
    counter++;
    
  }
  basis_oport->send(bmat);
  basis2_oport->send(bmat2);
  cerr << "Done with the Module!"<<endl;
#endif
} 
} // End namespace BioPSE
