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
 *  BuildLeadField.cc: Build the lead field matrix through reciprocity
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
#include <stdio.h>
#include <math.h>


namespace SCIRun {
vector<pair<int, double> > 
operator*(const vector<pair<int, double> >&r, double &) {
  ASSERTFAIL("BuildLeadField.cc Bogus operator");
  return r;
}
vector<pair<int, double> > 
operator+=(const vector<pair<int, double> >&r, 
	   const vector<pair<int, double> >&) {
  ASSERTFAIL("BuildLeadField.cc Bogus operator");
  return r;
}
}

namespace BioPSE {

using std::cerr;
using std::endl;
using std::pair;

using namespace SCIRun;


class BuildLeadField : public Module {    
  FieldIPort* mesh_iport_;
  FieldIPort* interp_iport_;
  MatrixIPort* sol_iport_;
  MatrixOPort* rhs_oport_;
  MatrixOPort* leadfield_oport_;

  MatrixHandle leadfield_;
  int last_mesh_generation_;
  int last_interp_generation_;
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
  : Module("BuildLeadField", id, Filter), leadfield_(0),
    last_mesh_generation_(-1), last_interp_generation_(-1)
{
  mesh_iport_ = new FieldIPort(this, "Domain Mesh",
			      FieldIPort::Atomic);
  add_iport(mesh_iport_);
  interp_iport_ = new FieldIPort(this, "Electrode Interpolant",
				FieldIPort::Atomic);
  add_iport(interp_iport_);
  sol_iport_ = new MatrixIPort(this,"Solution Vectors",
			      MatrixIPort::Atomic);
  add_iport(sol_iport_);
  rhs_oport_ = new MatrixOPort(this,"RHS Vector",
			      MatrixIPort::Atomic);
  add_oport(rhs_oport_);
  leadfield_oport_ = new MatrixOPort(this, "Leadfield (nelecs x nelemsx3)",
				 MatrixIPort::Atomic);
  add_oport(leadfield_oport_);
}

BuildLeadField::~BuildLeadField(){}

void BuildLeadField::execute() {
  FieldHandle mesh_in;
  if (!mesh_iport_->get(mesh_in)) {
    cerr << "BuildLeadField -- couldn't get mesh.  Returning.\n";
    return;
  }
  if (!mesh_in.get_rep() || mesh_in->get_type_name(0)!="TetVol") {
    cerr << "Error - BuildLeadField didn't get a TetVol for the mesh" << "\n";
    return;
  }
  TetVolMesh* mesh = 
    (TetVolMesh*)dynamic_cast<TetVolMesh*>(mesh_in->mesh().get_rep());

  FieldHandle interp_in;
  if (!interp_iport_->get(interp_in)) {
    cerr << "BuildLeadField -- couldn't get interp.  Returning.\n";
    return;
  }
  if (!interp_in.get_rep() || interp_in->get_type_name(0)!="TetVol") {
    cerr << "Error - BuildLeadField didn't get a TetVol for interp" << "\n";
    return;
  }
  TetVolMesh* interp_mesh = 
    (TetVolMesh*)dynamic_cast<TetVolMesh*>(interp_in->mesh().get_rep());

  TetVol<vector<pair<int,double> > >* interp = 
    (TetVol<vector<pair<int,double> > >*)
    dynamic_cast<TetVol<vector<pair<int,double> > >*>(interp_in.get_rep());
  
  if (leadfield_.get_rep() && 
      mesh_in->generation == last_mesh_generation_ &&
      interp_in->generation == last_interp_generation_) {
    leadfield_oport_->send(leadfield_);
    return;
  }
  last_mesh_generation_ = mesh_in->generation;
  last_interp_generation_ = interp_in->generation;

  int nelecs=interp_mesh->nodes_size();
  int nnodes=mesh->nodes_size();
  int nelems=mesh->cells_size();
  int counter=0;
  DenseMatrix *leadfield_mat=new DenseMatrix(nelecs, nelems*3);
  leadfield_mat->zero();
  
  while (counter<(nelecs-1)) {
    ColumnMatrix* rhs=new ColumnMatrix(nnodes);
    int i;
    for (i=0; i<nnodes; i++) (*rhs)[i]=0;
    
    vector<pair<int,double> >::iterator iter = interp->fdata()[0].begin();
    while (iter != interp->fdata()[0].end()) {
      int idx = (*iter).first;
      double val = (*iter).second;
      (*rhs)[idx]+=val;
    }

    iter = interp->fdata()[counter+1].begin();
    while (iter != interp->fdata()[counter+1].end()) {
      int idx = (*iter).first;
      double val = (*iter).second;
      (*rhs)[idx]-=val;
    }

    if (counter<(nelecs-2)) rhs_oport_->send_intermediate(rhs);
    else rhs_oport_->send(rhs);
    
    // read sol'n
    MatrixHandle sol_in;
    if (!sol_iport_->get(sol_in)) {
      cerr <<"BuildLeadField -- couldn't get solution vector.  Returning.\n";
      return;
    }
    for (i=0; i<nelems; i++)
      for (int j=0; j<3; j++) {
	(*leadfield_mat)[counter+1][i*3+j]=-(*sol_in.get_rep())[i][j];
      }
    cerr << "BuildLeadField: "<<counter<<"/"<<nelecs-1<<"\n";
    counter++;
    
  }
  leadfield_=leadfield_mat;
  leadfield_oport_->send(leadfield_);
  cerr << "Done with the Module!"<<endl;
} 
} // End namespace BioPSE
