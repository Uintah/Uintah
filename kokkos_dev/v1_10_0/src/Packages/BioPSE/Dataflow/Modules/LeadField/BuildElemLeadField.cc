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
 *  BuildElemLeadField.cc: Build the lead field matrix through reciprocity
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
#include <Core/Datatypes/PointCloudField.h>
#include <Core/Datatypes/TetVolField.h>

#include <iostream>
#include <stdio.h>
#include <math.h>


namespace SCIRun {
vector<pair<TetVolMesh::Node::index_type, double> > 
operator*(const vector<pair<TetVolMesh::Node::index_type, double> >&r, double &) {
  ASSERTFAIL("BuildElemLeadField.cc Bogus operator");
  return r;
}
vector<pair<TetVolMesh::Node::index_type, double> > 
operator+=(const vector<pair<TetVolMesh::Node::index_type, double> >&r, 
	   const vector<pair<TetVolMesh::Node::index_type, double> >&) {
  ASSERTFAIL("BuildElemLeadField.cc Bogus operator");
  return r;
}
}

namespace BioPSE {

using std::pair;

using namespace SCIRun;


class BuildElemLeadField : public Module {    
  FieldIPort* mesh_iport_;
  FieldIPort* interp_iport_;
  MatrixIPort* sol_iport_;
  MatrixOPort* rhs_oport_;
  MatrixOPort* leadfield_oport_;

  MatrixHandle leadfield_;
  int last_mesh_generation_;
  int last_interp_generation_;
public:
  BuildElemLeadField(GuiContext *context);
  virtual ~BuildElemLeadField();
  virtual void execute();
};


DECLARE_MAKER(BuildElemLeadField)


//---------------------------------------------------------------
BuildElemLeadField::BuildElemLeadField(GuiContext *context)
  : Module("BuildElemLeadField", context, Filter, "LeadField", "BioPSE"),
    leadfield_(0),
    last_mesh_generation_(-1),
    last_interp_generation_(-1)
{
}

BuildElemLeadField::~BuildElemLeadField(){}

void BuildElemLeadField::execute() {
  mesh_iport_ = (FieldIPort *)get_iport("Domain Mesh");
  interp_iport_ = (FieldIPort *)get_iport("Electrode Interpolant");
  sol_iport_ = (MatrixIPort *)get_iport("Solution Vectors");
  rhs_oport_ = (MatrixOPort *)get_oport("RHS Vector");
  leadfield_oport_ = (MatrixOPort *)get_oport("Leadfield (nelecs x nelemsx3)");

  if (!mesh_iport_) {
    error("Unable to initialize iport 'Domain Mesh'.");
    return;
  }
  if (!interp_iport_) {
    error("Unable to initialize iport 'Electrode Interpolant'.");
    return;
  }
  if (!sol_iport_) {
    error("Unable to initialize iport 'Solution Vectors'.");
    return;
  }
  if (!rhs_oport_) {
    error("Unable to initialize oport 'RHS Vector'.");
    return;
  }
  if (!leadfield_oport_) {
    error("Unable to initialize oport 'Leadfield (nelecs x nelemsx3)'.");
    return;
  }
  

  FieldHandle mesh_in;
  if (!mesh_iport_->get(mesh_in)) {
    error("Couldn't get input mesh.");
    return;
  }
  if (!mesh_in.get_rep() || mesh_in->get_type_name(0)!="TetVolField") {
    error("Expected a TetVolField for the mesh.");
    return;
  }
  TetVolMesh* mesh = 
    (TetVolMesh*)dynamic_cast<TetVolMesh*>(mesh_in->mesh().get_rep());

  FieldHandle interp_in;
  if (!interp_iport_->get(interp_in)) {
    error("Couldn't get electrode interpolant.");
    return;
  }
  if (!interp_in.get_rep() || interp_in->get_type_name(0)!="PointCloudField") {
    error("Didn't get a PointCloudField for interp.");
    return;
  }
  PointCloudMesh* interp_mesh = 
    (PointCloudMesh*)dynamic_cast<PointCloudMesh*>(interp_in->mesh().get_rep());
  //  PointCloudField<vector<pair<TetVolMesh::Node::index_type, double> > >* interp = 
  //    dynamic_cast<PointCloudField<vector<pair<TetVolMesh::Node::index_type, double> > > *>(interp_in.get_rep());

  PointCloudField<vector<pair<TetVolMesh::Node::index_type, double> > >* interp = 
    (PointCloudField<vector<pair<TetVolMesh::Node::index_type, double> > > *)(interp_in.get_rep());

  if (!interp) {
    error("Interp Field wasn't a PointCloudField<vector<pair<TetVolMesh::Node::index_type,double>>>. It's a '" + interp_in->get_type_description()->get_name() + "'.");
    return;
  }

  // can't shortcut return, downstream from the send intermediate may be 
  // waiting for output, so don't hang.
  last_mesh_generation_ = mesh_in->generation;
  last_interp_generation_ = interp_in->generation;

  PointCloudMesh::Node::size_type insize;  interp_mesh->size(insize);
  TetVolMesh::Node::size_type nsize;  mesh->size(nsize);
  TetVolMesh::Cell::size_type csize;  mesh->size(csize);
  int nelecs=insize;
  int nnodes=nsize;
  int nelems=csize;
  int counter=0;
  DenseMatrix *leadfield_mat=new DenseMatrix(nelecs, nelems*3);
  leadfield_mat->zero();
  
  while (counter<(nelecs-1)) {
    ColumnMatrix* rhs=new ColumnMatrix(nnodes);
    int i;
    for (i=0; i<nnodes; i++) (*rhs)[i]=0;
    
    vector<pair<TetVolMesh::Node::index_type,double> >::iterator iter = interp->fdata()[0].begin();
    while (iter != interp->fdata()[0].end()) {
      int idx = (*iter).first;
      double val = (*iter).second;
      (*rhs)[idx]+=val;
      ++iter;
    }

    iter = interp->fdata()[counter+1].begin();
    while (iter != interp->fdata()[counter+1].end()) {
      int idx = (*iter).first;
      double val = (*iter).second;
      (*rhs)[idx]-=val;
      ++iter;
    }
    
    if (counter<(nelecs-2)) {
      rhs_oport_->send_intermediate(rhs);
    } else {
      rhs_oport_->send(rhs);
    }
    // read sol'n
    MatrixHandle sol_in;
    if (!sol_iport_->get(sol_in)) {
      error("Couldn't get solution vector.");
      return;
    }
    for (i=0; i<nelems; i++)
      for (int j=0; j<3; j++) {
	(*leadfield_mat)[counter+1][i*3+j]=-(*sol_in.get_rep())[i][j];
      }
    msgStream_ << "BuildElemLeadField: "<<counter+1<<"/"<<nelecs-1<<"\n";
    counter++;
    
  }
  leadfield_=leadfield_mat;
  leadfield_oport_->send(leadfield_);
} 
} // End namespace BioPSE
