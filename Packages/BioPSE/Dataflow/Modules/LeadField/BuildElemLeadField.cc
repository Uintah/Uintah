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
#include <Core/Datatypes/PointCloud.h>
#include <Core/Datatypes/TetVol.h>

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

using std::cerr;
using std::endl;
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
  BuildElemLeadField(const string& id);
  virtual ~BuildElemLeadField();
  virtual void execute();
};


extern "C" Module* make_BuildElemLeadField(const string& id) {
  return new BuildElemLeadField(id);
}

//---------------------------------------------------------------
BuildElemLeadField::BuildElemLeadField(const string& id)
  : Module("BuildElemLeadField", id, Filter, "LeadField", "BioPSE"), leadfield_(0),
    last_mesh_generation_(-1), last_interp_generation_(-1)
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
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  if (!interp_iport_) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  if (!sol_iport_) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  if (!rhs_oport_) {
    postMessage("Unable to initialize "+name+"'s oport\n");
    return;
  }
  if (!leadfield_oport_) {
    postMessage("Unable to initialize "+name+"'s oport\n");
    return;
  }
  

  FieldHandle mesh_in;
  if (!mesh_iport_->get(mesh_in)) {
    cerr << "BuildElemLeadField -- couldn't get mesh.  Returning.\n";
    return;
  }
  if (!mesh_in.get_rep() || mesh_in->get_type_name(0)!="TetVol") {
    cerr << "Error - BuildElemLeadField didn't get a TetVol for the mesh" << "\n";
    return;
  }
  TetVolMesh* mesh = 
    (TetVolMesh*)dynamic_cast<TetVolMesh*>(mesh_in->mesh().get_rep());

  FieldHandle interp_in;
  if (!interp_iport_->get(interp_in)) {
    cerr << "BuildElemLeadField -- couldn't get interp.  Returning.\n";
    return;
  }
  if (!interp_in.get_rep() || interp_in->get_type_name(0)!="PointCloud") {
    cerr << "Error - BuildElemLeadField didn't get a PointCloud for interp" << "\n";
    return;
  }
  PointCloudMesh* interp_mesh = 
    (PointCloudMesh*)dynamic_cast<PointCloudMesh*>(interp_in->mesh().get_rep());
  //  PointCloud<vector<pair<TetVolMesh::Node::index_type, double> > >* interp = 
  //    dynamic_cast<PointCloud<vector<pair<TetVolMesh::Node::index_type, double> > > *>(interp_in.get_rep());

  PointCloud<vector<pair<TetVolMesh::Node::index_type, double> > >* interp = 
    (PointCloud<vector<pair<TetVolMesh::Node::index_type, double> > > *)(interp_in.get_rep());

  if (!interp) {
    cerr << "Error - Interp Field wasn't a PointCloud<vector<pair<TetVolMesh::Node::index_type,double>>>\n";

    cout << "It's a '" + interp_in->get_type_description()->get_name() + "'\n";
    return;
  }

  if (leadfield_.get_rep() && 
      mesh_in->generation == last_mesh_generation_ &&
      interp_in->generation == last_interp_generation_) {
    leadfield_oport_->send(leadfield_);
    return;
  }
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

    if (counter<(nelecs-2)) rhs_oport_->send_intermediate(rhs);
    else rhs_oport_->send(rhs);
    
    // read sol'n
    MatrixHandle sol_in;
    if (!sol_iport_->get(sol_in)) {
      cerr <<"BuildElemLeadField -- couldn't get solution vector.  Returning.\n";
      return;
    }
    for (i=0; i<nelems; i++)
      for (int j=0; j<3; j++) {
	(*leadfield_mat)[counter+1][i*3+j]=-(*sol_in.get_rep())[i][j];
      }
    cerr << "BuildElemLeadField: "<<counter<<"/"<<nelecs-1<<"\n";
    counter++;
    
  }
  leadfield_=leadfield_mat;
  leadfield_oport_->send(leadfield_);
  cerr << "Done with the Module!"<<endl;
} 
} // End namespace BioPSE
