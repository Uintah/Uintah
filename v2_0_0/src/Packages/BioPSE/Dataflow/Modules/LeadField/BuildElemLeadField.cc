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

#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/PointCloudField.h>
#include <Core/Datatypes/TetVolField.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/Ports/FieldPort.h>

#include <Dataflow/Modules/Fields/FieldInfo.h>

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
  FieldIPort *mesh_iport = (FieldIPort *)get_iport("Domain Mesh");
  MatrixIPort *interp_iport = 
    (MatrixIPort *)get_iport("Electrode Interpolant");
  MatrixIPort *sol_iport = (MatrixIPort *)get_iport("Solution Vectors");
  MatrixOPort *rhs_oport = (MatrixOPort *)get_oport("RHS Vector");
  MatrixOPort* leadfield_oport = 
    (MatrixOPort *)get_oport("Leadfield (nelecs x nelemsx3)");

  if (!mesh_iport) {
    error("Unable to initialize iport 'Domain Mesh'.");
    return;
  }
  if (!interp_iport) {
    error("Unable to initialize iport 'Electrode Interpolant'.");
    return;
  }
  if (!sol_iport) {
    error("Unable to initialize iport 'Solution Vectors'.");
    return;
  }
  if (!rhs_oport) {
    error("Unable to initialize oport 'RHS Vector'.");
    return;
  }
  if (!leadfield_oport) {
    error("Unable to initialize oport 'Leadfield (nelecs x nelemsx3)'.");
    return;
  }
  
  int nnodes;
  int nelems;
  FieldHandle mesh_in;
  if (!mesh_iport->get(mesh_in) || !mesh_in.get_rep()) {
    error("Couldn't get input mesh.");
    return;
  }
  const TypeDescription *meshtd = mesh_in->mesh()->get_type_description();
  CompileInfoHandle ci = FieldInfoAlgoCount::get_compile_info(meshtd);
  Handle<FieldInfoAlgoCount> algo;
  if (!module_dynamic_compile(ci, algo)) return;
  algo->execute(mesh_in->mesh(), nnodes, nelems);

  MatrixHandle interp_in;
  if (!interp_iport->get(interp_in) || !interp_in.get_rep()) {
    error("Couldn't get electrode interpolant.");
    return;
  }

  // can't shortcut return, downstream from the send intermediate may be 
  // waiting for output, so don't hang.
  last_mesh_generation_ = mesh_in->generation;
  last_interp_generation_ = interp_in->generation;

  int nelecs=interp_in->nrows();
  int counter=0;
  DenseMatrix *leadfield_mat=new DenseMatrix(nelecs, nelems*3);
  leadfield_mat->zero();

  while (counter<(nelecs-1)) {
    update_progress(counter*1./(nelecs-1));
    ColumnMatrix* rhs=new ColumnMatrix(nnodes);
    int i;
    for (i=0; i<nnodes; i++) (*rhs)[i]=0;

    Array1<int> idx;
    Array1<double> val;

    interp_in->getRowNonzeros(0, idx, val);
    if (!idx.size()) ASSERTFAIL("No mesh node assigned to this element!");
    for (i=0; i<idx.size(); i++) {
      if (idx[i] >= nnodes) ASSERTFAIL("Mesh node out of range!");
      (*rhs)[idx[i]]+=val[i];
    }

    interp_in->getRowNonzeros(counter+1, idx, val);
    if (!idx.size()) ASSERTFAIL("No mesh node assigned to this element!");
    for (i=0; i<idx.size(); i++) {
      if (idx[i] >= nnodes) ASSERTFAIL("Mesh node out of range!");
      (*rhs)[idx[i]]-=val[i];
    }

    if (counter<(nelecs-2)) {
      rhs_oport->send_intermediate(rhs);
    } else {
      rhs_oport->send(rhs);
    }
    // read sol'n
    MatrixHandle sol_in;
    if (!sol_iport->get(sol_in)) {
      error("Couldn't get solution vector.");
      return;
    }
    for (i=0; i<nelems; i++)
      for (int j=0; j<3; j++) {
	(*leadfield_mat)[counter+1][i*3+j]=-(*sol_in.get_rep())[i][j];
      }
    counter++;
  }
  leadfield_=leadfield_mat;
  leadfield_oport->send(leadfield_);
} 
} // End namespace BioPSE
