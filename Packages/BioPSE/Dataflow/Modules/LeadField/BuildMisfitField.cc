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
 *  BuildMisfitField.cc: Build the lead field matrix through reciprocity
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

// Take in a ColumnMatrix, c,  and a LeadField, L
//   Scan c through all of the elements in L
//   For each element, find the misfit for the optimal moment weights
//   Store the misfit value at the cells (for now, push out to nodes)

namespace BioPSE {

using std::cerr;
using std::endl;
using std::pair;

using namespace SCIRun;


class BuildMisfitField : public Module {    
  FieldIPort* mesh_iport_;
  MatrixIPort* leadfield_iport_;
  MatrixIPort* measurements_iport_;
  MatrixIPort* misfit_iport_;
  FieldOPort* mesh_oport_;
  MatrixOPort* basis_oport_;

  int last_mesh_generation_;
  int last_leadfield_generation_;
  int last_measurements_generation_;
  FieldHandle last_mesh_;
  MatrixHandle last_basis_;
public:
  BuildMisfitField(const string& id);
  virtual ~BuildMisfitField();
  virtual void execute();
};


extern "C" Module* make_BuildMisfitField(const string& id) {
  return new BuildMisfitField(id);
}

//---------------------------------------------------------------
BuildMisfitField::BuildMisfitField(const string& id)
  : Module("BuildMisfitField", id, Filter, "LeadField", "BioPSE"), 
  last_mesh_generation_(-1), last_leadfield_generation_(-1),
  last_measurements_generation_(-1),
  last_mesh_(0), last_basis_(0)
{
}

BuildMisfitField::~BuildMisfitField(){}

void BuildMisfitField::execute() {
  mesh_iport_ = (FieldIPort *)get_iport("FEM Mesh");
  leadfield_iport_ = (MatrixIPort *)get_iport("Leadfield (nelecs x nelemsx3)");
  measurements_iport_ = (MatrixIPort *)get_iport("Measurement Vector");
  misfit_iport_ = (MatrixIPort *)get_iport("Misfit Matrix");
  mesh_oport_ = (FieldOPort *)get_oport("Misfit Field");
  basis_oport_ = (MatrixOPort *)get_oport("Basis");

  if (!mesh_iport_) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  if (!leadfield_iport_) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  if (!measurements_iport_) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  if (!misfit_iport_) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  if (!mesh_oport_) {
    postMessage("Unable to initialize "+name+"'s oport\n");
    return;
  }
  if (!basis_oport_) {
    postMessage("Unable to initialize "+name+"'s oport\n");
    return;
  }

  FieldHandle mesh_in;
  if (!mesh_iport_->get(mesh_in)) {
    cerr << "BuildMisfitField -- couldn't get mesh.  Returning.\n";
    return;
  }
  if (!mesh_in.get_rep() || mesh_in->get_type_name(0)!="TetVol") {
    cerr << "Error - BuildMisfitField didn't get a TetVol for the mesh" << "\n";
    return;
  }
  TetVolMesh* mesh = 
    (TetVolMesh*)dynamic_cast<TetVolMesh*>(mesh_in->mesh().get_rep());

  MatrixHandle leadfield_in;
  if (!leadfield_iport_->get(leadfield_in) || !leadfield_in.get_rep()) {
    cerr << "BuildMisfitField -- couldn't get leadfield.  Returning.\n";
    return;
  }
  DenseMatrix *dm = dynamic_cast<DenseMatrix*>(leadfield_in.get_rep());
  if (!dm) {
    cerr << "BuildMisfitField -- error, leadfield wasn't a DenseMatrix.\n";
    return;
  }

  MatrixHandle measurements_in;
  if (!measurements_iport_->get(measurements_in) || !measurements_in.get_rep()) {
    cerr << "BuildMisfitField -- couldn't get measurement vector.  Returning.\n";
    return;
  }
  ColumnMatrix *cm = dynamic_cast<ColumnMatrix*>(measurements_in.get_rep());
  if (!cm) {
    cerr << "BuildMisfitField -- error, measurement vectors wasn't a ColumnMatrix.\n";
    return;
  }
  if (cm->nrows() != dm->nrows()) {
    cerr << "BuildMisfitField -- error, leadfield ("<<dm->nrows()<<") and measurements ("<<cm->nrows()<<") have different numbers of rows.\n";
    return;
  }
  int nr = cm->nrows();
  TetVolMesh::Cell::size_type csize;  mesh->size(csize);
  int ncells = csize;

  if (ncells * 3 != dm->ncols()) {
    cerr << "BuildMisfitField -- error, leadfield has "<<dm->ncols()<<" columns, and the mesh has "<<ncells<<" cells.\n";
    return;
  }

  if (last_mesh_generation_ == mesh->generation &&
      last_leadfield_generation_ == dm->generation &&
      last_measurements_generation_ == cm->generation) {
    cerr << "BuildMisfitField -- sending same data again.\n";
    mesh_oport_->send(last_mesh_);
    basis_oport_->send(last_basis_);
    MatrixHandle dummy;
    misfit_iport_->get(dummy);
    return;
  }

  cerr << "last-mesh-gen = "<<last_mesh_generation_<<"\n";
  cerr << "last-lf-gen = "<<last_leadfield_generation_<<"\n";
  cerr << "last-meas-gen = "<<last_measurements_generation_<<"\n";
  cerr << "  mesh-gen = "<<mesh->generation<<"\n";
  cerr << "  lf-gen = "<<dm->generation<<"\n";
  cerr << "  meas-gen = "<<cm->generation<<"\n";

  last_mesh_generation_ = mesh->generation;
  last_leadfield_generation_ = dm->generation;
  last_measurements_generation_ = cm->generation;


  double best_val;
  int best_idx;

//  TetVol<double> *tvd = scinew TetVol<double>(TetVolMeshHandle(mesh), 
//					      Field::CELL);
  Array1<int> node_refs(ncells);
  Array1<double> node_sums(ncells);
  node_refs.initialize(0);
  node_sums.initialize(0);

  // ok data is valid, iterate through all of the triple columns
  for (int i=0; i<ncells; i++) {
    if (i>0 && (i%100 == 0))
      cerr << i << "/" << ncells << "\n";
    DenseMatrix *basis = scinew DenseMatrix(nr, 3);
    last_basis_ = basis;
    for (int j=0; j<3; j++) 
      for (int r=0; r<nr; r++) 
	(*basis)[r][j] = (*dm)[r][i*3+j];
    if (i != ncells-1)
      basis_oport_->send_intermediate(last_basis_);
    else
      basis_oport_->send(last_basis_);
    MatrixHandle misfit_in;
    if (!misfit_iport_->get(misfit_in) || !misfit_in.get_rep()) {
      cerr << "BuildMisfitField -- error, wasn't able to read back misfit.\n";
      return;
    }
    if (!misfit_in->nrows() || !misfit_in->ncols()) {
      cerr << "BuildMisfitField -- error, 0-sized misfit matrix.\n";
      return;
    }
    double misfit = (*misfit_in.get_rep())[0][0];

//    tvd->fdata()[i] = misfit;
    TetVolMesh::Node::array_type::iterator ni;
    TetVolMesh::Node::array_type na(4);
    mesh->get_nodes(na,TetVolMesh::Cell::index_type(i));
    for (ni = na.begin(); ni != na.end(); ++ni) {
      node_refs[*ni]++;
      node_sums[*ni]+=misfit;
    }
    
    if (i==0 || misfit<best_val) {
      best_val = misfit;
      best_idx = i;
    }
  }

  // we only know how to isosurface when the data is at the nodes
  TetVol<double> *tvd =
    scinew TetVol<double>(TetVolMeshHandle(mesh), Field::NODE);
  TetVolMesh::Node::size_type nsize;  mesh->size(nsize);
  for (int i=0; i<nsize; i++) 
    tvd->fdata()[i]=node_sums[i]/node_refs[i];

  last_mesh_ = tvd;
  mesh_oport_->send(last_mesh_);
  cerr << "Best misfit was "<<best_val<<", which was cell "<<best_idx<<"\n";
}
} // End namespace BioPSE
