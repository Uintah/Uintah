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
#include <Core/Containers/StringUtil.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/PointCloudField.h>
#include <Core/Datatypes/TetVolField.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Math/Mat.h>

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
  FieldOPort* mesh_oport_;

  int last_mesh_generation_;
  int last_leadfield_generation_;
  int last_measurements_generation_;
  FieldHandle last_mesh_;
  string last_metric_;
  double last_pvalue_;
  GuiString metric_;
  GuiString pvalue_;
public:
  BuildMisfitField(GuiContext *context);
  virtual ~BuildMisfitField();
  virtual void execute();
};


DECLARE_MAKER(BuildMisfitField)


//---------------------------------------------------------------
BuildMisfitField::BuildMisfitField(GuiContext *context)
  : Module("BuildMisfitField", context, Filter, "LeadField", "BioPSE"), 
  last_mesh_generation_(-1), last_leadfield_generation_(-1),
  last_measurements_generation_(-1),
  last_mesh_(0), last_metric_(""), last_pvalue_(1),
  metric_(context->subVar("metric")), pvalue_(context->subVar("pvalue"))
{
}

BuildMisfitField::~BuildMisfitField(){}

void BuildMisfitField::execute() {
  mesh_iport_ = (FieldIPort *)get_iport("FEM Mesh");
  leadfield_iport_ = (MatrixIPort *)get_iport("Leadfield (nelecs x nelemsx3)");
  measurements_iport_ = (MatrixIPort *)get_iport("Measurement Vector");
  mesh_oport_ = (FieldOPort *)get_oport("Misfit Field");

  if (!mesh_iport_) {
    error("Unable to initialize iport 'FEM Mesh'.");
    return;
  }
  if (!leadfield_iport_) {
    error("Unable to initialize iport 'Leadfield (nelecs x nelemsx3)'.");
    return;
  }
  if (!measurements_iport_) {
    error("Unable to initialize iport 'Measurement Vector'.");
    return;
  }
  if (!mesh_oport_) {
    error("Unable to initialize oport 'Misfit Field'.");
    return;
  }

  FieldHandle mesh_in;
  if (!mesh_iport_->get(mesh_in)) {
    cerr << "BuildMisfitField -- couldn't get mesh.  Returning.\n";
    return;
  }
  if (!mesh_in.get_rep() || mesh_in->get_type_name(0)!="TetVolField") {
    cerr << "Error - BuildMisfitField didn't get a TetVolField for the mesh" << "\n";
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

  string metric=metric_.get();
  double pvalue;
  string_to_double(pvalue_.get(), pvalue);
 
  if (last_mesh_generation_ == mesh->generation &&
      last_leadfield_generation_ == dm->generation &&
      last_measurements_generation_ == cm->generation &&
      last_metric_ == metric && last_pvalue_ == pvalue) {
    cerr << "BuildMisfitField -- sending same data again.\n";
    mesh_oport_->send(last_mesh_);
    return;
  }

  last_mesh_generation_ = mesh->generation;
  last_leadfield_generation_ = dm->generation;
  last_measurements_generation_ = cm->generation;
  last_metric_ = metric;
  last_pvalue_ = pvalue;

  double best_val;
  int best_idx;

//  TetVolField<double> *tvd = scinew TetVolField<double>(TetVolMeshHandle(mesh), 
//					      Field::CELL);
  Array1<int> node_refs(ncells);
  Array1<double> node_sums(ncells);
  node_refs.initialize(0);
  node_sums.initialize(0);

  double *b = &((*cm)[0]);
  double *bprime = new double[nr];
  double *x = new double[3];
  double *A[3];
  int r, c;
  for (c=0; c<3; c++)
    A[c] = new double[nr];

  // ok data is valid, iterate through all of the triple columns
  for (int i=0; i<ncells; i++) {
//    if (i>0 && (i%100 == 0))
//      cerr << i << "/" << ncells << "\n";
    for (r=0; r<nr; r++)
      for (c=0; c<3; c++)
	A[c][r]=(*dm)[r][i*3+c];
	
    min_norm_least_sq_3(A, b, x, bprime, nr, 1);
    double misfit;
    double avg1=0, avg2=0;
    int iterate;
    for (iterate=0; iterate<nr; iterate++) {
      avg1+=b[iterate];
      avg2+=bprime[iterate];
    }
    avg1/=nr;
    avg2/=nr;
    
    double ccNum=0;
    double ccDenom1=0;
    double ccDenom2=0;
    double rms=0;
    
    for (iterate=0; iterate<nr; iterate++) {
      double shift1=(b[iterate]-avg1);
      double shift2=(bprime[iterate]-avg2);
      
      ccNum+=shift1*shift2;
      ccDenom1+=shift1*shift1;
      ccDenom2+=shift2*shift2;
      //         double tmp=fabs((*ivec1)[iterate]-(*ivec2)[iterate]);
      double tmp=fabs(shift1-shift2);
      if (pvalue==1) rms+=tmp;
      else if (pvalue==2) rms+=tmp*tmp; 
      else rms+=pow(tmp,pvalue);
    }
    rms = pow(rms/nr,1/pvalue);
    double ccDenom=Sqrt(ccDenom1*ccDenom2);
    double cc=Min(ccNum/ccDenom, 1000000.);
    double ccInv=Min(1.0-fabs(ccNum/ccDenom), 1000000.);
    double rmsRel=Min(rms/ccDenom1, 1000000.);
    if (metric == "rms") {
      misfit=rms;
    } else if (metric == "rmsRel") {
      misfit=rmsRel;
    } else if (metric == "invCC") {
      misfit=ccInv;
    } else if (metric == "CC") {
      misfit=cc;
    } else {
      cerr << "BuildMisfitField: error - unknown metric "<<metric<<endl;
      return;
    }

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

  delete[] bprime;
  delete[] x;
  for (c=0; c<3; c++) {
    delete[] A[c];
  }

  // we only know how to isosurface when the data is at the nodes
  TetVolField<double> *tvd =
    scinew TetVolField<double>(TetVolMeshHandle(mesh), Field::NODE);
  TetVolMesh::Node::size_type nsize;  mesh->size(nsize);
  for (int i=0; i<nsize; i++) 
    tvd->fdata()[i]=node_sums[i]/node_refs[i];

  last_mesh_ = tvd;
  mesh_oport_->send(last_mesh_);
  cerr << "Best misfit was "<<best_val<<", which was cell "<<best_idx<<"\n";
}
} // End namespace BioPSE
