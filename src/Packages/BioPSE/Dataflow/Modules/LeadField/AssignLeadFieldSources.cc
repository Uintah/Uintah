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
 *  AssignLeadFieldSources: Assign a leadfield solution vector to the field fdata
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   June 2001
 *
 *  Copyright (C) 2001 SCI Institute
 */

#include <Core/Persistent/Pstreams.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/PointCloudField.h>
#include <Core/Datatypes/TetVolField.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <iostream>
#include <stdio.h>

namespace BioPSE {

using namespace SCIRun;

class AssignLeadFieldSources : public Module
{
  FieldIPort *ifp; 
  MatrixIPort *imp;
  FieldOPort *ofp;
  FieldOPort *ofp2;
  FieldOPort *ofp3;
  FieldOPort *ofp4;
public:
  AssignLeadFieldSources(GuiContext *context);
  virtual ~AssignLeadFieldSources();
  virtual void execute();
};


DECLARE_MAKER(AssignLeadFieldSources)


AssignLeadFieldSources::AssignLeadFieldSources(GuiContext *context)
  : Module("AssignLeadFieldSources", context, Filter, "LeadField", "BioPSE")
{
}

AssignLeadFieldSources::~AssignLeadFieldSources()
{
}

void
AssignLeadFieldSources::execute()
{

  ifp = (FieldIPort*)get_iport("Mesh");
  imp = (MatrixIPort*)get_iport("Data");

  ofp = (FieldOPort*)get_oport("VectorField");
  ofp2 = (FieldOPort*)get_oport("Field(double)");
  ofp3 = (FieldOPort*)get_oport("PointCloud(Vector)");
  ofp4 = (FieldOPort*)get_oport("PrimaryPointCloud(Vector)");

  if (!ifp) {
    error("Unable to initialize iport 'Mesh'.");
    return;
  }
  if (!imp) {
    error("Unable to initialize iport 'Data'.");
    return;
  }
  if (!ofp) {
    error("Unable to initialize oport 'VectorField'.");
    return;
  }
  if (!ofp2) {
    error("Unable to initialize oport 'Field(double)'.");
    return;
  }
  if (!ofp3) {
    error("Unable to initialize oport 'PointCloud(Vector)'.");
    return;
  }
  if (!ofp4) {
    error("Unable to initialize oport 'PrimaryPointCloud(Vector)'.");
    return;
  }
  
  // Get input matrix.
  MatrixHandle imatrix;
  if (!(imp->get(imatrix)) || !(imatrix.get_rep())) {
    remark("No input matrix.");
    return;
  }
  ColumnMatrix *cm = dynamic_cast<ColumnMatrix *>(imatrix.get_rep());
  if (!cm) {
    remark("Matrix was supposed to be a ColumnMatrix.");
    return;
  }

  // Get input field.
  FieldHandle ifield;
  if (!(ifp->get(ifield)) || !(ifield.get_rep())) {
    remark("No input field.");
    return;
  }
  MeshHandle mbh = ifield->mesh();
  TetVolMesh *tvm = dynamic_cast<TetVolMesh *>(mbh.get_rep());
  if (!tvm) {
    remark("Field was supposed to be a TetVolField.");
    return;
  }
  TetVolMeshHandle tvmH(tvm);

  TetVolMesh::Cell::size_type csize;  tvm->size(csize);

  if (cm->nrows() != csize * 3) {
    remark("ColumnMatrix should be 3x as big as the number of mesh cells.");
    return;
  }

  // data looks good
  // make a new vector field and copy the matrix data into it
  TetVolField<Vector> *ofield = scinew TetVolField<Vector>(tvmH, Field::CELL);
  TetVolField<double> *ofield2 = scinew TetVolField<double>(tvmH, Field::NODE);

  TetVolMesh::Node::size_type nsize;  tvm->size(nsize);

  Array1<int> node_refs(nsize);
  Array1<double> node_sums(nsize);
  node_refs.initialize(0);
  node_sums.initialize(0);

  Array1<double> lengths(csize);
  double maxL=0;

  int i;
  for (i=0; i<cm->nrows()/3; i++) {
    Vector v((*cm)[i*3], (*cm)[i*3+1], (*cm)[i*3+2]);
    ofield->fdata()[i] = v;
    double l=v.length();
    lengths[i]=l;
    if (l>maxL) maxL=l;
    // get all of the nodes that are attached to this cell
    // for each one, increment its count and add this value to its sum
    TetVolMesh::Node::array_type::iterator ni;
    TetVolMesh::Node::array_type na(4);
    tvm->get_nodes(na,TetVolMesh::Cell::index_type(i));
    for (ni = na.begin(); ni != na.end(); ++ni) {
      node_refs[*ni]++;
      node_sums[*ni]+=l;
    }
  }

  vector<Vector> vecs;
  vector<Vector> vecs2;
  PointCloudMesh *pcm = scinew PointCloudMesh;
  PointCloudMesh *pcm2 = scinew PointCloudMesh;
  msgStream_ << "\n\n\nFocusing ``spikes'':\n";
  double halfMax = maxL/2.;
  for (i=0; i<lengths.size(); i++) {
    Point p;
    tvm->get_center(p, TetVolMesh::Cell::index_type(i));
    if (lengths[i]>0.000001) {
      pcm->add_point(p);
      vecs.push_back(ofield->fdata()[i]);
    }
    if (lengths[i]>halfMax) {
      pcm2->add_point(p);
      vecs2.push_back(ofield->fdata()[i]);
      msgStream_ << "   Magnitude="<<lengths[i]<<" cellIdx="<<i<<" centroid="<<p<<"\n";
    }
  }
  msgStream_ << "End of focusing spikes.\n";
  PointCloudMeshHandle pcmH(pcm);
  PointCloudField<Vector> *pc = scinew PointCloudField<Vector>(pcmH, Field::NODE);
  pc->fdata()=vecs;

  PointCloudMeshHandle pcm2H(pcm2);
  PointCloudField<Vector> *pc2 = scinew PointCloudField<Vector>(pcm2H, Field::NODE);
  pc2->fdata()=vecs2;

  for (i=0; i<nsize; i++) 
    ofield2->fdata()[i]=node_sums[i]/node_refs[i];

  ofp->send(FieldHandle(ofield));
  ofp2->send(FieldHandle(ofield2));
  ofp3->send(FieldHandle(pc));
  ofp4->send(FieldHandle(pc2));
}
} // End namespace BioPSE

