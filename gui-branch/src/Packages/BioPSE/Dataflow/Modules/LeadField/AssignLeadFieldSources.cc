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
#include <Core/Datatypes/TetVol.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <iostream>
#include <stdio.h>

namespace BioPSE {

using std::cerr;
using std::endl;

using namespace SCIRun;

class AssignLeadFieldSources : public Module
{
  FieldIPort *ifp;
  MatrixIPort *imp;
  FieldOPort *ofp;
  FieldOPort *ofp2;
public:
  AssignLeadFieldSources(const string& id);
  virtual ~AssignLeadFieldSources();
  virtual void execute();
};


extern "C" Module* make_AssignLeadFieldSources(const string& id)
{
  return new AssignLeadFieldSources(id);
}

AssignLeadFieldSources::AssignLeadFieldSources(const string& id)
  : Module("AssignLeadFieldSources", id, Filter, "LeadField", "BioPSE")
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

  if (!ifp) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  if (!imp) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  if (!ofp) {
    postMessage("Unable to initialize "+name+"'s oport\n");
    return;
  }
  if (!ofp2) {
    postMessage("Unable to initialize "+name+"'s oport\n");
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
    remark("Field was supposed to be a TetVol.");
    return;
  }
  TetVolMeshHandle tvmH(tvm);

  TetVolMesh::Cell::size_type csize;  tvm->size(csize);

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
  if (cm->nrows() != csize * 3) {
    remark("ColumnMatrix should be 3x as big as the number of mesh cells.");
    return;
  }

  // data looks good
  // make a new vector field and copy the matrix data into it
  TetVol<Vector> *ofield = scinew TetVol<Vector>(tvmH, Field::CELL);
  TetVol<double> *ofield2 = scinew TetVol<double>(tvmH, Field::NODE);

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

  cerr << "\n\n\nFocusing ``spikes'':\n";
  double halfMax = maxL/2.;
  for (i=0; i<lengths.size(); i++) {
    if (lengths[i]>halfMax) {
      Point p;
      tvm->get_center(p, TetVolMesh::Cell::index_type(i));
      cerr<<"   Magnitude="<<lengths[i]<<" cellIdx="<<i<<" centroid="<<p<<"\n";
    }
  }
  cerr << "End of focusing spikes.\n";

  for (i=0; i<nsize; i++) 
    ofield2->fdata()[i]=node_sums[i]/node_refs[i]*10000;

  ofp->send(FieldHandle(ofield));
  ofp2->send(FieldHandle(ofield2));
}
} // End namespace BioPSE

