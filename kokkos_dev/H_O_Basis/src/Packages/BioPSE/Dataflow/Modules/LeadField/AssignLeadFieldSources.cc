/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
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

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Basis/TetLinearLgn.h>
#include <Core/Datatypes/TetVolMesh.h>
#include <Core/Basis/Constant.h>
#include <Core/Datatypes/PointCloudMesh.h>
#include <Core/Datatypes/GenericField.h>
#include <iostream>
#include <stdio.h>

namespace BioPSE {

using namespace SCIRun;

class AssignLeadFieldSources : public Module
{
  typedef ConstantBasis<Vector>                           TVVBasis;
  typedef TetLinearLgn<double>                            TVSBasis;
  typedef TetVolMesh<TetLinearLgn<Point> >                TVMesh;
  typedef GenericField<TVMesh, TVSBasis, vector<double> > TVFieldD;
  typedef GenericField<TVMesh, TVVBasis, vector<Vector> > TVFieldV;
  typedef PointCloudMesh<ConstantBasis<Point> >                 PCMesh;
  typedef ConstantBasis<Vector>                                 FDCVectorBasis;
  typedef GenericField<PCMesh, FDCVectorBasis, vector<Vector> > PCField;
  
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
  TVMesh *tvm = dynamic_cast<TVMesh *>(mbh.get_rep());
  if (!tvm) {
    remark("Field was supposed to be a TetVolField.");
    return;
  }
  TVMesh::handle_type tvmH(tvm);

  TVMesh::Cell::size_type csize;  tvm->size(csize);

  if ((unsigned)cm->nrows() != csize * 3) {
    remark("ColumnMatrix should be 3x as big as the number of mesh cells.");
    return;
  }

  // data looks good
  // make a new vector field and copy the matrix data into it
  TVFieldV *ofield = scinew TVFieldV(tvmH);
  TVFieldD *ofield2 = scinew TVFieldD(tvmH);

  TVMesh::Node::size_type nsize;  tvm->size(nsize);

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
    TVMesh::Node::array_type::iterator ni;
    TVMesh::Node::array_type na;
    tvm->get_nodes(na,TVMesh::Cell::index_type(i));
    for (ni = na.begin(); ni != na.end(); ++ni) {
      node_refs[*ni]++;
      node_sums[*ni]+=l;
    }
  }

  vector<Vector> vecs;
  vector<Vector> vecs2;
  PCMesh *pcm = scinew PCMesh;
  PCMesh *pcm2 = scinew PCMesh;
  msgStream_ << "\n\n\nFocusing ``spikes'':\n";
  double halfMax = maxL/2.;
  for (i=0; i<lengths.size(); i++) {
    Point p;
    tvm->get_center(p, TVMesh::Cell::index_type(i));
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
  PCMesh::handle_type pcmH(pcm);
  PCField *pc = scinew PCField(pcmH);
  pc->fdata()=vecs;

  PCMesh::handle_type pcm2H(pcm2);
  PCField *pc2 = scinew PCField(pcm2H);
  pc2->fdata()=vecs2;

  for (unsigned int ui=0; ui<nsize; ui++) 
    ofield2->fdata()[ui]=node_sums[ui]/node_refs[ui];

  ofp->send(FieldHandle(ofield));
  ofp2->send(FieldHandle(ofield2));
  ofp3->send(FieldHandle(pc));
  ofp4->send(FieldHandle(pc2));
}
} // End namespace BioPSE

