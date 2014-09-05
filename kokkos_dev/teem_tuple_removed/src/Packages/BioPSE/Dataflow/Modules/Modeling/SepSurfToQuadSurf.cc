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
 *  SepSurfToQuadSurf.cc:  Extract a single QuadSurf component (boundary
 *                of a single connected component) from a Separating Surface
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   March 2003
 *
 *  Copyright (C) 2003 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Packages/BioPSE/Core/Datatypes/SepSurf.h>
#include <Core/Datatypes/QuadSurfField.h>
#include <Core/GuiInterface/GuiVar.h>
#include <map>
#include <queue>
#include <iostream>

using std::queue;

namespace BioPSE {

using namespace SCIRun;

class SepSurfToQuadSurf : public Module {
  GuiString data_;
public:
  SepSurfToQuadSurf(GuiContext *ctx);
  virtual ~SepSurfToQuadSurf();
  virtual void execute();
};

DECLARE_MAKER(SepSurfToQuadSurf)

SepSurfToQuadSurf::SepSurfToQuadSurf(GuiContext *ctx)
  : Module("SepSurfToQuadSurf", ctx, Filter, "Modeling", "BioPSE"),
    data_(ctx->subVar("data"))    
{
}

SepSurfToQuadSurf::~SepSurfToQuadSurf()
{
}

void SepSurfToQuadSurf::execute() {
  // make sure the ports exist
  FieldIPort *ifp = (FieldIPort *)get_iport("SepSurf");
  FieldHandle ifieldH;
  if (!ifp) {
    error("Unable to initialize iport 'SepSurf'.");
    return;
  }
  FieldOPort *ofp = (FieldOPort *)get_oport("QuadSurf");
  if (!ofp) {
    error("Unable to initialize oport 'QuadSurf'.");
    return;
  }

  // make sure the input data exists
  if (!ifp->get(ifieldH) || !ifieldH.get_rep()) {
    error("No input data");
    return;
  }
  SepSurf *ss = dynamic_cast<SepSurf *>(ifieldH.get_rep());
  if (!ss) {
    error("Input field was not a SepSurf");
    return;
  }

  QuadSurfMeshHandle mesh = new QuadSurfMesh;
  Array1<int> data;
  QuadSurfField<int> *qsf;
  int total_nnodes=0;
  Point p;
  QuadSurfMesh::Node::array_type nodes;
  
  for (int i=0; i<ss->ncomps(); i++) {
    qsf = ss->extractSingleComponent(i, data_.get());
    QuadSurfMeshHandle qsm = qsf->get_typed_mesh();
    QuadSurfMesh::Node::iterator nb, ne;
    QuadSurfMesh::Face::iterator fb, fe;
    QuadSurfMesh::Node::size_type nnodes;
    qsm->begin(nb);
    qsm->end(ne);
    while (nb != ne) {
      qsm->get_center(p, *nb);
      mesh->add_point(p);
      ++nb;
    }
    qsm->begin(fb);
    qsm->end(fe);
    while (fb != fe) {
      qsm->get_nodes(nodes, *fb);
      for (int ii=0; ii<4; ii++) nodes[ii]=((unsigned)nodes[ii])+total_nnodes;
      mesh->add_elem(nodes);
      data.add(qsf->value(*fb));
      ++fb;
    }
    qsm->size(nnodes);
    delete qsf;
    total_nnodes+=nnodes;
  }
  qsf = new QuadSurfField<int>(mesh, Field::FACE);
  QuadSurfMesh::Face::iterator fb, fe;
  int count=0;
  mesh->begin(fb);
  mesh->end(fe);
  while (fb != fe) {
    qsf->set_value(data[count], *fb);
    ++fb;
    ++count;
  }
  //  cerr << "count="<<count<<" data.size()="<<data.size()<<"\n";
  FieldHandle qsfH(qsf);
  ofp->send(qsfH);
}    

} // End namespace BioPSE
