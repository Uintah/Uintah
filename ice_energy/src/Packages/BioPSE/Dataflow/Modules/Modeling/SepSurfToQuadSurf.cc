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
  // Make sure the ports exist.
  FieldIPort *ifp = (FieldIPort *)get_iport("SepSurf");
  FieldOPort *ofp = (FieldOPort *)get_oport("QuadSurf");

  // Make sure the input data exists.
  FieldHandle ifieldH;
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
  qsf = new QuadSurfField<int>(mesh, 0);
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
