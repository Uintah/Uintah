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
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Packages/BioPSE/Core/Datatypes/SepSurf.h>
#include <Core/Basis/QuadBilinearLgn.h>
#include <Core/Basis/Constant.h>
#include <Core/Datatypes/QuadSurfMesh.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/GuiInterface/GuiVar.h>
#include <map>
#include <queue>
#include <iostream>

using std::queue;

namespace BioPSE {

using namespace SCIRun;

class SepSurfToQuadSurf : public Module {
  typedef QuadSurfMesh<QuadBilinearLgn<Point> > QSMesh;
  typedef ConstantBasis<int>                FDintBasis;
  typedef GenericField<QSMesh, FDintBasis, vector<int> > QSField;

  GuiString data_;
public:
  SepSurfToQuadSurf(GuiContext *ctx);
  virtual ~SepSurfToQuadSurf();
  virtual void execute();
};

DECLARE_MAKER(SepSurfToQuadSurf)

SepSurfToQuadSurf::SepSurfToQuadSurf(GuiContext *ctx)
  : Module("SepSurfToQuadSurf", ctx, Filter, "Modeling", "BioPSE"),
    data_(get_ctx()->subVar("data"))    
{
}

SepSurfToQuadSurf::~SepSurfToQuadSurf()
{
}

void
SepSurfToQuadSurf::execute()
{
  // Make sure the input data exists.
  FieldHandle ifieldH;
  if (!get_input_handle("SepSurf", ifieldH)) return;

  SepSurf *ss = dynamic_cast<SepSurf *>(ifieldH.get_rep());
  if (!ss) {
    error("Input field was not a SepSurf");
    return;
  }

  QSMesh::handle_type mesh = new QSMesh;
  Array1<int> data;
  QSField *qsf;
  int total_nnodes=0;
  Point p;
  QSMesh::Node::array_type nodes;
  
  for (int i=0; i<ss->ncomps(); i++) {
    qsf = ss->extractSingleComponent(i, data_.get());
    QSMesh::handle_type qsm = qsf->get_typed_mesh();
    QSMesh::Node::iterator nb, ne;
    QSMesh::Face::iterator fb, fe;
    QSMesh::Node::size_type nnodes;
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
  qsf = new QSField(mesh);
  QSMesh::Face::iterator fb, fe;
  int count=0;
  mesh->begin(fb);
  mesh->end(fe);
  while (fb != fe) {
    qsf->set_value(data[count], *fb);
    ++fb;
    ++count;
  }

  FieldHandle qsfH(qsf);
  send_output_handle("QuadSurf", qsfH);
}    

} // End namespace BioPSE
