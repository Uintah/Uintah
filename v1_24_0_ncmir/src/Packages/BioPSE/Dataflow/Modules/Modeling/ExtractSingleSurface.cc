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
 *  ExtractSingleSurface.cc:  Extract a single QuadSurf component (boundary
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

class ExtractSingleSurface : public Module {
  GuiInt surfid_;
  GuiString data_;
public:
  ExtractSingleSurface(GuiContext *ctx);
  virtual ~ExtractSingleSurface();
  virtual void execute();
};

DECLARE_MAKER(ExtractSingleSurface)

ExtractSingleSurface::ExtractSingleSurface(GuiContext *ctx)
  : Module("ExtractSingleSurface", ctx, Filter, "Modeling", "BioPSE"), 
    surfid_(ctx->subVar("surfid")), data_(ctx->subVar("data"))
{
}

ExtractSingleSurface::~ExtractSingleSurface()
{
}

void
ExtractSingleSurface::execute()
{
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

  int comp = surfid_.get();
  if (comp<0 || comp>=ss->ncomps()) {
    error("Selected Component Index is out of Range");
    return;
  }

  //    cerr << "ST has "<<st->bcIdx.size()<<" vals...\n";
  //    for (int i=0; i<st->bcIdx.size(); i++)
  //	 cerr <<"  "<<i<<"  "<<st->bcVal[i]<<"  "<<st->points[st->bcIdx[i]]<<"\n";

  QuadSurfField<int> *qsf = ss->extractSingleComponent(comp, data_.get());

  //    cerr << "surface11 "<<ts->name<<" has "<<ts->points.size()<<" points, "<<ts->elements.size()<<" elements and "<<ts->bcVal.size()<<" known vals.\n";

  //    cerr << "TS has "<<ts->bcIdx.size()<<" vals...\n";
  //    for (i=0; i<ts->bcIdx.size(); i++)
  //	 cerr <<"  "<<i<<"  "<<ts->bcVal[i]<<"  "<<ts->points[ts->bcIdx[i]]<<"\n";

  FieldHandle sh2(qsf);
  ofp->send(sh2);
}    

} // End namespace BioPSE
