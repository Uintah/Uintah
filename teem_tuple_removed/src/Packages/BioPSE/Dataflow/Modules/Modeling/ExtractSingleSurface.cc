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

void ExtractSingleSurface::execute() {
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
