/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
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
 *  GetSingleSurfaceFromSepSurfs.cc:  Extract a single QuadSurf component (boundary
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
#include <Dataflow/GuiInterface/GuiVar.h>
#include <map>
#include <queue>
#include <iostream>

using std::queue;

namespace BioPSE {

using namespace SCIRun;

class GetSingleSurfaceFromSepSurfs : public Module {

  GuiInt surfid_;
  GuiString data_;
public:
  GetSingleSurfaceFromSepSurfs(GuiContext *ctx);
  virtual ~GetSingleSurfaceFromSepSurfs();
  virtual void execute();
};

DECLARE_MAKER(GetSingleSurfaceFromSepSurfs)

GetSingleSurfaceFromSepSurfs::GetSingleSurfaceFromSepSurfs(GuiContext *ctx)
  : Module("GetSingleSurfaceFromSepSurfs", ctx, Filter, "Modeling", "BioPSE"), 
    surfid_(get_ctx()->subVar("surfid")), data_(get_ctx()->subVar("data"))
{
}

GetSingleSurfaceFromSepSurfs::~GetSingleSurfaceFromSepSurfs()
{
}

void
GetSingleSurfaceFromSepSurfs::execute()
{
  // Make sure the input data exists.
  FieldHandle ifieldH;
  if (!get_input_handle("SepSurf", ifieldH)) return;

  SepSurf *ss = dynamic_cast<SepSurf *>(ifieldH.get_rep());
  if (!ss) {
    error("Input field was not a SepSurf");
    return;
  }

  int comp = surfid_.get();
  if (comp < 0 || comp >= ss->ncomps()) {
    error("Selected Component Index is out of Range");
    return;
  }

  SSQSField *qsf = ss->extractSingleComponent(comp, data_.get());

  FieldHandle sh2(qsf);

  send_output_handle("QuadSurf", sh2);
}    

} // End namespace BioPSE
