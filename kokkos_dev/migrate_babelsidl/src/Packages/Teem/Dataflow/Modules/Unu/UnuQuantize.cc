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
 *  UnuQuantize
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   January 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Dataflow/Network/Ports/NrrdPort.h>

#include <sstream>
#include <iostream>
using std::endl;
#include <stdio.h>

namespace SCITeem {
using namespace SCIRun;

class UnuQuantize : public Module {
  GuiDouble minf_;
  GuiDouble maxf_;
  GuiInt useinputmin_;
  GuiInt useinputmax_;
  GuiDouble realmin_;
  GuiDouble realmax_;
  GuiInt nbits_;
  double last_minf_;
  double last_maxf_;
  int last_nbits_;
  int last_generation_;
  NrrdDataHandle last_nrrdH_;
public:
  UnuQuantize(GuiContext *ctx);
  virtual ~UnuQuantize();
  virtual void execute();
};

} // End namespace SCITeem

using namespace SCITeem;

DECLARE_MAKER(UnuQuantize)

UnuQuantize::UnuQuantize(GuiContext *ctx)
  : Module("UnuQuantize", ctx, Filter, "UnuNtoZ", "Teem"),
    minf_(get_ctx()->subVar("minf")),
    maxf_(get_ctx()->subVar("maxf")),
    useinputmin_(get_ctx()->subVar("useinputmin")),
    useinputmax_(get_ctx()->subVar("useinputmax")),
    realmin_(get_ctx()->subVar("realmin")),
    realmax_(get_ctx()->subVar("realmax")),
    nbits_(get_ctx()->subVar("nbits")), last_minf_(0),
    last_maxf_(0), last_nbits_(0), last_generation_(-1), last_nrrdH_(0)
{
}


UnuQuantize::~UnuQuantize()
{
}


void 
UnuQuantize::execute()
{
  update_state(NeedData);

  NrrdDataHandle nrrdH;
  if (!get_input_handle("Nrrd", nrrdH)) return;
  
  reset_vars();

  if (last_generation_ != nrrdH->generation) {
    // set default values for min,max
    NrrdRange *range = nrrdRangeNewSet(nrrdH->nrrd_, nrrdBlind8BitRangeState);
    realmin_.set(range->min);
    realmax_.set(range->max);
    nrrdRangeNix(range);
    minf_.reset();
    maxf_.reset();
    useinputmin_.reset();
    useinputmax_.reset();
  }

  double minf = minf_.get();
  double maxf = maxf_.get();

  // use input min/max if specified
  if (useinputmin_.get()) {
    minf = realmin_.get();
  }

  if (useinputmax_.get()) {
    maxf = realmax_.get();
  }
  
  const int nbits = nbits_.get();
  if (last_generation_ == nrrdH->generation &&
      last_minf_ == minf &&
      last_maxf_ == maxf &&
      last_nbits_ == nbits &&
      last_nrrdH_.get_rep())
  {
    send_output_handle("Nrrd", last_nrrdH_, true);
    return;
  }

  // must detach because we are about to modify the input nrrd.
  last_generation_ = nrrdH->generation;
  nrrdH.detach(); 

  Nrrd *nin = nrrdH->nrrd_;

  remark("Quantizing -- min=" + to_string(minf) +
         " max=" + to_string(maxf) + " nbits=" + to_string(nbits));
  NrrdRange *range = nrrdRangeNew(minf, maxf);
  NrrdData *nrrd = scinew NrrdData;
  if (nrrdQuantize(nrrd->nrrd_, nin, range, nbits))
  {
    char *err = biffGetDone(NRRD);
    error(string("Trouble quantizing: ") + err);
    free(err);
    return;
  }

  nrrdKeyValueCopy(nrrd->nrrd_, nin);
  last_minf_ = minf;
  last_maxf_ = maxf;
  last_nbits_ = nbits;
  last_nrrdH_ = nrrd;

  send_output_handle("Nrrd", last_nrrdH_, true);
}

