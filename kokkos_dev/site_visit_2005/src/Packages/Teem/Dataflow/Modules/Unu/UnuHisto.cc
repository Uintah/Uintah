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

//    File   : UnuHisto.cc
//    Author : Martin Cole
//    Date   : Mon Sep  8 09:46:49 2003

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Dataflow/Ports/NrrdPort.h>

namespace SCITeem {

using namespace SCIRun;

class UnuHisto : public Module {
public:
  UnuHisto(SCIRun::GuiContext *ctx);
  virtual ~UnuHisto();
  virtual void execute();

private:
  NrrdIPort*      inrrd_;
  NrrdIPort*      wnrrd_;
  NrrdOPort*      onrrd_;

  GuiInt       bins_;
  GuiDouble    min_;
  GuiInt       useinputmin_;
  GuiDouble    max_;
  GuiInt       useinputmax_;
  GuiString    type_;
  
  unsigned int get_type(string type);
};

DECLARE_MAKER(UnuHisto)

UnuHisto::UnuHisto(SCIRun::GuiContext *ctx) : 
  Module("UnuHisto", ctx, Filter, "UnuAtoM", "Teem"), 
  bins_(ctx->subVar("bins")),
  min_(ctx->subVar("min")),
  useinputmin_(ctx->subVar("useinputmin")),
  max_(ctx->subVar("max")),
  useinputmax_(ctx->subVar("useinputmax")),
  type_(ctx->subVar("type"))
{
}

UnuHisto::~UnuHisto() {
}

void 
UnuHisto::execute()
{
  NrrdDataHandle nrrd_handle;
  NrrdDataHandle weight_handle;

  update_state(NeedData);

  inrrd_ = (NrrdIPort *)get_iport("InputNrrd");
  wnrrd_ = (NrrdIPort *)get_iport("WeightNrrd");
  onrrd_ = (NrrdOPort *)get_oport("OutputNrrd");

  if (!inrrd_->get(nrrd_handle)) 
    return;

  if (!nrrd_handle.get_rep()) {
    error("Empty InputNrrd.");
    return;
  }

  reset_vars();

  Nrrd *nin = nrrd_handle->nrrd;
  Nrrd *weight = 0;
  if (wnrrd_->get(weight_handle)) {
    weight = weight_handle->nrrd;    
  }
  Nrrd *nout = nrrdNew();

  NrrdRange *range = NULL;

  double min = AIR_NAN, max = AIR_NAN;
  if (!useinputmin_.get())
    min = min_.get();
  if (!useinputmax_.get())
    max = max_.get();

  nrrdRangeNew(min, max);
  nrrdRangeSafeSet(range, nin, nrrdBlind8BitRangeState);

  unsigned int type = get_type(type_.get());

  if (nrrdHisto(nout, nin, range, weight, bins_.get(), type)) {
    char *err = biffGetDone(NRRD);
    error(string("Error creating Histogram nrrd: ") + err);
    free(err);
  }

  NrrdData *nrrd = scinew NrrdData;
  nrrd->nrrd = nout;

  NrrdDataHandle out(nrrd);

  onrrd_->send(out);
}

unsigned int
UnuHisto::get_type(string type) {
  if (type == "nrrdTypeChar") 
    return nrrdTypeChar;
  else if (type == "nrrdTypeUChar")  
    return nrrdTypeUChar;
  else if (type == "nrrdTypeShort")  
    return nrrdTypeShort;
  else if (type == "nrrdTypeUShort") 
    return nrrdTypeUShort;
  else if (type == "nrrdTypeInt")  
    return nrrdTypeInt;
  else if (type == "nrrdTypeUInt")   
    return nrrdTypeUInt;
  else if (type == "nrrdTypeFloat") 
    return nrrdTypeFloat;
  else if (type == "nrrdTypeDouble")  
    return nrrdTypeDouble;
  else    
    return nrrdTypeUInt;
}

} // End namespace SCITeem
