//  The contents of this file are subject to the University of Utah Public
//  License (the "License"); you may not use this file except in compliance
//  with the License.
//  
//  Software distributed under the License is distributed on an "AS IS"
//  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
//  License for the specific language governing rights and limitations under
//  the License.
//  
//  The Original Source Code is SCIRun, released March 12, 2001.
//  
//  The Original Source Code was developed by the University of Utah.
//  Portions created by UNIVERSITY are Copyright (C) 2001, 1994
//  University of Utah. All Rights Reserved.
//  
//    File   : UnuHisto.cc
//    Author : Martin Cole
//    Date   : Mon Sep  8 09:46:49 2003

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Teem/Dataflow/Ports/NrrdPort.h>

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
  GuiDouble    max_;
  GuiString    type_;
  
  unsigned int get_type(string type);
};

DECLARE_MAKER(UnuHisto)

UnuHisto::UnuHisto(SCIRun::GuiContext *ctx) : 
  Module("UnuHisto", ctx, Filter, "Unu", "Teem"), 
  bins_(ctx->subVar("bins")),
  min_(ctx->subVar("min")),
  max_(ctx->subVar("max")),
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

  if (!inrrd_) {
    error("Unable to initialize iport 'InputNrrd'.");
    return;
  }
  if (!wnrrd_) {
    error("Unable to initialize iport 'WeightNrrd'.");
    return;
  }
  if (!onrrd_) {
    error("Unable to initialize oport 'OutputNrrd'.");
    return;
  }
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

  NrrdRange *range = nrrdRangeNew(min_.get(), max_.get());
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
