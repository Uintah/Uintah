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
//    File   : UnuFlip.cc
//    Author : Martin Cole
//    Date   : Mon Sep  8 09:46:49 2003

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Teem/Dataflow/Ports/NrrdPort.h>
#include <Core/Containers/StringUtil.h>

namespace SCITeem {

using namespace SCIRun;

class UnuFlip : public Module {
public:
  UnuFlip(SCIRun::GuiContext *ctx);
  virtual ~UnuFlip();
  virtual void execute();

private:
  NrrdIPort*      inrrd_;
  NrrdOPort*      onrrd_;

  GuiInt          axis_;
};

DECLARE_MAKER(UnuFlip)

UnuFlip::UnuFlip(SCIRun::GuiContext *ctx) : 
  Module("UnuFlip", ctx, Filter, "Unu", "Teem"), 
  axis_(ctx->subVar("axis"))
{
}

UnuFlip::~UnuFlip() {
}

void 
UnuFlip::execute()
{
  NrrdDataHandle nrrd_handle;
  update_state(NeedData);
  inrrd_ = (NrrdIPort *)get_iport("InputNrrd");
  onrrd_ = (NrrdOPort *)get_oport("OutputNrrd");

  if (!inrrd_) {
    error("Unable to initialize iport 'InputNrrd'.");
    return;
  }
  if (!onrrd_) {
    error("Unable to initialize oport 'OutputNrrd'.");
    return;
  }
  if (!inrrd_->get(nrrd_handle))
    return;

  if (!nrrd_handle.get_rep()) {
    error("Empty input Nrrd.");
    return;
  }

  Nrrd *nin = nrrd_handle->nrrd;
  Nrrd *nout = nrrdNew();
  
  if (axis_.get() >= nin->dim) {
    error("Axis " + to_string(axis_.get()) + " out of bounds [0," + to_string(nin->dim-1));
    return;
  }

  if (nrrdFlip(nout, nin, axis_.get())) {
   char *err = biffGetDone(NRRD);
    error(string("Error Flipping nrrd: ") + err);
    free(err);
  }

  NrrdData *nrrd = scinew NrrdData;
  nrrd->nrrd = nout;

  NrrdDataHandle out(nrrd);
  // Copy the properties.
  *((PropertyManager *) out.get_rep()) =
    *((PropertyManager *) nrrd_handle.get_rep());

  // Copy the axis kinds
  for (int i=0; i<nin->dim; i++) {
    nout->axis[i].kind = nin->axis[i].kind;
  }

  onrrd_->send(out);

}

} // End namespace SCITeem
