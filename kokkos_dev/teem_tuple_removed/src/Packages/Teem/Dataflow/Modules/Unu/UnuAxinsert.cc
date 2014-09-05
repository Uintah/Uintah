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
//    File   : UnuAxinsert.cc Add a "stub" (length 1) axis to a nrrd.
//    Author : Martin Cole
//    Date   : Mon Sep  8 09:46:49 2003

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Teem/Dataflow/Ports/NrrdPort.h>
#include <Core/Containers/StringUtil.h>

namespace SCITeem {

using namespace SCIRun;

class UnuAxinsert : public Module {
public:
  UnuAxinsert(SCIRun::GuiContext *ctx);
  virtual ~UnuAxinsert();
  virtual void execute();

private:
  NrrdIPort*      inrrd_;
  NrrdOPort*      onrrd_;

  GuiInt          axis_;
  GuiString       label_;
};

DECLARE_MAKER(UnuAxinsert)

UnuAxinsert::UnuAxinsert(SCIRun::GuiContext *ctx) : 
  Module("UnuAxinsert", ctx, Filter, "Unu", "Teem"), 
  axis_(ctx->subVar("axis")), label_(ctx->subVar("label"))
{
}

UnuAxinsert::~UnuAxinsert() {
}

void 
UnuAxinsert::execute()
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

  
  if (nrrdAxesInsert(nout, nin, axis_.get())) {
   char *err = biffGetDone(NRRD);
    error(string("Error Axinserting nrrd: ") + err);
    free(err);
  }

  if (strlen(label_.get().c_str())) {
    int axis = axis_.get();
    nout->axis[axis].label = airStrdup(label_.get().c_str());
  }

  NrrdData *nrrd = scinew NrrdData;
  nrrd->nrrd = nout;

  NrrdDataHandle out(nrrd);

  // Copy the properties.
  *((PropertyManager *) out.get_rep()) =
    *((PropertyManager *) nrrd_handle.get_rep());

  // set kind
  // Copy the axis kinds
  int offset = 0;
  for (int i=0; i<nin->dim; i++) {
    if (i == axis_.get()) {
      offset = 1;
      nout->axis[i].kind = nrrdKindStub;
    }
    nout->axis[i+offset].kind = nin->axis[i].kind;
  }
  if (axis_.get() == nin->dim) 
    nout->axis[axis_.get()].kind = nrrdKindStub;

  onrrd_->send(out);

}

} // End namespace SCITeem


