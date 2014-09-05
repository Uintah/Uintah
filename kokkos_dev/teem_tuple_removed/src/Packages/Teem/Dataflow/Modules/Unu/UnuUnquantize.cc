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
/*
 *  UnuUnquantize.cc: Recover floating point values from quantized data
 *
 *  Written by:
 *   Darby Van Uitert
 *   April 2004
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Teem/Dataflow/Ports/NrrdPort.h>


namespace SCITeem {

using namespace SCIRun;

class PSECORESHARE UnuUnquantize : public Module {
public:
  UnuUnquantize(GuiContext*);

  virtual ~UnuUnquantize();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);

  NrrdIPort*      inrrd_;
  NrrdOPort*      onrrd_;

  GuiInt       min_;
  GuiInt       max_;
  GuiInt       double_;
};


DECLARE_MAKER(UnuUnquantize)
UnuUnquantize::UnuUnquantize(GuiContext* ctx)
  : Module("UnuUnquantize", ctx, Source, "Unu", "Teem"),
    min_(ctx->subVar("min")),
    max_(ctx->subVar("max")),
    double_(ctx->subVar("double"))
{
}

UnuUnquantize::~UnuUnquantize(){
}

void
 UnuUnquantize::execute(){

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
  reset_vars();

  Nrrd *nin = nrrd_handle->nrrd;
  Nrrd *nout = nrrdNew();
  
  Nrrd* copy = nrrdNew();
  nrrdCopy(copy, nin);

  if (min_.get() != -1)
    copy->oldMin = min_.get();
  else
    copy->oldMin = nin->oldMin;

  if (max_.get() != -1)
    copy->oldMax = max_.get();
  else 
    copy->oldMax = nin->oldMax;
  
  if (double_.get()) {
    if (nrrdUnquantize(nout, copy, nrrdTypeDouble)) {
      char *err = biffGetDone(NRRD);
      error(string("Error Unquantizing nrrd: ") + err);
      free(err);
    }
  } else {
    if (nrrdUnquantize(nout, copy, nrrdTypeFloat)) {
      char *err = biffGetDone(NRRD);
      error(string("Error Unquantizing nrrd: ") + err);
      free(err);
    }    
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

void
 UnuUnquantize::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace Teem


