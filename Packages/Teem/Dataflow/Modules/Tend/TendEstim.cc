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
//    File   : TendEstim.cc
//    Author : Martin Cole
//    Date   : Mon Sep  8 09:46:49 2003

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Teem/Dataflow/Ports/NrrdPort.h>
#include <teem/ten.h>

#include <sstream>
#include <iostream>
using std::endl;
#include <stdio.h>

namespace SCITeem {

using namespace SCIRun;

class TendEstim : public Module {
public:
  TendEstim(SCIRun::GuiContext *ctx);
  virtual ~TendEstim();
  virtual void execute();

private:
  NrrdIPort*      inbmat_;
  NrrdIPort*      indwi_;
  NrrdOPort*      otens_;
  //NrrdOPort*      oerr_;

  GuiDouble    threshold_;
  GuiDouble    soft_;
  GuiDouble    scale_;
};

DECLARE_MAKER(TendEstim)

TendEstim::TendEstim(SCIRun::GuiContext *ctx) : 
  Module("TendEstim", ctx, Filter, "Tend", "Teem"), 
  threshold_(ctx->subVar("threshold")),
  soft_(ctx->subVar("soft")),
  scale_(ctx->subVar("scale"))
{
}

TendEstim::~TendEstim() {
}

void 
TendEstim::execute()
{
  NrrdDataHandle bmat_handle;
  NrrdDataHandle dwi_handle;
  update_state(NeedData);
  inbmat_ = (NrrdIPort *)get_iport("Bmat");
  indwi_ = (NrrdIPort *)get_iport("DWI");
  otens_ = (NrrdOPort *)get_oport("Tensors");
  //  oerr_ = (NrrdOPort *)get_oport("Error");

  if (!inbmat_) {
    error("Unable to initialize iport 'Bmat'.");
    return;
  }
  if (!indwi_) {
    error("Unable to initialize iport 'DWI'.");
    return;
  }
  if (!otens_) {
    error("Unable to initialize oport 'Tensors'.");
    return;
  }
//   if (!oerr_) {
//     error("Unable to initialize oport 'Error'.");
//     return;
//   }

  Nrrd *sliced_bmat = 0;
  if (inbmat_->get(bmat_handle)){
    if (!bmat_handle.get_rep()) {
      error("Empty input Bmat Nrrd.");
      return;
    }
    sliced_bmat = nrrdNew();
    // slice the tuple axis off to send to tendEstim
    if (nrrdSlice(sliced_bmat, bmat_handle->nrrd, 0, 0)) {
      char *err = biffGetDone(NRRD);
      error(string("Error Slicing away bmat tuple axis: ") + err);
      free(err);
    }
  } else {
    error("Empty input Bmat Port.");
    return;
  }
  if (!indwi_->get(dwi_handle))
    return;

  if (!bmat_handle.get_rep()) {
    error("Empty input Bmat Nrrd.");
    return;
  }
  if (!dwi_handle.get_rep()) {
    error("Empty input DWI Nrrd.");
    return;
  }

 
  Nrrd *nout = nrrdNew();
  if (tenEstimate4D(nout, NULL, dwi_handle->nrrd, bmat_handle->nrrd, 
		    threshold_.get(), soft_.get(), scale_.get()))
  {
    char *err = biffGetDone(TEN);
    error(string("Error in epireg: ") + err);
    free(err);
    return;
  }
  
  nrrdNuke(sliced_bmat);
  NrrdData *output = scinew NrrdData;
  output->nrrd = nout;
  output->copy_sci_data(*dwi_handle.get_rep());
  output->nrrd->axis[0].label = strdup(dwi_handle->nrrd->axis[0].label);
  otens_->send(NrrdDataHandle(output));
}

} // End namespace SCITeem
