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
//    File   : EditTupleAxis.cc
//    Author : Martin Cole
//    Date   : Wed Mar 26 15:20:49 2003

#include <Teem/Dataflow/Ports/NrrdPort.h>
#include <Core/GuiInterface/GuiVar.h>
#include <iostream>
#include <sstream>

namespace SCITeem {

using namespace SCIRun;

class EditTupleAxis : public Module {
public:
  EditTupleAxis(GuiContext* ctx);
  virtual ~EditTupleAxis();
  virtual void execute();

private:
  GuiString      input_label_;
  GuiString      output_label_;
  NrrdDataHandle last_output_;
};


DECLARE_MAKER(EditTupleAxis)
  
EditTupleAxis::EditTupleAxis(GuiContext* ctx) : 
  Module("EditTupleAxis", ctx, Iterator,"NrrdData", "Teem"),
  input_label_(ctx->subVar("input-label")),
  output_label_(ctx->subVar("output-label")),
  last_output_(0)
{
}


EditTupleAxis::~EditTupleAxis()
{
}

void
EditTupleAxis::execute()
{
  update_state(NeedData);

  NrrdIPort *inrrd = (NrrdIPort *)get_iport("Nrrd");
  if (!inrrd) {
    error("Unable to initialize iport 'Nrrd'.");
    return;
  }
  NrrdDataHandle nrrd_handle;
  if (!(inrrd->get(nrrd_handle) && nrrd_handle.get_rep()))
  {
    error("Empty input NrrdData.");
    return;
  }
  
  NrrdOPort *onrrd_ = (NrrdOPort *)get_oport("Nrrd");
  if (!onrrd_) {
    error("Unable to initialize oport 'Nrrd'.");
    return;
  }

  update_state(JustStarted);
  input_label_.set(string(nrrd_handle->nrrd->axis[0].label));
  string output_label = output_label_.get();
  if (output_label == "") {
    last_output_ = nrrd_handle;
    onrrd_->send(nrrd_handle);
    return;
  }

  vector<string> dummy;
  if (nrrd_handle->verify_tuple_label(output_label, dummy)) {
    nrrd_handle.detach();
    free(nrrd_handle->nrrd->axis[0].label);
    nrrd_handle->nrrd->axis[0].label = strdup(output_label.c_str());
    last_output_ = nrrd_handle;
    onrrd_->send(nrrd_handle);
  } else {
    error("Invalid tuple label");
    return;
  }
}


} // End namespace SCITeem
