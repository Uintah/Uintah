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
//    File   : TendSatin.cc
//    Author : Martin Cole
//    Date   : Mon Sep  8 09:46:49 2003

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Teem/Dataflow/Ports/NrrdPort.h>

#include <sstream>
#include <iostream>
using std::endl;
#include <stdio.h>

namespace SCITeem {

using namespace SCIRun;

class TendSatin : public Module {
public:
  TendSatin(SCIRun::GuiContext *ctx);
  virtual ~TendSatin();
  virtual void execute();

private:
  NrrdOPort*      onrrd_;

  GuiDouble    anisotropy_;
  GuiDouble    min_;
  GuiDouble    max_;
  GuiDouble    boundary_;
  GuiDouble    thickness_;
  GuiInt       size_;
  GuiInt       torus_;
};

DECLARE_MAKER(TendSatin)

TendSatin::TendSatin(SCIRun::GuiContext *ctx) : 
  Module("TendSatin", ctx, Filter, "Tend", "Teem"), 
  anisotropy_(ctx->subVar("anisotropy")),
  min_(ctx->subVar("min")),
  max_(ctx->subVar("max")),
  boundary_(ctx->subVar("boundary")),
  thickness_(ctx->subVar("thickness")),
  size_(ctx->subVar("size")),
  torus_(ctx->subVar("torus"))
{
}

TendSatin::~TendSatin() {
}

void 
TendSatin::execute()
{
  NrrdDataHandle nrrd_handle;
  update_state(NeedData);
  onrrd_ = (NrrdOPort *)get_oport("nout");

  if (!onrrd_) {
    error("Unable to initialize oport 'Nrrd'.");
    return;
  }

  error("This module is a stub.  Implement me.");

  //onrrd_->send(NrrdDataHandle(nrrd_joined));
}

} // End namespace SCITeem
