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
//    File   : TendPoint.cc
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

class TendPoint : public Module {
public:
  TendPoint(SCIRun::GuiContext *ctx);
  virtual ~TendPoint();
  virtual void execute();

private:
  NrrdIPort*      indata_;
  NrrdIPort*      inpoint_;

  GuiString    point_;
};

DECLARE_MAKER(TendPoint)

TendPoint::TendPoint(SCIRun::GuiContext *ctx) : 
  Module("TendPoint", ctx, Filter, "Tend", "Teem"), 
  point_(ctx->subVar("point"))
{
}

TendPoint::~TendPoint() {
}

void 
TendPoint::execute()
{
  NrrdDataHandle data_handle;
  NrrdDataHandle point_handle;
  update_state(NeedData);
  indata_ = (NrrdIPort *)get_iport("Data");
  inpoint_ = (NrrdIPort *)get_iport("Point");


  if (!indata_) {
    error("Unable to initialize iport 'Data'.");
    return;
  }
  if (!inpoint_) {
    error("Unable to initialize iport 'Point'.");
    return;
  }

  if (!indata_->get(data_handle))
    return;
  if (!inpoint_->get(point_handle))
    return;

  if (!data_handle.get_rep()) {
    error("Empty input Data Nrrd.");
    return;
  }
  if (!point_handle.get_rep()) {
    error("Empty input Point Nrrd.");
    return;
  }

  //FIX_ME : do the following:
  // the point input to be supported both 
  // as an input Nrrd, as well as via the user typing a string in a 
  // text-entry field on the UI.  If both are supplied then the input Nrrd 
  // should be used (and the UI value should be ignored).

  error("This module is a stub.  Implement me.");

  //onrrd_->send(NrrdDataHandle(nrrd_joined));
}

} // End namespace SCITeem
