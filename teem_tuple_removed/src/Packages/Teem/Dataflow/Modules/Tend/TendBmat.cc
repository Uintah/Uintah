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
//    File   : TendBmat.cc
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

class TendBmat : public Module {
public:
  TendBmat(SCIRun::GuiContext *ctx);
  virtual ~TendBmat();
  virtual void execute();

private:
  // Create a memory for a new nrrd, that is arranged 3 x n;
  bool extract_gradients(vector<double> &);

  NrrdIPort*      inrrd_;
  NrrdOPort*      onrrd_;

  GuiString    gradient_list_;
};

DECLARE_MAKER(TendBmat)

TendBmat::TendBmat(SCIRun::GuiContext *ctx) : 
  Module("TendBmat", ctx, Filter, "Tend", "Teem"), 
  gradient_list_(ctx->subVar("gradient_list"))
{
}

TendBmat::~TendBmat() {
}


// Create a memory for a new nrrd, that is arranged 3 x n;
bool
TendBmat::extract_gradients(vector<double> &d)
{
  istringstream str(gradient_list_.get().c_str());
  while (true)
  {
    double tmp;
    str >> tmp;
    if (!str.eof() && !str.fail()) {
      d.push_back(tmp);
    }
    else {
      break;
    }
  }
  if (d.size() % 3 != 0) {
    error("Error: Number of input values must be divisible by 3");
    return false;
  }
  return true;
}


void 
TendBmat::execute()
{
  NrrdDataHandle nrrd_handle;
  update_state(NeedData);
  inrrd_ = (NrrdIPort *)get_iport("nin");
  onrrd_ = (NrrdOPort *)get_oport("nout");

  if (!inrrd_) {
    error("Unable to initialize iport 'nin'.");
    return;
  }
  if (!onrrd_) {
    error("Unable to initialize oport 'nout'.");
    return;
  }

  bool we_own_the_data;
  vector<double> *mat=0;
  Nrrd *nin;

  if (inrrd_->get(nrrd_handle) && nrrd_handle.get_rep()) {
    we_own_the_data = false;
    nin = nrrd_handle->nrrd;
  } else {
    we_own_the_data = true;
    mat = new vector<double>;
    if (! extract_gradients(*mat)) {
      error("Please adjust your input in the gui to represent a 3 x N set.");
      return;
    }
    nin = nrrdNew();
    nrrdWrap(nin, &(*mat)[0], nrrdTypeDouble, 2, 3, (*mat).size() / 3);
  }

  Nrrd *nout = nrrdNew();
  if (tenBMatrixCalc(nout, nin)) {
    char *err = biffGetDone(TEN);
    error(string("Error making aniso volume: ") + err);
    free(err);
    return;
  }
  
  Nrrd *ntup = nrrdNew();
  nrrdAxesInsert(ntup, nout, 0);
  ntup->axis[0].label = strdup("BMat:Scalar");
  ntup->axis[1].label = strdup("tensor components");
  ntup->axis[2].label = strdup("n");
  nrrdNuke(nout);

  if (we_own_the_data) {
    nrrdNix(nin);
    delete mat;
  }

  NrrdData *nrrd = scinew NrrdData;
  nrrd->nrrd = ntup;
  onrrd_->send(NrrdDataHandle(nrrd));
}

} // End namespace SCITeem
