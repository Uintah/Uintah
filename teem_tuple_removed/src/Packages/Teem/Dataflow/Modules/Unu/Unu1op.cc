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
//    File   : Unu1op.cc
//    Author : Martin Cole
//    Date   : Mon Sep  8 09:46:49 2003

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Teem/Dataflow/Ports/NrrdPort.h>

namespace SCITeem {

using namespace SCIRun;

class Unu1op : public Module {
public:
  Unu1op(SCIRun::GuiContext *ctx);
  virtual ~Unu1op();
  virtual void execute();

private:
  NrrdIPort*      inrrd_;
  NrrdOPort*      onrrd_;

  GuiString    operator_;

  unsigned int get_op(const string &op);
  
};

DECLARE_MAKER(Unu1op)

Unu1op::Unu1op(SCIRun::GuiContext *ctx) : 
  Module("Unu1op", ctx, Filter, "Unu", "Teem"), 
  operator_(ctx->subVar("operator"))
{
}

Unu1op::~Unu1op() {
}

void 
Unu1op::execute()
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
    error("Empty InputNrrd.");
    return;
  }

  reset_vars();

  Nrrd *nin = nrrd_handle->nrrd;
  Nrrd *nout = nrrdNew();

  if (nrrdArithUnaryOp(nout, get_op(operator_.get()), nin)) {
    char *err = biffGetDone(NRRD);
    error(string("Error performing 1op to nrrd: ") + err);
    free(err);
  }

  NrrdData *nrrd = scinew NrrdData;
  nrrd->nrrd = nout;

  NrrdDataHandle out(nrrd);

  onrrd_->send(out);
}

unsigned int
Unu1op::get_op(const string &op) {
  if (op == "-")
    return nrrdUnaryOpNegative;
  else if (op == "r") 
    return nrrdUnaryOpReciprocal;
  else if (op == "sin")
    return nrrdUnaryOpSin;
  else if (op == "cos")
    return nrrdUnaryOpCos;
  else if (op == "tan")
    return nrrdUnaryOpTan;
  else if (op == "asin")
    return nrrdUnaryOpAsin;
  else if (op == "acos")
    return nrrdUnaryOpAcos;
  else if (op == "atan")
    return nrrdUnaryOpAtan;
  else if (op == "exp")
    return nrrdUnaryOpExp;
  else if (op == "log")
    return nrrdUnaryOpLog;
  else if (op == "log10")
    return nrrdUnaryOpLog10;
  else if (op == "log1p")
    return nrrdUnaryOpLog1p;
  else if (op == "sqrt")
    return nrrdUnaryOpSqrt;
  else if (op == "cbrt")
    return nrrdUnaryOpCbrt;
  else if (op == "ceil")
    return nrrdUnaryOpCeil;
  else if (op == "floor")
    return nrrdUnaryOpFloor;
  else if (op == "erf")
    return nrrdUnaryOpErf;
  else if (op == "rup")
    return nrrdUnaryOpRoundUp;
  else if (op == "rdn")
    return nrrdUnaryOpRoundDown;
  else if (op == "abs")
    return nrrdUnaryOpAbs;
  else if (op == "sgn")
    return nrrdUnaryOpSgn;
  else if (op == "exists")
    return nrrdUnaryOpExists;
  else {
    warning("Unknown unary op. Performing random operation.");
    return nrrdUnaryOpRand;
  }
}


} // End namespace SCITeem


