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
//    File   : Unu2op.cc
//    Author : Martin Cole
//    Date   : Mon Sep  8 09:46:49 2003

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Teem/Dataflow/Ports/NrrdPort.h>

namespace SCITeem {

using namespace SCIRun;

class Unu2op : public Module {
public:
  Unu2op(SCIRun::GuiContext *ctx);
  virtual ~Unu2op();
  virtual void execute();

private:
  NrrdIPort*      inrrd1_;
  NrrdOPort*      onrrd_;

  GuiString    operator_;
  GuiDouble    float_input_;
  GuiInt       nrrd_on_left_;

  unsigned int get_op(const string &op);
  
};

DECLARE_MAKER(Unu2op)

Unu2op::Unu2op(SCIRun::GuiContext *ctx) : 
  Module("Unu2op", ctx, Filter, "Unu", "Teem"), 
  operator_(ctx->subVar("operator")),
  float_input_(ctx->subVar("float_input")),
  nrrd_on_left_(ctx->subVar("nrrd_on_left"))
{
}

Unu2op::~Unu2op() {
}

void 
Unu2op::execute()
{
  NrrdDataHandle nrrd_handle1;

  update_state(NeedData);

  inrrd1_ = (NrrdIPort *)get_iport("InputNrrd");
  onrrd_ = (NrrdOPort *)get_oport("OutputNrrd");

  if (!inrrd1_) {
    error("Unable to initialize iport 'InputNrrd'.");
    return;
  }
  if (!onrrd_) {
    error("Unable to initialize oport 'OutputNrrd'.");
    return;
  }
  if (!inrrd1_->get(nrrd_handle1)) 
    return;


  if (!nrrd_handle1.get_rep()) {
    error("Empty InputNrrd.");
    return;
  }

  reset_vars();

  Nrrd *nin1 = nrrd_handle1->nrrd;
  Nrrd *nout = nrrdNew();

  NrrdIter *in1 = nrrdIterNew();
  NrrdIter *in2 = nrrdIterNew();

  // check 'Use Nrrd as First Input' to determine order of 
  // inputs
  if (nrrd_on_left_.get()) {
    nrrdIterSetOwnNrrd(in1, nin1);
    nrrdIterSetValue(in2, float_input_.get());
  } else {
    nrrdIterSetValue(in1, float_input_.get());
    nrrdIterSetOwnNrrd(in2, nin1);
  }

  if (nrrdArithIterBinaryOp(nout, get_op(operator_.get()), in1, in2)) {
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
Unu2op::get_op(const string &op) {
  if (op == "+") 
    return nrrdBinaryOpAdd;
  else if (op == "-")
    return nrrdBinaryOpSubtract;
  else if (op == "x")
    return nrrdBinaryOpMultiply;
  else if (op == "/")
    return nrrdBinaryOpDivide;
  else if (op == "^")
    return nrrdBinaryOpPow;
  else if (op == "%")
    return nrrdBinaryOpMod;
  else if (op == "fmod")
    return nrrdBinaryOpFmod;
  else if (op == "atan2")
    return nrrdBinaryOpAtan2;
  else if (op == "min")
    return nrrdBinaryOpMin;
  else if (op == "max")
    return nrrdBinaryOpMax;
  else if (op == "lt")
    return nrrdBinaryOpLT;
  else if (op == "lte")
    return nrrdBinaryOpLTE;
  else if (op == "gt")
    return nrrdBinaryOpGT;
  else if (op == "gte")
    return nrrdBinaryOpGTE;
  else if (op == "eq")
    return nrrdBinaryOpEqual;
  else if (op == "neq")
    return nrrdBinaryOpNotEqual;
  else if (op == "comp")
    return nrrdBinaryOpCompare;
  else {
    error("Unknown operation. Using eq");
    return nrrdBinaryOpEqual;
  }
}


} // End namespace SCITeem


