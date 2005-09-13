/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/

//    File   : Unu2op.cc
//    Author : Martin Cole
//    Date   : Mon Sep  8 09:46:49 2003

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Dataflow/Ports/NrrdPort.h>

namespace SCITeem {

using namespace SCIRun;

class Unu2op : public Module {
public:
  Unu2op(SCIRun::GuiContext *ctx);
  virtual ~Unu2op();
  virtual void execute();

private:
  NrrdIPort*      inrrd1_;
  NrrdIPort*      inrrd2_;
  NrrdOPort*      onrrd_;

  GuiString    operator_;
  GuiDouble    float_input_;
  GuiString    type_;
  GuiInt       usetype_;
  bool         first_nrrd_;
  bool         second_nrrd_;

  unsigned int get_op(const string &op);
  unsigned int get_type(const string &type);
};

DECLARE_MAKER(Unu2op)

Unu2op::Unu2op(SCIRun::GuiContext *ctx) : 
  Module("Unu2op", ctx, Filter, "UnuAtoM", "Teem"), 
  operator_(ctx->subVar("operator")),
  float_input_(ctx->subVar("float_input")),
  type_(ctx->subVar("type")),
  usetype_(ctx->subVar("usetype")),
  first_nrrd_(true), second_nrrd_(true)
{
}

Unu2op::~Unu2op() {
}

void 
Unu2op::execute()
{
  first_nrrd_ = true;
  second_nrrd_ = true;

  NrrdDataHandle nrrd_handle1;
  NrrdDataHandle nrrd_handle2;

  update_state(NeedData);

  inrrd1_ = (NrrdIPort *)get_iport("InputNrrd1");
  inrrd2_ = (NrrdIPort *)get_iport("InputNrrd2");
  onrrd_ = (NrrdOPort *)get_oport("OutputNrrd");

  if (!inrrd1_->get(nrrd_handle1)) 
    first_nrrd_ = false;
  if (!inrrd2_->get(nrrd_handle2)) 
    second_nrrd_ = false;
  
  if (first_nrrd_ && !nrrd_handle1.get_rep()) {
    error("Empty InputNrrd1.");
    return;
  }

  if (second_nrrd_ && !nrrd_handle2.get_rep()) {
    error("Empty InputNrrd2.");
    return;
  }

  Nrrd *nin1 = 0;
  Nrrd *nin2 = 0;
  Nrrd *nout = nrrdNew();
  Nrrd *ntmp1 = NULL;
  Nrrd *ntmp2 = NULL;

  // can either have two nrrds, first nrrd and float, or second
  // nrrd and float
  if (!first_nrrd_ && !second_nrrd_) {
    error("Must have at least one nrrd connected.");
    return;
  }

  reset_vars();

  // convert nrrds if indicated
  if (!usetype_.get()) {
    if (first_nrrd_) {
      ntmp1 = nrrdNew();
      if (nrrdConvert(ntmp1, nrrd_handle1->nrrd, get_type(type_.get()))) {
	char *err = biffGetDone(NRRD);
	error(string("Error converting nrrd: ") + err);
	free(err);
	return;
      }
    }
    if (second_nrrd_) {
      ntmp2 = nrrdNew();
      if (nrrdConvert(ntmp2, nrrd_handle2->nrrd, get_type(type_.get()))) {
	char *err = biffGetDone(NRRD);
	error(string("Error converting nrrd: ") + err);
	free(err);
	return;
      }
    }
  }
  
  if (first_nrrd_) {
    if (!usetype_.get())
      nin1 = ntmp1;
    else
      nin1 = nrrd_handle1->nrrd;
  }
  if (second_nrrd_) {
    if (!usetype_.get())
      nin2 = ntmp2;
    else
      nin2 = nrrd_handle2->nrrd;
  }

  NrrdIter *in1 = nrrdIterNew();
  NrrdIter *in2 = nrrdIterNew();


  if (first_nrrd_ && second_nrrd_) {
    nrrdIterSetOwnNrrd(in1, nin1);
    nrrdIterSetOwnNrrd(in2, nin2);
  }
  else if (first_nrrd_) {
    nrrdIterSetOwnNrrd(in1, nin1);
    nrrdIterSetValue(in2, float_input_.get());
  } else {
    nrrdIterSetValue(in1, float_input_.get());
    nrrdIterSetOwnNrrd(in2, nin2);
  }

  if (nrrdArithIterBinaryOp(nout, get_op(operator_.get()), in1, in2)) {
    char *err = biffGetDone(NRRD);
    error(string("Error performing 2op to nrrd: ") + err);
    free(err);
  }

  NrrdData *nrrd = scinew NrrdData;
  nrrd->nrrd = nout;

  nrrdKeyValueCopy(nout, nin2);
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
  else if (op == "exists")
    return nrrdBinaryOpExists;
  else {
    error("Unknown operation. Using eq");
    return nrrdBinaryOpEqual;
  }
}

unsigned int
Unu2op::get_type(const string &type) {
  if (type == "nrrdTypeChar")
    return nrrdTypeChar;
  else if (type == "nrrdTypeUChar") 
    return nrrdTypeUChar;
  else if (type == "nrrdTypeShort")
    return nrrdTypeShort;
  else if (type == "nrrdTypeUShort")
    return nrrdTypeUShort;
  else if (type == "nrrdTypeInt")
    return nrrdTypeInt;
  else if (type == "nrrdTypeUInt")
    return nrrdTypeUInt;
  else if (type == "nrrdTypeLLong")
    return nrrdTypeLLong;
  else if (type == "nrrdTypeULLong")
    return nrrdTypeULLong;
  else if (type == "nrrdTypeFloat")
    return nrrdTypeFloat;
  else if (type == "nrrdTypeDouble")
    return nrrdTypeDouble;
  else {
    error("Unknown nrrd type. Defaulting to nrrdTypeFloat");
    return nrrdTypeFloat;
  }
}


} // End namespace SCITeem


