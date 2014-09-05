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

//    File   : Unu3op.cc
//    Author : Martin Cole
//    Date   : Mon Sep  8 09:46:49 2003

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Dataflow/Ports/NrrdPort.h>

namespace SCITeem {

using namespace SCIRun;

class Unu3op : public Module {
public:
  Unu3op(SCIRun::GuiContext *ctx);
  virtual ~Unu3op();
  virtual void execute();

private:
  NrrdIPort*      inrrd1_;
  NrrdIPort*      inrrd2_;
  NrrdIPort*      inrrd3_;
  NrrdOPort*      onrrd_;

  GuiString    operator_;
  GuiDouble    float1_;
  GuiDouble    float2_;
  GuiDouble    float3_;
  GuiString    type_;
  GuiInt       usetype_;
  bool         first_nrrd_;
  bool         second_nrrd_;
  bool         third_nrrd_;

  unsigned int get_op(const string &op);
  unsigned int get_type(const string &op);
  
};

DECLARE_MAKER(Unu3op)

Unu3op::Unu3op(SCIRun::GuiContext *ctx) : 
  Module("Unu3op", ctx, Filter, "UnuAtoM", "Teem"), 
  operator_(ctx->subVar("operator")),
  float1_(ctx->subVar("float1")),
  float2_(ctx->subVar("float2")),
  float3_(ctx->subVar("float3")),
  type_(ctx->subVar("type")),
  usetype_(ctx->subVar("usetype")),
  first_nrrd_(true), second_nrrd_(true), third_nrrd_(true)
{
}

Unu3op::~Unu3op() {
}

void 
Unu3op::execute()
{
  first_nrrd_ = true;
  second_nrrd_ = true;
  third_nrrd_ = true;

  NrrdDataHandle nrrd_handle1;
  NrrdDataHandle nrrd_handle2;
  NrrdDataHandle nrrd_handle3;

  update_state(NeedData);

  inrrd1_ = (NrrdIPort *)get_iport("InputNrrd1");
  inrrd2_ = (NrrdIPort *)get_iport("InputNrrd2");
  inrrd3_ = (NrrdIPort *)get_iport("InputNrrd3");
  onrrd_ = (NrrdOPort *)get_oport("OutputNrrd");

  if (!inrrd1_->get(nrrd_handle1)) 
    first_nrrd_ = false;
  if (!inrrd2_->get(nrrd_handle2)) 
    second_nrrd_ = false;
  if (!inrrd3_->get(nrrd_handle3)) 
    third_nrrd_ = false;
  
  if (first_nrrd_ && !nrrd_handle1.get_rep()) {
    error("Empty InputNrrd1.");
    return;
  }

  if (second_nrrd_ && !nrrd_handle2.get_rep()) {
    error("Empty InputNrrd2.");
    return;
  }

  if (third_nrrd_ && !nrrd_handle3.get_rep()) {
    error("Empty InputNrrd3.");
    return;
  }

  Nrrd *nin1 = 0;
  Nrrd *nin2 = 0;
  Nrrd *nin3 = 0;
  Nrrd *nout = nrrdNew();
  Nrrd *ntmp1 = NULL;
  Nrrd *ntmp2 = NULL;
  Nrrd *ntmp3 = NULL;

  // can either have two nrrds, first nrrd and float, or second
  // nrrd and float
  if (!first_nrrd_ && !second_nrrd_ && !third_nrrd_) {
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
    if (third_nrrd_) {
      ntmp3 = nrrdNew();
      if (nrrdConvert(ntmp3, nrrd_handle3->nrrd, get_type(type_.get()))) {
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
  if (third_nrrd_) {
    if (!usetype_.get())
      nin3 = ntmp3;
    else
      nin3 = nrrd_handle3->nrrd;
  }


  NrrdIter *in1 = nrrdIterNew();
  NrrdIter *in2 = nrrdIterNew();
  NrrdIter *in3 = nrrdIterNew();

  // if a nrrd is connected, it overrides a float
  if (first_nrrd_) {
    nrrdIterSetOwnNrrd(in1, nin1);
  } else {
    nrrdIterSetValue(in1, float1_.get());   
  }

  if (second_nrrd_) {
    nrrdIterSetOwnNrrd(in2, nin2);
  } else {
   nrrdIterSetValue(in2, float2_.get());  
  }

  if (third_nrrd_) {
    nrrdIterSetOwnNrrd(in3, nin3);
  } else {
    nrrdIterSetValue(in3, float3_.get());  
  }

  if (nrrdArithIterTernaryOp(nout, get_op(operator_.get()), in1, in2, in3)) {
    char *err = biffGetDone(NRRD);
    error(string("Error performing 3op to nrrd: ") + err);
    free(err);
  }

  NrrdData *nrrd = scinew NrrdData;
  nrrd->nrrd = nout;

  NrrdDataHandle out(nrrd);

  onrrd_->send(out);
}

unsigned int
Unu3op::get_op(const string &op) {
  if (op == "+") 
    return nrrdTernaryOpAdd;
  else if (op == "x")
    return nrrdTernaryOpMultiply;
  else if (op == "min")
    return nrrdTernaryOpMin;
  else if (op == "max")
    return nrrdTernaryOpMax;
  else if (op == "clamp")
    return nrrdTernaryOpClamp;
  else if (op == "ifelse")
    return nrrdTernaryOpIfElse;
  else if (op == "lerp")
    return nrrdTernaryOpLerp;
  else if (op == "exists")
    return nrrdTernaryOpExists;
  else if (op == "in_op")
    return nrrdTernaryOpInOpen;
  else if (op == "in_cl")
    return nrrdTernaryOpInClosed;
  else {
    error("Unknown operation. Using +");
    return nrrdTernaryOpAdd;
  }
}

unsigned int
Unu3op::get_type(const string &type) {
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


