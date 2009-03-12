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

//    File   : Unu1op.cc
//    Author : Martin Cole
//    Date   : Mon Sep  8 09:46:49 2003

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Dataflow/Network/Ports/NrrdPort.h>

namespace SCITeem {

using namespace SCIRun;

class Unu1op : public Module {
public:
  Unu1op(SCIRun::GuiContext *ctx);
  virtual ~Unu1op();
  virtual void execute();

private:
  GuiString    operator_;
  GuiString    type_;
  GuiInt       usetype_;

  unsigned int get_op(const string &op);
};


DECLARE_MAKER(Unu1op)

Unu1op::Unu1op(SCIRun::GuiContext *ctx) : 
  Module("Unu1op", ctx, Filter, "UnuAtoM", "Teem"), 
  operator_(get_ctx()->subVar("operator")),
  type_(get_ctx()->subVar("type")),
  usetype_(get_ctx()->subVar("usetype"))
{
}


Unu1op::~Unu1op()
{
}


void 
Unu1op::execute()
{
  update_state(NeedData);

  NrrdDataHandle nrrd_handle;
  if (!get_input_handle("InputNrrd", nrrd_handle)) return;

  reset_vars();

  Nrrd *nin = nrrd_handle->nrrd_;
  Nrrd *nout = nrrdNew();
  Nrrd *ntmp = NULL;

  if (!usetype_.get())
  {
    ntmp = nrrdNew();
    if (nrrdConvert(ntmp, nin, string_to_nrrd_type(type_.get())))
    {
      char *err = biffGetDone(NRRD);
      error(string("Error converting nrrd: ") + err);
      free(err);
      return;
    }
    if (nrrdArithUnaryOp(nout, get_op(operator_.get()), ntmp))
    {
      char *err = biffGetDone(NRRD);
      error(string("Error performing 1op to nrrd: ") + err);
      free(err);
      return;
    }
  }
  else
  {
    if (nrrdArithUnaryOp(nout, get_op(operator_.get()), nin))
    {
      char *err = biffGetDone(NRRD);
      error(string("Error performing 1op to nrrd: ") + err);
      free(err);
      return;
    }
  }

  nrrdKeyValueCopy(nout, nin);

  NrrdDataHandle out(scinew NrrdData(nout));

  send_output_handle("OutputNrrd", out);
}


unsigned int
Unu1op::get_op(const string &op)
{
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
    error("Unknown unary op. Performing random operation.");
    return nrrdUnaryOpRand;
  }
}


} // End namespace SCITeem


