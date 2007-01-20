/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
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

//    File   : TendMake.cc
//    Author : Martin Cole
//    Date   : Mon Sep  8 09:46:49 2003

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Dataflow/GuiInterface/GuiVar.h>
#include <Dataflow/Network/Ports/NrrdPort.h>
#include <teem/ten.h>

namespace SCITeem {

using namespace SCIRun;

class TendMake : public Module {
public:
  TendMake(SCIRun::GuiContext *ctx);
  virtual ~TendMake();
  virtual void execute();
};


DECLARE_MAKER(TendMake)

TendMake::TendMake(SCIRun::GuiContext *ctx) : 
  Module("TendMake", ctx, Filter, "Tend", "Teem")
{
}


TendMake::~TendMake()
{
}


void 
TendMake::execute()
{
  update_state(NeedData);

  NrrdDataHandle conf_handle;
  if (!get_input_handle("Confidence", conf_handle)) return;

  NrrdDataHandle eval_handle;
  if (!get_input_handle("Evals", eval_handle)) return;

  NrrdDataHandle evec_handle;
  if (!get_input_handle("Evecs", evec_handle)) return;

  Nrrd *confidence = conf_handle->nrrd_;
  Nrrd *eval = eval_handle->nrrd_;
  Nrrd *evec = evec_handle->nrrd_;
  Nrrd *nout = nrrdNew();

  if (tenMake(nout, confidence, eval, evec))
  {
    char *err = biffGetDone(TEN);
    error(string("Error creating DT volume: ") + err);
    free(err);
    return;
  }

  NrrdDataHandle out(scinew NrrdData(nout));

  send_output_handle("OutputNrrd", out);
}


} // End namespace SCITeem
