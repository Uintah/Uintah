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

class TendSim : public Module {
public:
  TendSim(SCIRun::GuiContext *ctx);
  virtual ~TendSim();
  virtual void execute();

private:
  GuiDouble       bvalue_;
};

DECLARE_MAKER(TendSim)

TendSim::TendSim(SCIRun::GuiContext *ctx) : 
  Module("TendSim", ctx, Filter, "Tend", "Teem"),
  bvalue_(get_ctx()->subVar("bvalue"), 1.0)
{
}


TendSim::~TendSim()
{
}


void 
TendSim::execute()
{
  update_state(NeedData);

  NrrdDataHandle bmat_handle;
  if (!get_input_handle("BMatrixNrrd", bmat_handle)) return;

  NrrdDataHandle referenceimg_handle;
  if (!get_input_handle("ReferenceNrrd", referenceimg_handle)) return;

  NrrdDataHandle tensor_handle;
  if (!get_input_handle("TensorNrrd", tensor_handle)) return;

  Nrrd *bmat = bmat_handle->nrrd_;
  Nrrd *referenceimg = referenceimg_handle->nrrd_;
  Nrrd *tensor = tensor_handle->nrrd_;
  Nrrd *nout = nrrdNew();

  if (tenSimulate(nout, referenceimg, tensor, bmat, bvalue_.get())) {
    char *err = biffGetDone(TEN);
    error(string("Error simulating tensors: ") + err);
    free(err);
    return;
  }

  NrrdDataHandle ntmp(scinew NrrdData(nout));

  send_output_handle("OutputNrrd", ntmp);
}

} // End namespace SCITeem
