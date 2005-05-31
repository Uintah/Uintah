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

//    File   : TendMake.cc
//    Author : Martin Cole
//    Date   : Mon Sep  8 09:46:49 2003

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Dataflow/Ports/NrrdPort.h>
#include <teem/ten.h>

namespace SCITeem {

using namespace SCIRun;

class TendMake : public Module {
public:
  TendMake(SCIRun::GuiContext *ctx);
  virtual ~TendMake();
  virtual void execute();

private:
  NrrdIPort*      inconfidence_;
  NrrdIPort*      inevals_;
  NrrdIPort*      inevecs_;
  NrrdOPort*      onrrd_;

};

DECLARE_MAKER(TendMake)

TendMake::TendMake(SCIRun::GuiContext *ctx) : 
  Module("TendMake", ctx, Filter, "Tend", "Teem")
{
}

TendMake::~TendMake() {
}

void 
TendMake::execute()
{
  NrrdDataHandle conf_handle;
  NrrdDataHandle eval_handle;
  NrrdDataHandle evec_handle;
  update_state(NeedData);
  inconfidence_ = (NrrdIPort *)get_iport("Confidence");
  inevals_ = (NrrdIPort *)get_iport("Evals");
  inevecs_ = (NrrdIPort *)get_iport("Evecs");

  onrrd_ = (NrrdOPort *)get_oport("OutputNrrd");

  if (!inconfidence_->get(conf_handle))
    return;
  if (!inevals_->get(eval_handle))
    return;
  if (!inevecs_->get(evec_handle))
    return;

  if (!conf_handle.get_rep()) {
    error("Empty input Confidence Nrrd.");
    return;
  }
  if (!eval_handle.get_rep()) {
    error("Empty input Evals Nrrd.");
    return;
  }
  if (!evec_handle.get_rep()) {
    error("Empty input Evecs Nrrd.");
    return;
  }

  Nrrd *confidence = conf_handle->nrrd;
  Nrrd *eval = eval_handle->nrrd;
  Nrrd *evec = evec_handle->nrrd;
  Nrrd *nout = nrrdNew();

  if (tenMake(nout, confidence, eval, evec)) {
    char *err = biffGetDone(TEN);
    error(string("Error creating DT volume: ") + err);
    free(err);
    return;
  }

  NrrdData *nrrd = scinew NrrdData;
  nrrd->nrrd = nout;

  NrrdDataHandle out(nrrd);

  onrrd_->send(out);

}

} // End namespace SCITeem
