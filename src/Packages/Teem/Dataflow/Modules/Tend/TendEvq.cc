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

//    File   : TendEvq.cc
//    Author : Martin Cole
//    Date   : Mon Sep  8 09:46:49 2003

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Dataflow/Ports/NrrdPort.h>
#include <teem/ten.h>

namespace SCITeem {

using namespace SCIRun;

class TendEvq : public Module {
public:
  TendEvq(SCIRun::GuiContext *ctx);
  virtual ~TendEvq();
  virtual void execute();

private:
  NrrdIPort *     inrrd_;
  NrrdOPort*      onrrd_;

  GuiInt          index_;
  GuiString       anisotropy_;
  GuiInt          ns_;

  unsigned int get_anisotropy(const string &an);

};

DECLARE_MAKER(TendEvq)

TendEvq::TendEvq(SCIRun::GuiContext *ctx) : 
  Module("TendEvq", ctx, Filter, "Tend", "Teem"),
  index_(ctx->subVar("index")),
  anisotropy_(ctx->subVar("anisotropy")),
  ns_(ctx->subVar("ns"))
{
}

TendEvq::~TendEvq() {
}

void 
TendEvq::execute()
{
  NrrdDataHandle nrrd_handle;

  update_state(NeedData);
  inrrd_ = (NrrdIPort *)get_iport("InputNrrd");

  onrrd_ = (NrrdOPort *)get_oport("OutputNrrd");

  if (!inrrd_->get(nrrd_handle))
    return;

  if (!nrrd_handle.get_rep()) {
    error("Empty input InputNrrd.");
    return;
  }

  Nrrd *nin = nrrd_handle->nrrd;
  Nrrd *nout = nrrdNew();

  if (tenEvqVolume(nout, nin, index_.get(), get_anisotropy(anisotropy_.get()), ns_.get())) {
    char *err = biffGetDone(TEN);
    error(string("Error quantizing directions of diffusions: ") + err);
    free(err);
    return;
  }

  NrrdData *nrrd = scinew NrrdData;
  nrrd->nrrd = nout;

  NrrdDataHandle out(nrrd);

  onrrd_->send(out);

}

unsigned int
TendEvq::get_anisotropy(const string &an) {
  if (an == "cl1")
    return tenAniso_Cl1;
  else if (an == "cl2")
    return tenAniso_Cl2;
  else if (an == "cp1")
    return tenAniso_Cp1;
  else if (an == "cp2")
    return tenAniso_Cp2;
  else if (an == "ca1")
    return tenAniso_Ca1;
  else if (an == "ca2")
    return tenAniso_Ca2;
  else if (an == "cs1")
    return tenAniso_Cs1;
  else if (an == "cs2")
    return tenAniso_Cs2;
  else if (an == "ct1")
    return tenAniso_Ct1;
  else if (an == "ct2")
    return tenAniso_Ct1;
  else if (an == "ra")
    return tenAniso_RA;
  else if (an == "fa")
    return tenAniso_FA;
  else if (an == "vf")
    return tenAniso_VF;
  else if (an == "tr")
    return tenAniso_Tr;
  else {
    error("Unkown anisotropy metric.  Using trace");
    return tenAniso_Tr;
  }
  
}

} // End namespace SCITeem


