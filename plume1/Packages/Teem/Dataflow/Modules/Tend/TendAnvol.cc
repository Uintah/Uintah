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

//    File   : TendAnvol.cc
//    Author : Martin Cole
//    Date   : Mon Sep  8 09:46:49 2003

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Dataflow/Ports/NrrdPort.h>
#include <teem/ten.h>

#include <sstream>
#include <iostream>
using std::endl;
#include <stdio.h>

namespace SCITeem {

using namespace SCIRun;

class TendAnvol : public Module {
public:
  TendAnvol(SCIRun::GuiContext *ctx);
  virtual ~TendAnvol();
  virtual void execute();

private:
  unsigned get_method(const string &s) const;

  NrrdIPort*      inrrd_;
  NrrdOPort*      onrrd_;

  GuiString    aniso_metric_;
  GuiDouble    threshold_;
};

DECLARE_MAKER(TendAnvol)

TendAnvol::TendAnvol(SCIRun::GuiContext *ctx) : 
  Module("TendAnvol", ctx, Filter, "Tend", "Teem"), 
  aniso_metric_(ctx->subVar("aniso_metric")),
  threshold_(ctx->subVar("threshold"))
{
}

TendAnvol::~TendAnvol() {
}


unsigned 
TendAnvol::get_method(const string &s) const
{
  if (s == "tenAniso_Cl1") { /* Westin's linear (first version) */
    return tenAniso_Cl1;
  }
  if (s == "tenAniso_Cp1") { /* Westin's planar (first version) */
    return   tenAniso_Cp1;
  }
  if (s == "tenAniso_Ca1") { /* Westin's linear + planar (first version) */
    return   tenAniso_Ca1;
  }
  if (s == "tenAniso_Cs1") { /* Westin's spherical (first version) */
    return   tenAniso_Cs1;
  }
  if (s == "tenAniso_Ct1") { /* gk's anisotropy type (first version) */
    return   tenAniso_Ct1;
  }
  if (s == "tenAniso_Cl2") { /* Westin's linear (second version) */
    return   tenAniso_Cl2;
  }
  if (s == "tenAniso_Cp2") { /* Westin's planar (second version) */
    return   tenAniso_Cp2;
  }
  if (s == "tenAniso_Ca2") { /* Westin's linear + planar (second version) */
    return   tenAniso_Ca2;
  }
  if (s == "tenAniso_Cs2") { /* Westin's spherical (second version) */
    return   tenAniso_Cs2;
  }
  if (s == "tenAniso_Ct2") { /* gk's anisotropy type (second version) */
    return   tenAniso_Ct2;
  }
  if (s == "tenAniso_RA") {  /* Bass+Pier's relative anisotropy */
    return   tenAniso_RA;
  }
  if (s == "tenAniso_FA") {  /* (Bass+Pier's fractional anisotropy)/sqrt(2) */
    return   tenAniso_FA;
  }
  if (s == "tenAniso_VF") {  /* volume fraction= 1-(Bass+Pier's volume ratio)*/
    return   tenAniso_VF;
  }
  if (s == "tenAniso_Q") {   /* radius of root circle is 2*sqrt(Q/9) 
				his is 9 times proper Q in cubic solution) */
    return   tenAniso_Q;
  }
		             
  if (s == "tenAniso_R") {   /* phase of root circle is acos(R/Q^3) */
    return   tenAniso_R;
  }
  if (s == "tenAniso_S") {   /* sqrt(Q^3 - R^2) */
    return   tenAniso_S;
  }
  if (s == "tenAniso_Th") {  /* R/Q^3 */
    return   tenAniso_Th;
  }
  if (s == "tenAniso_Cz") {  /* Zhukov's invariant-based anisotropy metric */
    return   tenAniso_Cz;
  }
  if (s == "tenAniso_Tr") {  /* plain old trace */
    return   tenAniso_Tr;
  }
  return 0;
}

void 
TendAnvol::execute()
{
  NrrdDataHandle nrrd_handle;
  update_state(NeedData);
  inrrd_ = (NrrdIPort *)get_iport("nin");
  onrrd_ = (NrrdOPort *)get_oport("nout");

  if (!inrrd_->get(nrrd_handle))
    return;

  if (!nrrd_handle.get_rep()) {
    error("Empty input Nrrd.");
    return;
  }
  reset_vars();

  Nrrd *nin = nrrd_handle->nrrd;
  Nrrd *nout = nrrdNew();

  if (tenAnisoVolume(nout, nin, get_method(aniso_metric_.get()), 
		     threshold_.get())) {
    char *err = biffGetDone(TEN);
    error(string("Error making aniso volume: ") + err);
    free(err);
    return;
  }

  NrrdData *nrrd = scinew NrrdData;

  //Nrrd *ntup = nrrdNew();
  //nrrdAxesInsert(ntup, nout, 0);
  //ntup->axis[0].label = airStrdup("Aniso:Scalar");
  //nrrd->copy_sci_data(*nrrd_handle.get_rep());
  //nrrdNuke(nout);
  //nrrd->nrrd = ntup;
  nrrd->nrrd = nout;
  //nrrd->copy_sci_data(*nrrd_handle.get_rep()); 
  onrrd_->send(NrrdDataHandle(nrrd));
}

} // End namespace SCITeem
