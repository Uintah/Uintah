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
//    File   : TendAnvol.cc
//    Author : Martin Cole
//    Date   : Mon Sep  8 09:46:49 2003

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Teem/Dataflow/Ports/NrrdPort.h>
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

  if (!inrrd_) {
    error("Unable to initialize iport 'Nrrd'.");
    return;
  }
  if (!onrrd_) {
    error("Unable to initialize oport 'Nrrd'.");
    return;
  }
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

  Nrrd *ntup = nrrdNew();
  nrrdAxesInsert(ntup, nout, 0);
  ntup->axis[0].label = strdup("Aniso:Scalar");
  //nrrd->copy_sci_data(*nrrd_handle.get_rep());
  nrrdNuke(nout);
  nrrd->nrrd = ntup;
  onrrd_->send(NrrdDataHandle(nrrd));
}

} // End namespace SCITeem
