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

//    File   : TendEvecRGB.cc
//    Author : Martin Cole
//    Date   : Mon Sep  8 09:46:49 2003

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Dataflow/GuiInterface/GuiVar.h>
#include <Dataflow/Network/Ports/NrrdPort.h>
#include <teem/ten.h>

#include <sstream>
#include <iostream>
using std::endl;
#include <stdio.h>

namespace SCITeem {

using namespace SCIRun;

class TendEvecRGB : public Module {
public:
  TendEvecRGB(SCIRun::GuiContext *ctx);
  virtual ~TendEvecRGB();
  virtual void execute();

private:
  unsigned get_method(const string &s) const;

  GuiInt       evec_;
  GuiString    aniso_metric_;
  GuiDouble    background_;
  GuiDouble    gray_;
  GuiDouble    gamma_;
  GuiDouble    threshold_;
};


DECLARE_MAKER(TendEvecRGB)

TendEvecRGB::TendEvecRGB(SCIRun::GuiContext *ctx) : 
  Module("TendEvecRGB", ctx, Filter, "Tend", "Teem"), 
  evec_(get_ctx()->subVar("evec"), 2),
  aniso_metric_(get_ctx()->subVar("aniso_metric"), "tendAniso_FA"),
  background_(get_ctx()->subVar("background"), 0.0),
  gray_(get_ctx()->subVar("gray"), 0.0),
  gamma_(get_ctx()->subVar("gamma"), 1.0),
  threshold_(get_ctx()->subVar("threshold"), 0.5)
{
}


TendEvecRGB::~TendEvecRGB()
{
}


unsigned 
TendEvecRGB::get_method(const string &s) const
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
  if (s == "tenAniso_Tr") {  /* plain old trace */
    return   tenAniso_Tr;
  }
  return 0;
}

void 
TendEvecRGB::execute()
{
  update_state(NeedData);

  NrrdDataHandle nrrd_handle;
  if (!get_input_handle("nin", nrrd_handle)) return;

  reset_vars();

  Nrrd *nin = nrrd_handle->nrrd_;
  Nrrd *nout = nrrdNew();
  
  tenEvecRGBParm *rgbp = tenEvecRGBParmNew();
  rgbp->which = evec_.get(); 
  rgbp->aniso = get_method(aniso_metric_.get()); 
  rgbp->confThresh = threshold_.get(); 
  rgbp->gamma = gamma_.get(); 
  rgbp->bgGray = background_.get(); 
  rgbp->isoGray = gray_.get(); 

  if (tenEvecRGB(nout, nin, rgbp)) {
    char *err = biffGetDone(TEN);
    error(string("Error making tendEvecRGB volume: ") + err);
    free(err);
    return;
  }

  nout->axis[0].kind = nrrdKind3Vector;
  remark("nrrdKind changed to nrrdKind3Vector");

  NrrdDataHandle ntmp(scinew NrrdData(nout));

  send_output_handle("nout", ntmp);
}

} // End namespace SCITeem
