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
//    File   : TendPoint.cc
//    Author : Martin Cole
//    Date   : Mon Sep  8 09:46:49 2003

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Teem/Dataflow/Ports/NrrdPort.h>
#include <Core/Containers/StringUtil.h>
#include <teem/ten.h>

namespace SCITeem {

using namespace SCIRun;

class TendPoint : public Module {
public:
  TendPoint(SCIRun::GuiContext *ctx);
  virtual ~TendPoint();
  virtual void execute();

private:
  NrrdIPort*      inrrd_;
  
  GuiInt          x_;
  GuiInt          y_;
  GuiInt          z_;
  GuiDouble       confidence_;
  GuiDouble       tensor1_, tensor2_, tensor3_, tensor4_, tensor5_, tensor6_;
  GuiDouble       eval1_, eval2_, eval3_;
  GuiDouble       evec1_, evec2_, evec3_, evec4_, evec5_;
  GuiDouble       evec6_, evec7_, evec8_, evec9_;
  GuiDouble       angle_;
  GuiDouble       axis1_, axis2_, axis3_;
  GuiDouble       mat1_, mat2_, mat3_, mat4_, mat5_;
  GuiDouble       mat6_, mat7_, mat8_, mat9_;
  GuiDouble       cl1_;
  GuiDouble       cp1_;
  GuiDouble       ca1_;
  GuiDouble       cs1_;
  GuiDouble       ct1_;
  GuiDouble       cl2_;
  GuiDouble       cp2_;
  GuiDouble       ca2_;
  GuiDouble       cs2_;
  GuiDouble       ct2_;
  GuiDouble       ra_;
  GuiDouble       fa_;
  GuiDouble       vf_;
  GuiDouble       b_;
  GuiDouble       q_;
  GuiDouble       r_;
  GuiDouble       s_;
  GuiDouble       skew_;
  GuiDouble       th_;
  GuiDouble       cz_;
  GuiDouble       det_;
  GuiDouble       tr_;
};

DECLARE_MAKER(TendPoint)

TendPoint::TendPoint(SCIRun::GuiContext *ctx) : 
  Module("TendPoint", ctx, Filter, "Tend", "Teem"), 
  x_(ctx->subVar("x")),
  y_(ctx->subVar("y")),
  z_(ctx->subVar("z")),
  confidence_(ctx->subVar("confidence")),
  tensor1_(ctx->subVar("tensor1")),  tensor2_(ctx->subVar("tensor2")),
  tensor3_(ctx->subVar("tensor3")),  tensor4_(ctx->subVar("tensor4")),
  tensor5_(ctx->subVar("tensor5")),  tensor6_(ctx->subVar("tensor6")),
  eval1_(ctx->subVar("eval1")),  eval2_(ctx->subVar("eval2")),
  eval3_(ctx->subVar("eval3")),  evec1_(ctx->subVar("evec1")),
  evec2_(ctx->subVar("evec2")),  evec3_(ctx->subVar("evec3")),
  evec4_(ctx->subVar("evec4")),  evec5_(ctx->subVar("evec5")),
  evec6_(ctx->subVar("evec6")),  evec7_(ctx->subVar("evec7")),
  evec8_(ctx->subVar("evec8")),  evec9_(ctx->subVar("evec9")),
  angle_(ctx->subVar("angle")),  axis1_(ctx->subVar("axis1")),
  axis2_(ctx->subVar("axis2")),  axis3_(ctx->subVar("axis3")),
  mat1_(ctx->subVar("mat1")),  mat2_(ctx->subVar("mat2")),
  mat3_(ctx->subVar("mat3")),  mat4_(ctx->subVar("mat4")),
  mat5_(ctx->subVar("mat5")),  mat6_(ctx->subVar("mat6")),
  mat7_(ctx->subVar("mat7")),  mat8_(ctx->subVar("mat8")),
  mat9_(ctx->subVar("mat9")),  cl1_(ctx->subVar("cl1")),
  cp1_(ctx->subVar("cp1")),  ca1_(ctx->subVar("ca1")),
  cs1_(ctx->subVar("cs1")),  ct1_(ctx->subVar("ct1")),
  cl2_(ctx->subVar("cl2")),  cp2_(ctx->subVar("cp2")),
  ca2_(ctx->subVar("ca2")),  cs2_(ctx->subVar("cs2")),
  ct2_(ctx->subVar("ct2")),  ra_(ctx->subVar("ra")),
  fa_(ctx->subVar("fa")),  vf_(ctx->subVar("vf")),
  b_(ctx->subVar("b")),  q_(ctx->subVar("q")),
  r_(ctx->subVar("r")),  s_(ctx->subVar("s")),
  skew_(ctx->subVar("skew")),  th_(ctx->subVar("th")),
  cz_(ctx->subVar("cz")),  det_(ctx->subVar("det")),
  tr_(ctx->subVar("tr"))
{
}

TendPoint::~TendPoint() {
}

void 
TendPoint::execute()
{
  NrrdDataHandle nrrd_handle;

  update_state(NeedData);

  inrrd_ = (NrrdIPort *)get_iport("InputNrrd");

  if (!inrrd_) {
    error("Unable to initialize iport 'InputNrrd'.");
    return;
  }

  if (!inrrd_->get(nrrd_handle))
    return;

  if (!nrrd_handle.get_rep()) {
    error("Empty input InputNrrd.");
    return;
  }

  reset_vars();

  Nrrd *nin = nrrd_handle->nrrd;

  if (tenTensorCheck(nin, nrrdTypeFloat, AIR_TRUE, AIR_TRUE)) {
    char *err = biffGetDone(TEN);
    error(string("Didn't get a valid DT volume: ") + err);
    free(err);
    return;
  }

  int loc[3];
  loc[0] = x_.get();
  loc[1] = y_.get();
  loc[2] = z_.get();

  int sx = 0, sy = 0, sz = 0;
  sx = nin->axis[1].size;
  sy = nin->axis[2].size;
  sz = nin->axis[3].size;

  if (!( AIR_IN_CL(0, loc[0], sx-1) &&
	 AIR_IN_CL(0, loc[1], sy-1) &&
	 AIR_IN_CL(0, loc[2], sz-1) )) {
    char *err = biffGetDone(TEN);
    error(string("Location : " + to_string(loc[0]) + ", " +
		 to_string(loc[1]) + ", " + to_string(loc[2]) + 
		 " not inside volume" + err));
    free(err);
    return;
  }
  
  float eval[3], evec[9], c[TEN_ANISO_MAX+1];
  float * tdata =  (float*)(nin->data) + 7*(loc[0] + sx*(loc[1] + sy*loc[2]));
  fprintf(stderr, "location = (%d,%d,%d)\n", loc[0], loc[1], loc[2]);
  fprintf(stderr, "confidence = %g\n", tdata[0]);
  confidence_.set(tdata[0]);
  fprintf(stderr, "tensor =\n");
  fprintf(stderr, "{%.7f,%.7f,%.7f,%.7f,%.7f,%.7f} = \n",
	  tdata[1], tdata[2], tdata[3], tdata[4], tdata[5], tdata[6]);
  tensor1_.set(tdata[1]);
  tensor2_.set(tdata[2]);
  tensor3_.set(tdata[3]);
  tensor4_.set(tdata[4]);
  tensor5_.set(tdata[5]);
  tensor6_.set(tdata[6]);
  fprintf(stderr, "% 15.7f % 15.7f % 15.7f\n", tdata[1], tdata[2], tdata[3]);
  fprintf(stderr, "% 15.7f % 15.7f % 15.7f\n", tdata[2], tdata[4], tdata[5]);
  fprintf(stderr, "% 15.7f % 15.7f % 15.7f\n", tdata[3], tdata[5], tdata[6]);
  tenEigensolve_f(eval, evec, tdata);
  eval1_.set(eval[0]);
  eval2_.set(eval[1]);
  eval3_.set(eval[2]);
  evec1_.set(evec[0]);
  evec2_.set(evec[1]);
  evec3_.set(evec[2]);
  evec4_.set(evec[3]);
  evec5_.set(evec[4]);
  evec6_.set(evec[5]);
  evec7_.set(evec[6]);
  evec8_.set(evec[7]);
  evec9_.set(evec[8]);
  fprintf(stderr, "eigensystem = (<eigenvalue> : <eigenvector>):\n");
  fprintf(stderr, "% 15.7f : % 15.7f % 15.7f % 15.7f\n",
	  eval[0], evec[0], evec[1], evec[2]);
  fprintf(stderr, "% 15.7f : % 15.7f % 15.7f % 15.7f\n",
	  eval[1], evec[3], evec[4], evec[5]);
  fprintf(stderr, "% 15.7f : % 15.7f % 15.7f % 15.7f\n",
	  eval[2], evec[6], evec[7], evec[8]);
  float axis[3], mat[0];
  float angle = ell_3m_to_aa_f(axis, evec);
  angle_.set(angle);
  axis1_.set(axis[0]);
  axis2_.set(axis[1]);
  axis3_.set(axis[2]);
  fprintf(stderr, "eigenvector rotation: %g around {%g,%g,%g}\n",
	  angle, axis[0], axis[1], axis[2]);
  ell_aa_to_3m_f(mat, angle, axis);
  mat1_.set(mat[0]);
  mat2_.set(mat[1]);
  mat3_.set(mat[2]);
  mat4_.set(mat[3]);
  mat5_.set(mat[4]);
  mat6_.set(mat[5]);
  mat7_.set(mat[6]);
  mat8_.set(mat[7]);
  mat9_.set(mat[8]);
  fprintf(stderr, "% 15.7f % 15.7f % 15.7f\n",
	  mat[0], mat[1], mat[2]);
  fprintf(stderr, "% 15.7f % 15.7f % 15.7f\n",
	  mat[3], mat[4], mat[5]);
  fprintf(stderr, "% 15.7f % 15.7f % 15.7f\n",
	  mat[6], mat[7], mat[8]);
  tenAnisoCalc(c, eval);
  fprintf(stderr, "anisotropies = \n");
  cl1_.set(c[1]);
  cp1_.set(c[2]);
  ca1_.set(c[3]);
  cs1_.set(c[4]);
  ct1_.set(c[5]);
  cl2_.set(c[6]);
  cp2_.set(c[7]);
  ca2_.set(c[8]);
  cs2_.set(c[9]);
  ct2_.set(c[10]);
  ra_.set(c[11]);
  fa_.set(c[12]);
  vf_.set(c[13]);
  b_.set(c[14]);
  q_.set(c[15]);
  r_.set(c[16]);
  s_.set(c[17]);
  skew_.set(c[18]);
  th_.set(c[19]);
  cz_.set(c[20]);
  det_.set(c[21]);
  tr_.set(c[22]);
  for (int i=1; i<=TEN_ANISO_MAX; i++) {
    fprintf(stderr, "%s: % 15.7f\n",
	    airEnumStr(tenAniso, i), c[i]);
  }
}

} // End namespace SCITeem
