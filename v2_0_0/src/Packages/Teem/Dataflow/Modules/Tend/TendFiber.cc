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
//    File   : TendFiber.cc
//    Author : Martin Cole
//    Date   : Mon Sep  8 09:46:49 2003

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Teem/Dataflow/Ports/NrrdPort.h>
#include <Core/Datatypes/PointCloudMesh.h>
#include <Core/Datatypes/CurveField.h>
#include <Dataflow/Ports/FieldPort.h>
#include <teem/ten.h>

#include <sstream>
#include <iostream>
using std::endl;
#include <stdio.h>

namespace SCITeem {

using namespace SCIRun;

class TendFiber : public Module {
public:
  TendFiber(SCIRun::GuiContext *ctx);
  virtual ~TendFiber();
  virtual void execute();

private:
  unsigned get_aniso(const string &s) const;
  unsigned get_fibertype(const string &s) const;
  unsigned get_integration(const string &s) const;

  NrrdIPort*      inrrd_;
  FieldIPort*     iseeds_;
  FieldOPort*     ofibers_;

  GuiString    fibertype_;
  GuiDouble    puncture_;
  GuiDouble    neighborhood_;
  GuiDouble    stepsize_;
  GuiString    integration_;
  GuiInt       use_aniso_;
  GuiString    aniso_metric_;
  GuiDouble    aniso_thresh_;
  GuiInt       use_length_;
  GuiDouble    length_;
  GuiInt       use_steps_;
  GuiInt       steps_;
  GuiInt       use_conf_;
  GuiDouble    conf_thresh_;
  GuiString    kernel_;
};

DECLARE_MAKER(TendFiber)

TendFiber::TendFiber(SCIRun::GuiContext *ctx) : 
  Module("TendFiber", ctx, Filter, "Tend", "Teem"), 
  fibertype_(ctx->subVar("fibertype")),
  puncture_(ctx->subVar("puncture")),
  neighborhood_(ctx->subVar("neighborhood")),
  stepsize_(ctx->subVar("stepsize")),
  integration_(ctx->subVar("integration")),
  use_aniso_(ctx->subVar("use-aniso")),
  aniso_metric_(ctx->subVar("aniso-metric")),
  aniso_thresh_(ctx->subVar("aniso-thresh")),
  use_length_(ctx->subVar("use-length")),
  length_(ctx->subVar("length")),
  use_steps_(ctx->subVar("use-steps")),
  steps_(ctx->subVar("steps")),
  use_conf_(ctx->subVar("use-conf")),
  conf_thresh_(ctx->subVar("conf-thresh")),
  kernel_(ctx->subVar("kernel"))
{
}

TendFiber::~TendFiber() {
}

unsigned 
TendFiber::get_aniso(const string &s) const
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

unsigned
TendFiber::get_integration(const string &s) const
{
  if (s == "Euler") {
    return tenFiberIntgEuler;
  }
  if (s == "RK4") {
    return tenFiberIntgRK4;
  }
  error("Unknown integration method");
  return tenFiberIntgEuler;
}

unsigned 
TendFiber::get_fibertype(const string &s) const
{
  if (s == "evec1") {
    return tenFiberTypeEvec1;
  }
  if (s == "tensorline") {
    return tenFiberTypeTensorLine;
  }
  if (s == "zhukov") {
    return tenFiberTypeZhukov;
  }
  error("Unknown fiber-tracing algorithm");
  return tenFiberTypeEvec1;
}

void 
TendFiber::execute()
{
  NrrdDataHandle nrrd_handle;
  update_state(NeedData);
  inrrd_ = (NrrdIPort *)get_iport("nin");
  iseeds_ = (FieldIPort *)get_iport("SeedPoints");
  ofibers_ = (FieldOPort *)get_oport("Fibers");

  if (!inrrd_) {
    error("Unable to initialize iport 'nin'.");
    return;
  }
  if (!iseeds_) {
    error("Unable to initialize oport 'SeedPoints'.");
    return;
  }
  if (!ofibers_) {
    error("Unable to initialize oport 'Fibers'.");
    return;
  }

  if (!inrrd_->get(nrrd_handle))
    return;
  if (!nrrd_handle.get_rep()) {
    error("Empty input Nrrd.");
    return;
  }
  Nrrd *nin = nrrd_handle->nrrd;

  FieldHandle fldH;
  if (!iseeds_->get(fldH))
    return;
  if (!fldH.get_rep()) {
    error("Empty field.");
    return;
  }
  PointCloudMesh *pcm = dynamic_cast<PointCloudMesh*>(fldH->mesh().get_rep());
  if (!pcm) {
    error("Input field wasn't a PointCloudField -- use GatherPoints");
    return;
  }

  unsigned fibertype = get_fibertype(fibertype_.get());
  double puncture = puncture_.get();
  double neighborhood = neighborhood_.get();
  double stepsize = stepsize_.get();
  int integration = get_integration(integration_.get());
  bool use_aniso = use_aniso_.get();
  unsigned aniso_metric = get_aniso(aniso_metric_.get());
  double aniso_thresh = aniso_thresh_.get();
  bool use_length = use_length_.get();
  double length = length_.get();
  bool use_steps = use_steps_.get();
  int steps = steps_.get();
  bool use_conf = use_conf_.get();
  double conf_thresh = conf_thresh_.get();
  string kernel = kernel_.get();

  NrrdKernel *kern;
  double p[NRRD_KERNEL_PARMS_NUM];
  memset(p, 0, NRRD_KERNEL_PARMS_NUM * sizeof(double));
  p[0] = 1.0;
  if (kernel == "box") {
    kern = nrrdKernelBox;
  } else if (kernel == "tent") {
    kern = nrrdKernelTent;
  } else if (kernel == "gaussian") { 
    kern = nrrdKernelGaussian; 
    p[1] = 1.0;
    p[2] = 3.0;
  } else if (kernel == "cubicCR") { 
    kern = nrrdKernelBCCubic; 
    p[1] = 0; 
    p[2] = 0.5; 
  } else if (kernel == "cubicBS") { 
    kern = nrrdKernelBCCubic; 
    p[1] = 1; 
    p[2] = 0; 
  } else  { // default is quartic
    kern = nrrdKernelAQuartic; 
    p[1] = 0.0834; 
  }

  tenFiberContext *tfx;
  double start[3];
  int si, stopLen, E;

  E = 0;
  if (use_aniso && !E) 
    E |= tenFiberStopSet(tfx, tenFiberStopAniso, aniso_metric, aniso_thresh);
  if (use_length && !E)
    E |= tenFiberStopSet(tfx, tenFiberStopLength, length);
  if (use_steps && !E)
    E |= tenFiberStopSet(tfx, tenFiberStopNumSteps, steps);
  if (use_conf && !E)
    E |= tenFiberStopSet(tfx, tenFiberStopConfidence, conf_thresh);
  
  if (!E) E |= tenFiberTypeSet(tfx, fibertype);
  if (!E) E |= tenFiberKernelSet(tfx, kern, p);
  if (!E) E |= tenFiberIntgSet(tfx, integration);
  if (!E) E |= tenFiberUpdate(tfx);
  if (E) {
    char *err = biffGetDone(TEN);
    error(string("Error setting fiber stop conditions: ") + err);
    free(err);
    return;
  }

  tenFiberParmSet(tfx, tenFiberParmStepSize, stepsize);
  tenFiberParmSet(tfx, tenFiberParmWPunct, stepsize);

  Nrrd *nout = nrrdNew();

  PointCloudMesh::Node::iterator ib, ie;
  pcm->begin(ib);
  pcm->end(ie);

  Array1<Array1<Point> > fibers;

  int npts=0;

  int fiberIdx=0;
  while (ib != ie) {
    Point p;
    double pp[3];
    pcm->get_center(p, *ib);
    pp[0]=p.x();
    pp[1]=p.y();
    pp[2]=p.z();
    if (tenFiberTrace(tfx, nout, pp)) {
      char *err = biffGetDone(TEN);
      error(string("Error tracing fiber: ") + err);
      free(err);
      return;
    }
    fibers.resize(fiberIdx+1);
    fibers[fiberIdx].resize(nout->axis[0].size);
    double *data = (double *)(nout->data);
    for (int i=0; i<nout->axis[0].size; i+=3)
      fibers[fiberIdx][i] = 
	Point(data[i], data[i+1], data[i+2]);
    ++fiberIdx;
    ++ib;
  }
  
  nrrdNuke(nout);

  CurveMesh *cm = scinew CurveMesh;
  CurveMesh::Node::index_type curr, prev;
  CurveMesh::Node::array_type a;
  a.resize(2);
  for (int i=0; i<fibers.size(); i++) {
    if (fibers[i].size()) {
      a[1] = cm->add_point(fibers[i][0]);
      for (int j=1; j<fibers[i].size(); j++) {
	a[0] = a[1];
	a[1] = cm->add_point(fibers[i][j]);
	cm->add_elem(a);
      }
    }
  }
  
  CurveField<double> *cf = scinew CurveField<double>(cm, Field::NONE);
  ofibers_->send(cf);
}

} // End namespace SCITeem
