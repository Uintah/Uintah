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

//    File   : NrrdGradient.cc
//    Author : Martin Cole
//    Date   : Mon Sep  8 09:46:49 2003

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Dataflow/Ports/NrrdPort.h>
#include <Core/Containers/StringUtil.h>

namespace SCITeem {

using namespace SCIRun;

class NrrdGradient : public Module {
public:
  NrrdGradient(SCIRun::GuiContext *ctx);
  virtual ~NrrdGradient();
  virtual void execute();

private:
  NrrdIPort*      inrrd_;
  NrrdOPort*      onrrd_;
};

DECLARE_MAKER(NrrdGradient)

NrrdGradient::NrrdGradient(SCIRun::GuiContext *ctx) : 
  Module("NrrdGradient", ctx, Filter, "NrrdData", "Teem")
{
}

NrrdGradient::~NrrdGradient() {
}

void 
NrrdGradient::execute()
{
  NrrdDataHandle nin_handle;
  update_state(NeedData);
  inrrd_ = (NrrdIPort *)get_iport("InputNrrd");
  onrrd_ = (NrrdOPort *)get_oport("OutputNrrd");

  if (!inrrd_->get(nin_handle))
    return;

  if (!nin_handle.get_rep()) {
    error("Empty input Nrrd.");
    return;
  }

  Nrrd *nin = nin_handle->nrrd;
  Nrrd *nout = nrrdNew();
  int size[NRRD_DIM_MAX];
  int dim;

  // Create a local array of axis sizes, so we can allocate the output Nrrd
  for (dim=0; dim<nin->dim; dim++)
    size[dim+1]=nin->axis[dim].size;
  size[0]=nin->dim;

  // Allocate the nrrd's data, set the size of each axis
  nrrdAlloc_nva(nout, nrrdTypeDouble, nin->dim+1, size);

  // Set axis info for (new) axis 0
  if (nin->dim == 2) nout->axis[0].kind=nrrdKind2Vector;
  else if (nin->dim == 3) nout->axis[0].kind=nrrdKind3Vector;
  else if (nin->dim == 4) nout->axis[0].kind=nrrdKind4Vector;
  else nout->axis[0].kind=nrrdKindList;
  nout->axis[0].center=nin->axis[0].center;
  nout->axis[0].spacing=AIR_NAN;
  nout->axis[0].min=AIR_NAN;
  nout->axis[0].max=AIR_NAN;

  double spacing_inv[NRRD_DIM_MAX];
  double min, max, spacing;
  int center;

  // Copy the other axes' info
  int nsamples=1;
  for (dim=0; dim<nin->dim; dim++) {
    nout->axis[dim+1].kind=nin->axis[dim].kind;
    nout->axis[dim+1].center=center=nin->axis[dim].center;
    nout->axis[dim+1].spacing=spacing=nin->axis[dim].spacing;
    nout->axis[dim+1].min=min=nin->axis[dim].min;
    nout->axis[dim+1].max=max=nin->axis[dim].max;
    nsamples *= nin->axis[dim].size;
    // Note spacing so we can correctly scale derivatives
    if (min != AIR_NAN && max != AIR_NAN) {
      if (center == nrrdCenterNode) 
	spacing_inv[dim]=(size[dim+1]-1.)/(max-min);
      else
	spacing_inv[dim]=(size[dim+1]*1.)/(max-min);
    } else {
      if (spacing != AIR_NAN)
	spacing_inv[dim]=spacing;
      else
	spacing_inv[dim]=AIR_NAN;
    }
  }
  
  // Build the resampling kernel
  NrrdResampleInfo *infoD = nrrdResampleInfoNew();
  infoD->boundary = nrrdBoundaryBleed;
  infoD->type = nrrdTypeDouble;
  infoD->renormalize = AIR_TRUE;
  for (dim=0; dim<nin->dim; dim++) {
    infoD->samples[dim] = nin->axis[dim].size;
    infoD->min[dim] = nin->axis[dim].min;
    infoD->max[dim] = nin->axis[dim].max;
  }

  // This Nrrd will be reused through each loop iteration -- it holds
  //   the first spatial derivative of 'nin' with respect to the current
  //   dimension, 'dim'
  Nrrd *deriv=nrrdNew();
  for (dim=0; dim<nin->dim; dim++) {
    for (int d=0; d<nin->dim; d++) 
      if (d==dim)
	infoD->kernel[d]=nrrdKernelCentDiff;
      else
	infoD->kernel[d]=NULL;
    nrrdSpatialResample(deriv, nin, infoD);

    // copy data out of NrrdCurr into nout
    double *nout_data_ptr = (double *)(nout->data);
    double *deriv_data_ptr = (double *)(deriv->data);
    nout_data_ptr += dim;
    for (int s=0; s<nsamples; s++, nout_data_ptr+=nin->dim, deriv_data_ptr++)
      *nout_data_ptr = *deriv_data_ptr * spacing_inv[dim];
  }
  nrrdNuke(deriv);
  nrrdResampleInfoNix(infoD);

  // Create SCIRun data structure wrapped around nout
  NrrdData *nd = scinew NrrdData;
  nd->nrrd = nout;
  NrrdDataHandle nout_handle(nd);

  // Copy the properties
  nout_handle->copy_properties(nin_handle.get_rep());

  onrrd_->send(nout_handle);
}

} // End namespace SCITeem
