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

//    File   : NrrdSetupTexture.cc
//    Author : Michael Callahan
//    Date   : Feb 2005

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Dataflow/Ports/NrrdPort.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Geometry/Transform.h>

namespace SCITeem {

using namespace SCIRun;

class NrrdSetupTexture : public Module {
public:
  NrrdSetupTexture(SCIRun::GuiContext *ctx);
  virtual ~NrrdSetupTexture();
  virtual void execute();

  GuiDouble minf_;
  GuiDouble maxf_;
  GuiInt useinputmin_;
  GuiInt useinputmax_;
  GuiDouble realmin_;
  GuiDouble realmax_;
  double last_minf_;
  double last_maxf_;
  int last_generation_;
  //NrrdDataHandle last_nrrdH_;

};

DECLARE_MAKER(NrrdSetupTexture)

NrrdSetupTexture::NrrdSetupTexture(SCIRun::GuiContext *ctx) : 
  Module("NrrdSetupTexture", ctx, Filter, "NrrdData", "Teem"),
  minf_(ctx->subVar("minf")),
  maxf_(ctx->subVar("maxf")),
  useinputmin_(ctx->subVar("useinputmin")),
  useinputmax_(ctx->subVar("useinputmax")),
  realmin_(ctx->subVar("realmin")),
  realmax_(ctx->subVar("realmax")),
  last_minf_(0),
  last_maxf_(0),
  last_generation_(-1)
  //last_nrrdH_(0)
{
}

NrrdSetupTexture::~NrrdSetupTexture()
{
}


static inline
unsigned char
NtoUC(double a)
{
  return (unsigned char)(a * 127.5 + 127.5);
}


static inline
unsigned char
VtoUC(double v, double dmin, double dmaxplus)
{
  return (unsigned char)((v - dmin) * dmaxplus);
}


template <class T>
void
compute_data(T *nindata, unsigned char *nvoutdata, float *gmoutdata,
             const int ni, const int nj, const int nk, Transform &transform,
             double dmin, double dmax)
{    
  // Compute the data.
  int i, j, k;
  //const unsigned int nk = nin->axis[0].size;
  const unsigned int njk = nj * nk;
  const double dmaxplus = (dmax - dmin) * 255.0;
  for (i = 0; i < ni; i++)
  {
    for (j = 0; j < nj; j++)
    {
      for (k = 0; k < nk; k++)
      {
        double xscale = 0.5;
        double yscale = 0.5;
        double zscale = 0.5;
        int i0 = i-1;
        int i1 = i+1;
        int j0 = j-1;
        int j1 = j+1;
        int k0 = k-1;
        int k1 = k+1;
        if (i == 0)   { i0 = i; xscale = 1.0; }
        if (i1 == ni) { i1 = i; xscale = 1.0; }
        if (j == 0)   { j0 = j; xscale = 1.0; }
        if (j1 == nj) { j1 = j; xscale = 1.0; }
        if (k == 0)   { k0 = k; xscale = 1.0; }
        if (k1 == nk) { k1 = k; xscale = 1.0; }

        const double val = (double)nindata[i * njk + j * nk + k];
        const double x0 = nindata[i0 * njk + j * nk + k];
        const double x1 = nindata[i1 * njk + j * nk + k];
        const double y0 = nindata[i * njk + j0 * nk + k];
        const double y1 = nindata[i * njk + j1 * nk + k];
        const double z0 = nindata[i * njk + j * nk + k0];
        const double z1 = nindata[i * njk + j * nk + k1];

        const Vector g((x1 - x0)*xscale, (y1 - y0)*yscale, (z1 - z0)*zscale);
        Vector gradient = transform.project_normal(g);
        const float gm = gradient.safe_normalize();

        nvoutdata[(i * njk + j * nk + k) * 4 + 0] = NtoUC(gradient.x());
        nvoutdata[(i * njk + j * nk + k) * 4 + 1] = NtoUC(gradient.y());
        nvoutdata[(i * njk + j * nk + k) * 4 + 2] = NtoUC(gradient.z());
        nvoutdata[(i * njk + j * nk + k) * 4 + 3] = VtoUC(val, dmin, dmaxplus);
        gmoutdata[i * njk + j * nk + k] = gm;
      }
    }
  }
}


void 
NrrdSetupTexture::execute()
{
  NrrdDataHandle nin_handle;
  update_state(NeedData);
  NrrdIPort *inrrd_ = (NrrdIPort *)get_iport("Value");
  if (!inrrd_) {
    error("Unable to initialize iport 'Value'.");
    return;
  }

  if (!inrrd_->get(nin_handle))
    return;

  if (!nin_handle.get_rep()) {
    error("Empty input Nrrd.");
    return;
  }

  Nrrd *nin = nin_handle->nrrd;

  if (nin->dim != 3)
  {
    error("Input nrrd must be three dimensional.");
    return;
  }

  if (last_generation_ != nin_handle->generation)
  {
    // Set default values for min,max
    NrrdRange *range = nrrdRangeNewSet(nin, nrrdBlind8BitRangeState);
    realmin_.set(range->min);
    realmax_.set(range->max);
    delete range;
    minf_.reset();
    maxf_.reset();
    useinputmin_.reset();
    useinputmax_.reset();
  }

  const double minf = useinputmin_.get()?realmin_.get():minf_.get();
  const double maxf = useinputmax_.get()?realmax_.get():maxf_.get();

#if 0  
  if (last_generation_ == nin_handle->generation &&
      last_minf_ == minf &&
      last_maxf_ == maxf &&
      last_nrrdH_.get_rep())
  {
    onrrd_->send(last_nrrdH_);
    return;
  }
#endif

  last_generation_ = nin_handle->generation;
  last_minf_ = minf;
  last_maxf_ = maxf;

  Nrrd *nvout = nrrdNew();
  Nrrd *gmout = nrrdNew();
  int nvsize[NRRD_DIM_MAX];
  int gmsize[NRRD_DIM_MAX];
  int dim;

  // Create a local array of axis sizes, so we can allocate the output Nrrd
  for (dim=0; dim<nin->dim; dim++)
  {
    gmsize[dim] = nin->axis[dim].size;
    nvsize[dim+1]=nin->axis[dim].size;
  }
  nvsize[0] = nin->dim + 1;

  // Allocate the nrrd's data, set the size of each axis
  nrrdAlloc_nva(nvout, nrrdTypeUChar, nin->dim+1, nvsize);

  // Allocate the nrrd's data, set the size of each axis
  nrrdAlloc_nva(gmout, nrrdTypeFloat, nin->dim, gmsize);

  // Set axis info for (new) axis 0
  nvout->axis[0].kind=nrrdKind4Vector;
  nvout->axis[0].center=nin->axis[0].center;
  nvout->axis[0].spacing=AIR_NAN;
  nvout->axis[0].min=AIR_NAN;
  nvout->axis[0].max=AIR_NAN;

  // Copy the other axes' info
  for (dim=0; dim<nin->dim; dim++)
  {
    nvout->axis[dim+1].kind = nin->axis[dim].kind;
    nvout->axis[dim+1].center = nin->axis[dim].center;
    nvout->axis[dim+1].spacing = nin->axis[dim].spacing;
    nvout->axis[dim+1].min = nin->axis[dim].min;
    nvout->axis[dim+1].max = nin->axis[dim].max;

    gmout->axis[dim].kind = nin->axis[dim].kind;
    gmout->axis[dim].center = nin->axis[dim].center;
    gmout->axis[dim].spacing = nin->axis[dim].spacing;
    gmout->axis[dim].min = nin->axis[dim].min;
    gmout->axis[dim].max = nin->axis[dim].max;
  }

  // Build the transform here.
  Transform transform;
  string trans_str;
  // See if it's stored in the nrrd first.
  if (nin_handle->get_property("Transform", trans_str) &&
      trans_str != "Unknown")
  {
    double t[16];
    int old_index=0, new_index=0;
    for(int i=0; i<16; i++)
    {
      new_index = trans_str.find(" ", old_index);
      string temp = trans_str.substr(old_index, new_index-old_index);
      old_index = new_index+1;
      string_to_double(temp, t[i]);
    }
    transform.set(t);
  } 
  else
  {
    // Reconstruct the axis aligned transform.
    const Point nmin(nin->axis[0].min, nin->axis[1].min, nin->axis[2].min);
    const Point nmax(nin->axis[0].max, nin->axis[1].max, nin->axis[2].max);
    transform.pre_scale(nmax - nmin);
    transform.pre_translate(nmin.asVector());
  }

  if (nin->type == nrrdTypeChar)
  {
    compute_data((char *)nin->data,
                 (unsigned char *)nvout->data, (float *)gmout->data,
                 nin->axis[2].size, nin->axis[1].size, nin->axis[0].size,
                 transform, minf, maxf);
  }
  else if (nin->type == nrrdTypeUChar)
  {
    compute_data((unsigned char *)nin->data,
                 (unsigned char *)nvout->data, (float *)gmout->data,
                 nin->axis[2].size, nin->axis[1].size, nin->axis[0].size,
                 transform, minf, maxf);
  }
  else if (nin->type == nrrdTypeShort)
  {
    compute_data((short *)nin->data,
                 (unsigned char *)nvout->data, (float *)gmout->data,
                 nin->axis[2].size, nin->axis[1].size, nin->axis[0].size,
                 transform, minf, maxf);
  }
  else if (nin->type == nrrdTypeUShort)
  {
    compute_data((unsigned short *)nin->data,
                 (unsigned char *)nvout->data, (float *)gmout->data,
                 nin->axis[2].size, nin->axis[1].size, nin->axis[0].size,
                 transform, minf, maxf);
  }
  else if (nin->type == nrrdTypeInt)
  {
    compute_data((int *)nin->data,
                 (unsigned char *)nvout->data, (float *)gmout->data,
                 nin->axis[2].size, nin->axis[1].size, nin->axis[0].size,
                 transform, minf, maxf);
  }
  else if (nin->type == nrrdTypeUInt)
  {
    compute_data((unsigned int *)nin->data,
                 (unsigned char *)nvout->data, (float *)gmout->data,
                 nin->axis[2].size, nin->axis[1].size, nin->axis[0].size,
                 transform, minf, maxf);
  }
  else if (nin->type == nrrdTypeFloat)
  {
    compute_data((float *)nin->data,
                 (unsigned char *)nvout->data, (float *)gmout->data,
                 nin->axis[2].size, nin->axis[1].size, nin->axis[0].size,
                 transform, minf, maxf);
  }
  else if (nin->type == nrrdTypeDouble)
  {
    compute_data((double *)nin->data,
                 (unsigned char *)nvout->data, (float *)gmout->data,
                 nin->axis[2].size, nin->axis[1].size, nin->axis[0].size,
                 transform, minf, maxf);
  }
  else
  {
    error("Unsupported input type.");
    return;
  }
  
  // Create SCIRun data structure wrapped around nout
  NrrdData *nvnd = scinew NrrdData;
  nvnd->nrrd = nvout;
  NrrdDataHandle nvout_handle(nvnd);

  // Copy the properties
  nvout_handle->copy_properties(nin_handle.get_rep());

  // Create SCIRun data structure wrapped around nout
  NrrdData *gmnd = scinew NrrdData;
  gmnd->nrrd = gmout;
  NrrdDataHandle gmout_handle(gmnd);

  // Copy the properties
  gmout_handle->copy_properties(nin_handle.get_rep());

  NrrdOPort *onvnrrd = (NrrdOPort *)get_oport("Normal/Value");
  if (!onvnrrd) {
    error("Unable to initialize oport 'Normal/Value'.");
    return;
  }
  onvnrrd->send(nvout_handle);

  NrrdOPort *ogmnrrd = (NrrdOPort *)get_oport("Gradient Magnitude");
  if (!ogmnrrd) {
    error("Unable to initialize oport 'Gradient Magnitude'.");
    return;
  }
  ogmnrrd->send(gmout_handle);
}


} // End namespace SCITeem
