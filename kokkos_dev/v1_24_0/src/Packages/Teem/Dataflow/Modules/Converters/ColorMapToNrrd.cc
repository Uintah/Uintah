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


/*
 *  ColorMapToNrrd.cc: Converts a SCIRun ColorMap to Nrrd(s).  
 *
 *  Written by:
 *   Darby Van Uitert
 *   April 2004
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

#include <Dataflow/share/share.h>

#include <Dataflow/Ports/NrrdPort.h>
#include <Dataflow/Ports/ColorMapPort.h>


namespace SCITeem {

using namespace SCIRun;

class PSECORESHARE ColorMapToNrrd : public Module {
public:

  ColorMapIPort* icmap_;
  NrrdOPort*   nout_;
  int          colormap_generation_;

  ColorMapToNrrd(GuiContext*);

  virtual ~ColorMapToNrrd();

  virtual void execute();
};


DECLARE_MAKER(ColorMapToNrrd)
ColorMapToNrrd::ColorMapToNrrd(GuiContext* ctx)
  : Module("ColorMapToNrrd", ctx, Source, "Converters", "Teem"),
    icmap_(0), nout_(0),
    colormap_generation_(-1)
{
}


ColorMapToNrrd::~ColorMapToNrrd()
{
}


void
ColorMapToNrrd::execute()
{
  // Get ports
  icmap_ = (ColorMapIPort *)get_iport("ColorMap");
  nout_ = (NrrdOPort *)get_oport("Output");

  ColorMapHandle cmapH;
  if (!icmap_->get(cmapH)) {
    return;
  }

  if (colormap_generation_ != cmapH->generation)
  {
    colormap_generation_ = cmapH->generation;

    const unsigned int size = cmapH->resolution();
  
    NrrdData *nd = scinew NrrdData();
    nrrdAlloc(nd->nrrd, nrrdTypeFloat, 2, 4, size);
    nd->nrrd->axis[0].kind = nrrdKind4Color;
    nd->nrrd->axis[0].label = airStrdup("Colors");
    nd->nrrd->axis[0].center = nrrdCenterNode;
    nd->nrrd->axis[0].spacing = AIR_NAN;
    nd->nrrd->axis[0].min = 0.0;
    nd->nrrd->axis[0].max = 1.0;
    nd->nrrd->axis[1].kind = nrrdKindDomain;
    nd->nrrd->axis[1].label = airStrdup("Data Value");
    nd->nrrd->axis[1].center = nrrdCenterUnknown;
    nd->nrrd->axis[1].spacing = AIR_NAN;
    nd->nrrd->axis[1].min = AIR_NAN;
    nd->nrrd->axis[1].max = AIR_NAN;

    float *val = (float *)nd->nrrd->data;
    const float *data = cmapH->get_rgba();

    const int range = size*4;
    for(unsigned int start=0; start<size; start++) 
      for(int cur=0; cur<range; cur+=size, ++data) 
	val[start+cur] = *data;

    // Send the data nrrd.
    nd->nrrd->axis[0].label = airStrdup("Colors");
    NrrdDataHandle dataH(nd);
    nout_->send(dataH);
  }
}


} // End namespace Teem


