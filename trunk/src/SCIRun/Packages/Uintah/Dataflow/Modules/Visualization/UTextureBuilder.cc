//
//  For more information, please see: http://software.sci.utah.edu
//
//  The MIT License
//
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
//  University of Utah.
//
//  License for the specific language governing rights and limitations under
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//
//    File   : TextureBuilder.cc
//    Author : Milan Ikits
//    Date   : Fri Jul 16 00:11:18 2004

#include <Dataflow/Modules/Visualization/ConvertNrrdsToTexture.h>
#include <Dataflow/Network/Ports/ColorMapPort.h>


namespace Uintah {
using namespace SCIRun;

class UTextureBuilder : public ConvertNrrdsToTexture
{
  double last_minf_;
  double last_maxf_;
  int last_nbits_;
  int last_generation_;
  NrrdDataHandle last_nrrdH_;
public:
  UTextureBuilder(GuiContext*);
  virtual ~UTextureBuilder();

  virtual void execute();

};
} // namespace Uintah

using namespace Uintah;

using SCIRun::ColorMapIPort;
using SCIRun::ColorMapOPort;
using SCIRun::ColorMapHandle;
using SCIRun::ConvertNrrdsToTexture;
using SCIRun::Module;


DECLARE_MAKER(UTextureBuilder)

UTextureBuilder::UTextureBuilder(GuiContext* ctx)
  : ConvertNrrdsToTexture(ctx, "UTextureBuilder", Source, "Visualization", "Uintah"),
    last_minf_(1), last_maxf_(-1), last_generation_(-1), last_nbits_(0), last_nrrdH_(0)
{}

UTextureBuilder::~UTextureBuilder()
{}


void
UTextureBuilder::execute()
{

    // Get a handle to the ColorMap port.
  ColorMapIPort* cmap_iport = ( ColorMapIPort *) get_iport("ColorMap");
  ColorMapHandle cmap_h;

  if( !cmap_iport->get( cmap_h ) || !(cmap_h.get_rep()) ) {
    ConvertNrrdsToTexture::execute();
    return;
  }

  bool nothing_changed = false;

  NrrdDataHandle nrrdH;
  if (!get_input_handle("Nrrd", nrrdH)) {
    send_output_handle("ColorMap", cmap_h, true);
    return;
  }

  double minf = cmap_h->getMin();
  double maxf = cmap_h->getMax();
  const int nbits = 8;

  if (last_generation_ == nrrdH->generation &&
      last_minf_ == minf &&
      last_maxf_ == maxf &&
      last_nbits_ == nbits &&
      last_nrrdH_.get_rep())
  {
    // nothing changed...
    // TODO - send texture: send_output_handle("Nrrd", last_nrrdH_, true);
    send_output_handle("ColorMap", cmap_h, true);
    return;
  }

  // quantize the input nrrd

  // must detach because we are about to modify the input nrrd.
  nrrdH.detach(); 

  Nrrd *nin = nrrdH->nrrd_;

  remark("Quantizing -- min=" + to_string(minf) +
         " max=" + to_string(maxf) + " nbits=" + to_string(nbits));
  NrrdRange *range = nrrdRangeNew(minf, maxf);
  NrrdData *nrrd = scinew NrrdData;
  if (nrrdQuantize(nrrd->nrrd_, nin, range, nbits))
  {
    char *err = biffGetDone(NRRD);
    error(string("Trouble quantizing: ") + err);
    free(err);
    return;
  }

  nrrdKeyValueCopy(nrrd->nrrd_, nin);
  // set current state for next execution
  last_generation_ = nrrdH->generation;
  last_minf_ = minf;
  last_maxf_ = maxf;
  last_nbits_ = nbits;
  last_nrrdH_ = nrrdH;

  send_output_handle("Nrrd", last_nrrdH_, true);
  /*
  if( cmap_h->IsScaled() ){
    gui_fixed_.set( 1 );
    gui_vminval_.set(cmap_h->getMin() );
    gui_vmaxval_.set(cmap_h->getMax() );
  }
  */

  // Get a handle to the output ColorMap port.
   ColorMapOPort* cmap_oport = ( ColorMapOPort *) get_oport("ColorMap");
  

  if (!cmap_oport) {
     error("Unable to initialize oport 'ColorMap'.");
      return;
    }

  ConvertNrrdsToTexture::execute();
  cmap_oport->send(cmap_h);
}
