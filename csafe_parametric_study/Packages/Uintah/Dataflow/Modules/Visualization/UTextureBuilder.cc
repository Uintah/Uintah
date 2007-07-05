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

#include <Dataflow/Modules/Visualization/TextureBuilder.h>
#include <Dataflow/Network/Ports/ColorMapPort.h>


namespace Uintah {
using namespace SCIRun;

class UTextureBuilder : public TextureBuilder
{
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
using SCIRun::TextureBuilder;
using SCIRun::Module;


DECLARE_MAKER(UTextureBuilder)

UTextureBuilder::UTextureBuilder(GuiContext* ctx)
  : TextureBuilder(ctx, "UTextureBuilder", Source, "Visualization", "Uintah")
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
    TextureBuilder::execute();
    return;
  }

  if( cmap_h->IsScaled() ){
    gui_fixed_.set( 1 );
    gui_vminval_.set(cmap_h->getMin() );
    gui_vmaxval_.set(cmap_h->getMax() );
  }


  // Get a handle to the output ColorMap port.
   ColorMapOPort* cmap_oport = ( ColorMapOPort *) get_oport("ColorMap");
  

  if (!cmap_oport) {
     error("Unable to initialize oport 'ColorMap'.");
      return;
    }

  TextureBuilder::execute();
  cmap_oport->send(cmap_h);
}



