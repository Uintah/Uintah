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
//    File   : NrrdTextureBuilder.cc
//    Author : Milan Ikits
//    Date   : Fri Jul 16 03:28:21 2004

#include <sci_defs/ogl_defs.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Datatypes/LatVolField.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Malloc/Allocator.h>

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>

#include <Dataflow/Ports/NrrdPort.h>

#include <Core/Volume/VideoCardInfo.h>
#include <Dataflow/Ports/TexturePort.h>
#include <Core/Geom/ShaderProgramARB.h>
#include <Core/Algorithms/Visualization/NrrdTextureBuilderAlgo.h>

#include <sstream>
using std::ostringstream;

namespace SCIRun {

class NrrdTextureBuilder : public Module
{
public:
  NrrdTextureBuilder(GuiContext*);
  virtual ~NrrdTextureBuilder();

  virtual void execute();

private:
  GuiInt gui_card_mem_;
  GuiInt gui_card_mem_auto_;
  int card_mem_;
  
  int nvfield_last_generation_;
  int gmfield_last_generation_;
  
  NrrdTextureBuilderAlgo builder_;
  TextureHandle texture_;
};

DECLARE_MAKER(NrrdTextureBuilder)
  
NrrdTextureBuilder::NrrdTextureBuilder(GuiContext* ctx)
  : Module("NrrdTextureBuilder", ctx, Source, "Visualization", "SCIRun"),
    gui_card_mem_(ctx->subVar("card_mem")),
    gui_card_mem_auto_(ctx->subVar("card_mem_auto")),
    card_mem_(video_card_memory_size()),
    nvfield_last_generation_(-1), gmfield_last_generation_(-1),
    texture_(new Texture)
{}


NrrdTextureBuilder::~NrrdTextureBuilder()
{}


void
NrrdTextureBuilder::execute()
{
  if (card_mem_ != 0 && gui_card_mem_auto_.get())
  {
    gui_card_mem_.set(card_mem_);
  }
  else if (card_mem_ == 0)
  {
    gui_card_mem_auto_.set(0);
  }

  NrrdIPort* ivfield = (NrrdIPort*)get_iport("Value / Normal-Value Nrrd");
  NrrdIPort* igfield = (NrrdIPort*)get_iport("Gradmag Nrrd");
  TextureOPort* otexture = (TextureOPort *)get_oport("Texture");

  // check rep
  NrrdDataHandle nvfield;
  ivfield->get(nvfield);
  if (!nvfield.get_rep())
  {
    error("Field has no representation.");
    return;
  }

  // check type
  if (nvfield->nrrd->type != nrrdTypeUChar)
  {
    error("Normal-Value input nrrd type is not unsigned char.");
    return;
  }
  
  if (nvfield->generation != nvfield_last_generation_)
  {
    nvfield_last_generation_ = nvfield->generation;
  }

  NrrdDataHandle gmfield = 0;
  if (igfield)
  {
    igfield->get(gmfield);
    if (gmfield.get_rep()) 
    {
      if (!ShaderProgramARB::shaders_supported())
      {
        warning("This machine does not support advanced volume rendering.  The gradient magnitude will be ignored.");
        gmfield = 0;
      }
      else
      {
        if (gmfield->generation != gmfield_last_generation_)
        {
          gmfield_last_generation_ = gmfield->generation;
        }
        // check type
        if (gmfield->nrrd->type != nrrdTypeUChar)
        {
          error("Gradmag input nrrd type is not unsigned char.");
          return;
        }
      }
    }
  }

  if (builder_.build(this, texture_, nvfield, gmfield, gui_card_mem_.get()))
  {
    otexture->send(texture_);
  }
}


} // namespace SCIRun
