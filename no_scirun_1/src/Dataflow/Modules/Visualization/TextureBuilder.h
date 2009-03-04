/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
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


#if !defined(TEXTURE_BUILDER_H)
#define TEXTURE_BUILDER_H

#include <Core/GuiInterface/GuiVar.h>
#include <Core/GuiInterface/GuiContext.h>
#include <Core/Volume/Texture.h>
#include <Dataflow/Network/Module.h>

#include <Dataflow/Modules/Visualization/share.h>

namespace SCIRun {

class SCISHARE TextureBuilder : public Module
{
public:
  TextureBuilder(GuiContext* ctx, const std::string& name="TextureBuilder",
                 SchedClass sc = Source,  const string& cat="Visualization", 
                 const string& pack="SCIRun");
  virtual ~TextureBuilder();

  virtual void execute();

protected:
  TextureHandle tHandle_;

  GuiDouble gui_vminval_;
  GuiDouble gui_vmaxval_;
  GuiDouble gui_gminval_;
  GuiDouble gui_gmaxval_;

  GuiInt gui_fixed_;
  GuiInt gui_card_mem_;
  GuiInt gui_card_mem_auto_;
  int card_mem_;

  int vfield_last_generation_;
  int gfield_last_generation_;
  double vminval_, vmaxval_;
  double gminval_, gmaxval_;
};

} // end namespace SCIRun

#endif
