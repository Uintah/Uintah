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


#ifndef TEXTUREVOLVIS_H
#define TEXTUREVOLVIS_H
/*
 * TextureVolVis.cc
 *
 * Simple interface to volume rendering stuff
 */

#include <Dataflow/Network/Module.h>
#include <Core/Geom/ColorMap.h>
#include <Dataflow/Ports/ColorMapPort.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Thread/CrowdMonitor.h>
#include <Dataflow/Widgets/PointWidget.h>

#include <Core/GLVolumeRenderer/GLVolumeRenderer.h>
#include <Dataflow/Ports/GLTexture3DPort.h>


namespace SCIRun {

class GeomObj;


class TextureVolVis : public Module {

public:
  TextureVolVis(GuiContext* ctx);

  virtual ~TextureVolVis();
  virtual void widget_moved(bool last, BaseWidget*);
  virtual void execute();
  //  void tcl_command( TCLArgs&, void* );

private:
  
  GLTexture3DHandle tex;

  ColorMapIPort* icmap;
  GLTexture3DIPort* intexture;
  GeometryOPort* ogeom;
  ColorMapOPort* ocmap;
   
  CrowdMonitor control_lock; 
  PointWidget *control_widget;
  GeomID control_id;


  int cmap_id;  // id associated with color map...
  

  GuiInt num_slices;
  GuiInt draw_mode;
  GuiInt render_style;
  GuiDouble alpha_scale;
  GuiInt interp_mode;
  
  CrowdMonitor geom_lock_;
  GLVolumeRenderer *volren;



  void SwapXZ( /*FieldHandle sfh*/ );
};

} // End namespace SCIRun

#endif
