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


#ifndef TEXCUTTINGPLANES_H
#define TEXCUTTINGPLANES_H

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

class TexCuttingPlanes : public Module {

public:
  TexCuttingPlanes( GuiContext* ctx);

  virtual ~TexCuttingPlanes();
  virtual void widget_moved(bool last, BaseWidget*);
  virtual void execute();
  void tcl_command( GuiArgs&, void* );

private:
  GLTexture3DHandle       tex_;
  ColorMapIPort          *icmap_;
  GLTexture3DIPort       *intexture_;
  ColorMapOPort          *ocmap_;
  GeometryOPort          *ogeom_;
  CrowdMonitor            control_lock_; 
  PointWidget            *control_widget_;
  GeomID                  control_id_;
  GuiInt                  control_pos_saved_;
  GuiDouble               control_x_;
  GuiDouble               control_y_;
  GuiDouble               control_z_;
  GuiInt                  drawX_;
  GuiInt                  drawY_;
  GuiInt                  drawZ_;
  GuiInt                  drawView_;
  GuiInt                  interp_mode_;
  GuiInt                  draw_phi0_;
  GuiInt                  draw_phi1_;
  GuiDouble		  phi0_;
  GuiDouble		  phi1_;
  GuiInt                  cyl_active_;
  GLTexture3DHandle       old_tex_;
  ColorMapHandle          old_cmap_;
  Point                   old_min_, old_max_;
  CrowdMonitor            geom_lock_;
  GLVolumeRenderer       *volren_;
  Point                   dmin_;
  Vector                  ddx_;
  Vector                  ddy_;
  Vector                  ddz_;
  double                  ddview_;
};


} // End namespace SCIRun

#endif
