/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
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
  virtual void widget_moved(bool last);    
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
  GLVolumeRenderer       *volren_;
  Point                   dmin_;
  Vector                  ddx_;
  Vector                  ddy_;
  Vector                  ddz_;
  double                  ddview_;
};


} // End namespace SCIRun

#endif
