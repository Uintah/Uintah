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
  virtual void widget_moved(bool last);    
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
  GLVolumeRenderer *volren;



  void SwapXZ( /*FieldHandle sfh*/ );
};

} // End namespace SCIRun

#endif
