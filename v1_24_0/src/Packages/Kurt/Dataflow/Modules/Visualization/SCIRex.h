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

#ifndef SCIREX_H
#define SCIREX_H
/*
 * SCIRex.cc
 *
 * Simple interface to volume rendering stuff
 */

#include <Packages/Kurt/Core/Geom/SCIRexRenderer.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Ports/GLTexture3DPort.h>
#include <Dataflow/Ports/ColorMapPort.h>
#include <Dataflow/Ports/GeometryPort.h>

#include <Core/GLVolumeRenderer/GLTexture3D.h>


namespace Kurt {

using SCIRun::Module;
using SCIRun::FieldIPort;
using SCIRun::GLTexture3DIPort;
using SCIRun::ColorMapIPort;
using SCIRun::GeometryOPort;
using SCIRun::GuiInt;
using SCIRun::GuiDouble;
using SCIRun::GuiString;
using SCIRun::GeomID;

using SCIRun::GLTexture3DHandle;

class SCIRex : public Module {

public:
  SCIRex(SCIRun::GuiContext* ctx);

  virtual ~SCIRex();
  virtual void execute();
  //  void tcl_command( TCLArgs&, void* );

private:
  

  ColorMapIPort* incolormap_;
  FieldIPort* infield_;
  GeometryOPort* ogeom_;
 

  FieldHandle tex_;
  SCIRexRenderer *volren_;
  GLTexture3DHandle texH_;
   
  int cmap_id_;  // id associated with color map...
  
  GuiInt is_fixed_;
  GuiInt max_brick_dim_;
  GuiDouble min_, max_;
  GuiInt draw_mode_;
  GuiInt num_slices_;
  GuiInt render_style_;
  GuiDouble alpha_scale_;
  GuiInt interp_mode_;
  GuiInt dump_frames_;
  GuiInt use_depth_;
  GuiString displays_;
  GuiInt compositers_;
};

} // End namespace Kurt

#endif

