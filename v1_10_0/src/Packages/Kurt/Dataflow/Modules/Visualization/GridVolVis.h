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

#ifndef GRIDVOLVIS_H
#define GRIDVOLVIS_H
/*
 * GridVolVis.cc
 *
 * Simple interface to volume rendering stuff
 */

#include <Packages/Kurt/Core/Geom/BrickGrid.h>
#include <Dataflow/Network/Module.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Ports/ColorMapPort.h>
#include <Dataflow/Ports/GeometryPort.h>



namespace Kurt {

using SCIRun::Module;
using SCIRun::ColorMapIPort;
using SCIRun::FieldIPort;
using SCIRun::GeometryOPort;
using SCIRun::GuiInt;
using SCIRun::GuiDouble;

class VolumeRenderer;
class GridVolRen;

class GridVolVis : public Module {

public:
  GridVolVis(SCIRun::GuiContext* ctx);

  virtual ~GridVolVis();
  virtual void execute();
  //  void tcl_command( TCLArgs&, void* );

private:
  
  FieldHandle tex;

  ColorMapIPort* incolormap;
  FieldIPort* infield;
  GeometryOPort* ogeom;
   
  int cmap_id;  // id associated with color map...
  
  GuiInt is_fixed_;
  GuiInt max_brick_dim_;
  GuiDouble min_, max_;
  GuiInt num_slices;
  GuiInt render_style;
  GuiDouble alpha_scale;
  GuiInt interp_mode;
  VolumeRenderer *volren;
  GridVolRen *gvr;




};

} // End namespace Kurt

#endif
