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

#ifndef GRIDSLICEVIS_H
#define GRIDSLICEVIS_H
/*
 * GridSliceVis.cc
 *
 * Simple interface to volume rendering stuff
 */

#include <Packages/Kurt/Core/Geom/SliceRenderer.h>
#include <Packages/Kurt/Core/Geom/GridSliceRen.h>
#include <Core/Containers/Array1.h>
#include <Dataflow/Network/Module.h>
#include <Core/Datatypes/ColorMap.h>
#include <Dataflow/Ports/ColorMapPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Geometry/Vector.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/GuiInterface/TCL.h>
#include <Core/Thread/CrowdMonitor.h>
#include <Dataflow/Widgets/PointWidget.h>

namespace Kurt {
using SCIRun::Module;
using SCIRun::ColorMapIPort;
using SCIRun::FieldIPort;
using SCIRun::GeometryOPort;
using SCIRun::GuiInt;
using SCIRun::GuiDouble;
using SCIRun::TCLArgs;
using SCIRun::CrowdMonitor;
using SCIRun::Vector;
using SCIRun::GeomID;
using SCIRun::PointWidget;
class GeomObj;



class GridSliceVis : public Module {

public:
  GridSliceVis( const string& id);

  virtual ~GridSliceVis();
  virtual void widget_moved(int last);    
  virtual void execute();
  void tcl_command( TCLArgs&, void* );

private:

  
  FieldHandle tex;

  ColorMapIPort* incolormap;
  FieldIPort* infield;
  GeometryOPort* ogeom;
   
   
  CrowdMonitor control_lock; 
  PointWidget *control_widget;
  GeomID control_id;


  int cmap_id;  // id associated with color map...
  

  GuiInt is_fixed_;
  GuiInt max_brick_dim_;
  GuiDouble min_, max_;
  GuiInt drawX;
  GuiInt drawY;
  GuiInt drawZ;
  GuiInt drawView;
  GuiInt interp_mode;
  GuiDouble point_x;
  GuiDouble point_y;
  GuiDouble point_z;
  GuiInt point_init;

  SliceRenderer* sliceren;
  Vector ddv;
  double ddview;
  GridSliceRen *svr;


};


} // End namespace SCIRun

#endif
