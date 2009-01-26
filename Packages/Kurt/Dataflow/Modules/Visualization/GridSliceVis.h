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

#include <Core/Containers/Array1.h>
#include <Dataflow/Network/Module.h>
#include <Core/Geom/ColorMap.h>
#include <Dataflow/Network/Ports/ColorMapPort.h>
#include <Core/Datatypes/Field.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Dataflow/Network/Ports/GeometryPort.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Geometry/Vector.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Thread/CrowdMonitor.h>

namespace SCIRun{
  class PointWidget;
}

namespace Kurt {
using SCIRun::Module;
using SCIRun::ColorMapIPort;
using SCIRun::FieldHandle;
using SCIRun::FieldIPort;
using SCIRun::GeometryOPort;
using SCIRun::GuiInt;
using SCIRun::GuiDouble;
using SCIRun::CrowdMonitor;
using SCIRun::Vector;
using SCIRun::GeomID;
using SCIRun::PointWidget;


class SliceRenderer;
class GridSliceRen;
class GridSliceVis : public Module {

public:
  GridSliceVis(SCIRun::GuiContext *ctx);

  virtual ~GridSliceVis();
  virtual void widget_moved(int last, BaseWidget*);
  virtual void execute();
  void tcl_command( SCIRun::GuiArgs&, void* );

private:

  
  FieldHandle tex;

  ColorMapIPort* incolormap;
  FieldIPort* infield;
  GeometryOPort* ogeom;
   
   
  CrowdMonitor control_lock; 
  PointWidget *control_widget;
  GeomID control_id;


  int cmap_id;  // get_id() associated with color map...
  

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
