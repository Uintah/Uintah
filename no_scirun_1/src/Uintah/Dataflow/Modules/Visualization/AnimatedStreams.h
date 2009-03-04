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


#ifndef ANIMATEDSTREAMS_H
#define ANIMATEDSTREAMS_H
/*
 * AnimatedStreams.cc
 *
 * Simple interface to volume rendering stuff
 */

#include <Core/Containers/Array1.h>
#include <Core/Datatypes/Field.h>
#include <Core/Thread/Mutex.h>
#include <Core/Geom/ColorMap.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Thread/CrowdMonitor.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/Ports/ColorMapPort.h>
#include <Dataflow/Network/Ports/GeometryPort.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Dataflow/Widgets/PointWidget.h>

#include <Uintah/Core/Datatypes/GLAnimatedStreams.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

namespace Uintah {
using namespace SCIRun;


class AnimatedStreams : public Module {

public:
  AnimatedStreams(GuiContext* ctx);

  virtual ~AnimatedStreams();
  virtual void widget_moved(bool last, BaseWidget*);
  virtual void execute();
  void tcl_command( GuiArgs&, void* );

private:
  
  FieldHandle vf;
  GLAnimatedStreams  *anistreams;

  ColorMapIPort* incolormap;
  FieldIPort* infield;
  GeometryOPort* ogeom;
   
  int cmap_id;  // id associated with color map...
  string varname; // the current variable name
  int generation; // the current generation of Uintah::DataArchive
  int timestep; // the current timestep
  
  GuiInt pause;
  GuiInt normals;
  GuiInt lighting;
  GuiInt normal_method;
  GuiInt use_deltat;
  GuiDouble stepsize;
  GuiInt linewidth;

  CrowdMonitor control_lock; 
  PointWidget *control_widget;
  GeomID control_id;


  Mutex mutex;



};

} // namespace Uintah

#endif
