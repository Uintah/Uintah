#ifndef ANIMATEDSTREAMS_H
#define ANIMATEDSTREAMS_H
/*
 * AnimatedStreams.cc
 *
 * Simple interface to volume rendering stuff
 */

#include <Core/Containers/Array1.h>
#include <Core/Thread/Mutex.h>
#include <Dataflow/Network/Module.h>
#include <Core/Geom/ColorMap.h>
#include <Dataflow/Ports/ColorMapPort.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/Field.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Thread/CrowdMonitor.h>
#include <Dataflow/Widgets/PointWidget.h>

#include <Packages/Uintah/Core/Datatypes/GLAnimatedStreams.h>
#include <sgi_stl_warnings_off.h>
#include <string>
#include <sgi_stl_warnings_on.h>

namespace Uintah {
using namespace SCIRun;


class AnimatedStreams : public Module {

public:
  AnimatedStreams(GuiContext* ctx);

  virtual ~AnimatedStreams();
  virtual void widget_moved(bool last);    
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
