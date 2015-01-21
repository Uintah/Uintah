#ifndef ANIMATEDSTREAMS_H
#define ANIMATEDSTREAMS_H
/*
 * AnimatedStreams.cc
 *
 * Simple interface to volume rendering stuff
 */

#include <SCIRun/Core/Containers/Array1.h>
#include <SCIRun/Core/Datatypes/Field.h>
#include <SCIRun/Core/Thread/Mutex.h>
#include <SCIRun/Core/Geom/ColorMap.h>
#include <Core/Malloc/Allocator.h>
#include <SCIRun/Dataflow/GuiInterface/GuiVar.h>
#include <SCIRun/Core/Thread/CrowdMonitor.h>
#include <SCIRun/Dataflow/Network/Module.h>
#include <SCIRun/Dataflow/Network/Ports/ColorMapPort.h>
#include <SCIRun/Dataflow/Network/Ports/GeometryPort.h>
#include <SCIRun/Dataflow/Network/Ports/FieldPort.h>
#include <SCIRun/Dataflow/Widgets/PointWidget.h>

#include <Core/Datatypes/GLAnimatedStreams.h>
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
