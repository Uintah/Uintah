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
#include <Core/Datatypes/ColorMap.h>
#include <Dataflow/Ports/ColorMapPort.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/Field.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Thread/CrowdMonitor.h>
#include <Dataflow/Widgets/PointWidget.h>

#include <Packages/Uintah/Core/Datatypes/GLAnimatedStreams.h>
#include <string>

namespace Uintah {


using namespace SCIRun;


class AnimatedStreams : public Module {

public:
  AnimatedStreams( const string& id);

  virtual ~AnimatedStreams();
  virtual void widget_moved(int last);    
  virtual void execute();
  void tcl_command( TCLArgs&, void* );

private:
  
  FieldHandle vf;

  ColorMapIPort* incolormap;
  FieldIPort* infield;
  GeometryOPort* ogeom;
   
  int cmap_id;  // id associated with color map...
  int generation; // the current generation of Uintah::DataArchive
  int timestep; // the current timestep
  string varname; // the current variable name
  
  GuiInt pause;
  GuiInt normals;
  GuiInt lighting;
  GuiInt normal_method;
  GuiInt iter_method;
  GuiInt linewidth;
  GuiDouble stepsize;
  GuiDouble iterations;

  GLAnimatedStreams  *anistreams;

  Mutex mutex;

  CrowdMonitor control_lock; 
  PointWidget *control_widget;
  GeomID control_id;


};

} // namespace Uintah

#endif
