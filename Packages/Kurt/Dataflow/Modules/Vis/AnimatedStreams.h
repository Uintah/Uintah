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
#include <Core/Datatypes/VectorField.h>
#include <Dataflow/Ports/VectorFieldPort.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>

#include <Packages/Kurt/Core/Datatypes/GLAnimatedStreams.h>

namespace Kurt {

using namespace SCIRun;

class AnimatedStreams : public Module {

public:
  AnimatedStreams( const clString& id);

  virtual ~AnimatedStreams();
  virtual void widget_moved(int last);    
  virtual void execute();
  //  void tcl_command( TCLArgs&, void* );

private:
  
  VectorFieldHandle vf;

  ColorMapIPort* incolormap;
  VectorFieldIPort* infield;
  GeometryOPort* ogeom;
   
  int cmap_id;  // id associated with color map...
  

  GuiInt pause;
  GuiInt normals;
  GuiInt linewidth;
  GuiDouble stepsize;

  GLAnimatedStreams  *anistreams;

  Mutex mutex;

  CrowdMonitor control_lock; 
  PointWidget *control_widget;
  GeomID control_id;


};

} // End namespace Kurt

#endif
