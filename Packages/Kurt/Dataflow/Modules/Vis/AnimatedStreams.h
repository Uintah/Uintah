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
#include <Core/TclInterface/TCLvar.h>

#include <Packages/Kurt/Core/Datatypes/GLAnimatedStreams.h>

namespace Kurt {
using namespace SCIRun;

class GeomObj;

class AnimatedStreams : public Module {

public:
  AnimatedStreams( const clString& id);

  virtual ~AnimatedStreams();
  virtual void execute();
  //  void tcl_command( TCLArgs&, void* );

private:
  
  VectorFieldHandle vf;

  ColorMapIPort* incolormap;
  VectorFieldIPort* infield;
  GeometryOPort* ogeom;
   
  int cmap_id;  // id associated with color map...
  

  TCLint pause;
  TCLint normals;
  TCLint linewidth;
  TCLdouble stepsize;

  GLAnimatedStreams  *anistreams;

  Mutex mutex;



};
} // End namespace Kurt


#endif
