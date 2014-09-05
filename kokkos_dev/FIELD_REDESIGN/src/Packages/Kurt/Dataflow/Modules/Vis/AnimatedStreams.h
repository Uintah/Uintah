#ifndef ANIMATEDSTREAMS_H
#define ANIMATEDSTREAMS_H
/*
 * AnimatedStreams.cc
 *
 * Simple interface to volume rendering stuff
 */

#include <SCICore/Containers/Array1.h>
#include <SCICore/Thread/Mutex.h>
#include <PSECore/Dataflow/Module.h>
#include <SCICore/Datatypes/ColorMap.h>
#include <PSECore/Datatypes/ColorMapPort.h>
#include <PSECore/Datatypes/GeometryPort.h>
#include <SCICore/Datatypes/VectorField.h>
#include <PSECore/Datatypes/VectorFieldPort.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLvar.h>

#include <Kurt/Datatypes/GLAnimatedStreams.h>

namespace Kurt {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;
using namespace SCICore::Geometry;
using SCICore::Thread::Mutex;

class SCICore::GeomSpace::GeomObj;

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

} // namespace Modules
} // namespace Uintah

#endif
