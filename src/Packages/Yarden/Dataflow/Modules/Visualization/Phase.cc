/*
 *  Phase.cc   View Depended Iso Surface Extraction
 *             for Structures Grids (Bricks)
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Oct 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#define DEF_CLOCK

#include <stdio.h>

#include <SCICore/Containers/String.h>
#include <SCICore/Util/Timer.h>
#include <PSECore/Dataflow/Module.h>
#include <SCICore/Datatypes/ScalarFieldRGdouble.h>
#include <SCICore/Datatypes/ScalarFieldRGfloat.h>
#include <SCICore/Datatypes/ScalarFieldRGshort.h>
#include <PSECore/Datatypes/ScalarFieldPort.h>
#include <PSECore/Datatypes/ColorMapPort.h>
#include <PSECore/Datatypes/GeometryPort.h>
#include <PSECore/Datatypes/SurfacePort.h>

#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glx.h>
#include <SCICore/Geom/Color.h>
#include <SCICore/Geom/GeomOpenGL.h>

#include <SCICore/TclInterface/TCLTask.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <SCICore/TclInterface/TCL.h>
#include <tcl.h>
#include <tk.h>

#include <SCICore/Geom/Material.h>
#include <SCICore/Geom/GeomTriangles.h>
#include <SCICore/Geom/View.h>
#include <SCICore/Geom/GeomGroup.h>
#include <SCICore/Geom/GeomObj.h>
#include <SCICore/Geom/GeomTri.h>
#include <SCICore/Geom/GeomLine.h>
#include <SCICore/Geom/GeomBox.h>
#include <SCICore/Geom/Pt.h>
#include <SCICore/Geom/GeomTransform.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/Transform.h>
#include <SCICore/Geom/BBoxCache.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Math/Trig.h>

#include <math.h>
#include <iostream>
#include <sstream>
#include <values.h>

#include <Yarden/Modules/Visualization/Screen.h>
#include <Yarden/Datatypes/General/Clock.h>
#include <Yarden/Modules/Visualization/mcube_scan.h>

namespace Yarden {
namespace Modules {
  
using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::Containers;
using namespace SCICore::GeomSpace;
using namespace SCICore::Geometry;
using namespace SCICore::Math;
using namespace SCICore::TclInterface;
using namespace Yarden::Datatypes;


//#define DOUBLE
//#define VIS_WOMAN
#define FLOAT
  
#ifdef VIS_WOMAN
#define SHORT
#endif
  
#ifdef CHAR
  typedef unsigned char Value;
  typedef ScalarFieldRGchar FIELD_TYPE ;
#define GET_FIELD(f) (f->getRGBase()->getRGChar())
#endif
  
#ifdef FLOAT
  typedef float Value;
  typedef ScalarFieldRGfloat FIELD_TYPE ;
#define GET_FIELD(f) (f->getRGBase()->getRGFloat())
#endif
  
#ifdef SHORT
  typedef short Value;
  typedef ScalarFieldRGshort FIELD_TYPE ;
#define GET_FIELD(f) (f->getRGBase()->getRGShort())
#endif
  
#ifdef DOUBLE
  typedef double Value;
  typedef ScalarFieldRGdouble FIELD_TYPE ;
#define GET_FIELD(f) (f->getRGBase()->getRGDouble())
#endif
  
