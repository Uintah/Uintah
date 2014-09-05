/*
 *  Phase.cc   View Depended Iso Surface Extraction
 *             for Structures Grids (Bricks)
 *  Written by:
 *   Packages/Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Oct 1999
 *
 *  Copyright (C) 1999 SCI Group
 */

#define DEF_CLOCK

#include <stdio.h>

#include <Core/Containers/String.h>
#include <Core/Util/Timer.h>
#include <Dataflow/Network/Module.h>
#include <Core/Datatypes/ScalarFieldRG.h>
#include <Dataflow/Ports/ScalarFieldPort.h>
#include <Dataflow/Ports/ColorMapPort.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/SurfacePort.h>

#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glx.h>
#include <Core/Datatypes/Color.h>
#include <Core/Geom/GeomOpenGL.h>

#include <Core/GuiInterface/TCLTask.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/GuiInterface/TCL.h>
#include <tcl.h>
#include <tk.h>

#include <Core/Geom/Material.h>
#include <Core/Geom/GeomTriangles.h>
#include <Core/Geom/View.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/GeomObj.h>
#include <Core/Geom/GeomTri.h>
#include <Core/Geom/GeomLine.h>
#include <Core/Geom/GeomBox.h>
#include <Core/Geom/Pt.h>
#include <Core/Geom/GeomTransform.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Transform.h>
#include <Core/Geom/BBoxCache.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Geometry/Vector.h>
#include <Core/Math/Trig.h>

#include <math.h>
#include <iostream>
#include <sstream>
#include <values.h>

#include <Packages/Yarden/Core/Algorithms/Visualization/mcube_scan.h>
#include <Packages/Yarden/Core/Datatypes/Screen.h>
#include <Packages/Yarden/Core/Datatypes/Clock.h>

namespace Yarden {
using namespace SCIRun;

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

}
