#ifndef VOLVIS_H
#define VOLVIS_H
/*
 * VolVis.cc
 *
 * Simple interface to volume rendering stuff
 */

#include <SCICore/Containers/Array1.h>
#include <PSECore/Dataflow/Module.h>
#include <SCICore/Datatypes/ColorMap.h>
#include <PSECore/Datatypes/ColorMapPort.h>
#include <PSECore/Datatypes/GeometryPort.h>
#include <PSECore/Datatypes/ScalarFieldPort.h>
#include <SCICore/Geom/GeomTriangles.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <SCICore/Thread/CrowdMonitor.h>
#include <PSECore/Widgets/PointWidget.h>

#include <Kurt/Geom/MultiBrick.h>

namespace Kurt {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace PSECore::Widgets;
using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;
using namespace SCICore::Geometry;
class SCICore::GeomSpace::GeomObj;


class VolVis : public Module {

public:
  VolVis( const clString& id);

  virtual ~VolVis();
  //  virtual void widget_moved(int last);    
  virtual void execute();
  void tcl_command( TCLArgs&, void* );

private:
  ScalarFieldIPort *inscalarfield;

  ColorMapIPort* incolormap;
  
  GeometryOPort* ogeom;
   
  int init;
  CrowdMonitor widget_lock;
  PointWidget *widget;

  int widgetMoved;
  
  int field_id; // id for the scalar field...
  int cmap_id;  // id associated with color map...
  
  //  GeomTexVolRender      *rvol;  // this guy does all the work..
  
  //  GeomTrianglesP        *triangles;
    
  Point Smin,Smax;
  Vector ddv;

  TCLint avail_tex;
  TCLint max_brick_dim;
  TCLint num_slices;
  TCLint draw_mode;
  TCLdouble alpha;
  int mode;
  

};

} // namespace Modules
} // namespace Uintah

#endif
