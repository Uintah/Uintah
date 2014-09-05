#ifndef VOLVIS_H
#define VOLVIS_H
/*
 * VolVis.cc
 *
 * Simple interface to volume rendering stuff
 */

#include <Core/Containers/Array1.h>
#include <Dataflow/Network/Module.h>
#include <Core/Datatypes/ColorMap.h>
#include <Dataflow/Ports/ColorMapPort.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/ScalarFieldPort.h>
#include <Core/Geom/GeomTriangles.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Thread/CrowdMonitor.h>
#include <Dataflow/Widgets/PointWidget.h>

#include <Packages/Kurt/Geom/MultiBrick.h>

namespace Kurt {
using namespace SCIRun;
class GeomObj;


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
  GeomID widget_id;
  void widget_moved(int /*last*/);

  CrowdMonitor  res_lock;
  PointWidget *res;
  GeomID res_id;
  
  int field_id; // id for the scalar field...
  int cmap_id;  // id associated with color map...
  
  //  GeomTexVolRender      *rvol;  // this guy does all the work..
  
  //  GeomTrianglesP        *triangles;
    
  Point Smin,Smax;
  Vector ddv;

  GuiInt avail_tex;
  GuiDouble influence;
  GuiInt max_brick_dim;
  GuiInt num_slices;
  GuiInt draw_mode;
  GuiInt debug;
  GuiInt level;
  GuiDouble alpha;
  int mode;

  MultiBrick *brick;
  void SwapXZ( ScalarFieldHandle sfh );

};
} // End namespace Kurt


#endif
