//static char *id="@(#) $Id$";

/*
 *  CuttingPlane.cc:  
 *
 *  Written by:
 *   Colette Mullenhoff
 *   Department of Computer Science
 *   University of Utah
 *   May 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#include <SCICore/Containers/Array1.h>
#include <SCICore/Util/NotFinished.h>
#include <PSECore/Dataflow/Module.h>
#include <PSECore/CommonDatatypes/GeometryPort.h>
#include <PSECore/CommonDatatypes/ScalarFieldPort.h>
#include <SCICore/CoreDatatypes/ScalarFieldRG.h>
#include <PSECore/CommonDatatypes/ColorMapPort.h>
#include <SCICore/Geom/TimeGrid.h>
#include <SCICore/Geom/GeomGroup.h>
#include <SCICore/Geom/GeomLine.h>
#include <SCICore/Geom/Material.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Math/MinMax.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/Multitask/Task.h>
#include <SCICore/Multitask/ITC.h>
#include <SCICore/TclInterface/TCLvar.h>

#include <PSECore/Widgets/ScaledFrameWidget.h>
#include <iostream.h>

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::CommonDatatypes;
using namespace PSECore::Widgets;
using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;
using namespace SCICore::Geometry;
using namespace SCICore::Math;

class CuttingPlaneTex : public Module {
  ScalarFieldIPort *inscalarfield;
  ColorMapIPort *incolormap;
  GeometryOPort* ogeom;
  CrowdMonitor widget_lock;
  int init;
  int widget_id;
  ScaledFrameWidget *widget;
  virtual void widget_moved(int last);
  TCLint cutting_plane_type;
  TCLint num_contours;   
  TCLdouble offset;
  TCLdouble scale;
  MaterialHandle outcolor;
  int grid_id;
  int need_find;

  ScalarFieldRG *ingrid; // this only works on regular grids for chaining

  int u_num, v_num;
  Point corner;
  Vector u, v;
  ScalarField* sfield;
  ColorMap* cmap;
  TimeGrid* grid;
  Array1<double> times;
public:
  CuttingPlaneTex(const clString& id);
  CuttingPlaneTex(const CuttingPlaneTex&, int deep);
  virtual ~CuttingPlaneTex();
  virtual Module* clone(int deep);
  virtual void execute();

  virtual void tcl_command(TCLArgs&, void*);
};

Module* make_CuttingPlaneTex(const clString& id) {
  return new CuttingPlaneTex(id);
}

//static clString module_name("CuttingPlaneTex");
static clString widget_name("CuttingPlaneTex Widget");

CuttingPlaneTex::CuttingPlaneTex(const clString& id)
: Module("CuttingPlaneTex", id, Filter), 
  cutting_plane_type("cutting_plane_type",id, this),
  scale("scale", id, this), offset("offset", id, this),
  num_contours("num_contours", id, this)
{
  // Create the input ports
  // Need a scalar field and a colormap
  inscalarfield = scinew ScalarFieldIPort( this, "Scalar Field",
					  ScalarFieldIPort::Atomic);
  add_iport( inscalarfield);
  incolormap = scinew ColorMapIPort( this, "ColorMap",
				    ColorMapIPort::Atomic);
  add_iport( incolormap);
  
  // Create the output port
  ogeom = scinew GeometryOPort(this, "Geometry", 
			       GeometryIPort::Atomic);
  add_oport(ogeom);
  init = 1;
  float INIT(.1);

  widget = scinew ScaledFrameWidget(this, &widget_lock, INIT);
  grid_id=0;

  need_find=3;
  
  outcolor=scinew Material(Color(0,0,0), Color(0,0,0), Color(0,0,0), 0);
}

CuttingPlaneTex::CuttingPlaneTex(const CuttingPlaneTex& copy, int deep)
: Module(copy, deep), cutting_plane_type("cutting_plane_type",id, this),
  scale("scale", id, this), offset("offset", id, this),
  num_contours("num_contours", id, this)
{
  NOT_FINISHED("CuttingPlaneTex::CuttingPlaneTex");
}

CuttingPlaneTex::~CuttingPlaneTex()
{
}

Module* CuttingPlaneTex::clone(int deep)
{
  return scinew CuttingPlaneTex(*this, deep);
}

void CuttingPlaneTex::execute()
{
  int old_grid_id = grid_id;

  // get the scalar field and colormap...if you can
  ScalarFieldHandle sfieldh;
  if (!inscalarfield->get( sfieldh ))
    return;
  sfield=sfieldh.get_rep();

  if (!sfield->getRG())
    return;

  ingrid = sfield->getRG();

  ColorMapHandle cmaph;
  if (!incolormap->get( cmaph ))
    return;
  cmap=cmaph.get_rep();

  WallClockTimer timer;
  timer.start();
  if (init == 1) 
    {
      init = 0;
      GeomObj *w = widget->GetWidget() ;
      widget_id = ogeom->addObj( w, widget_name, &widget_lock );
      widget->Connect( ogeom );
      widget->SetRatioR( 0.4 );
      widget->SetRatioD( 0.4 );
    }
  if (need_find != 0)
    {
      Point min, max;
      sfield->get_bounds( min, max );
      Point center = min + (max-min)/2.0;
      double max_scale;
      if (need_find == 1)
	{   // Find the field and put in optimal place
	  // in xy plane with reasonable frame thickness
	  Point right( max.x(), center.y(), center.z());
	  Point down( center.x(), min.y(), center.z());
	  widget->SetPosition( center, right, down);
	  max_scale = Max( (max.x() - min.x()), (max.y() - min.y()) );
	}
      else if (need_find == 2)
	{   // Find the field and put in optimal place
	  // in yz plane with reasonable frame thickness
	  Point right( center.x(), center.y(), max.z());
	  Point down( center.x(), min.y(), center.z());	    
	  widget->SetPosition( center, right, down);
	  max_scale = Max( (max.z() - min.z()), (max.y() - min.y()) );
	}
      else
	{   // Find the field and put in optimal place
	  // in xz plane with reasonable frame thickness
	  Point right( max.x(), center.y(), center.z());
	  Point down( center.x(), center.y(), min.z());	    
	  widget->SetPosition( center, right, down);
	  max_scale = Max( (max.x() - min.x()), (max.z() - min.z()) );
	}
      widget->SetScale( max_scale/30. );
      need_find = 0;
    }

  // get the position of the frame widget
  Point 	center, R, D;
  widget->GetPosition( center, R, D);
  Vector v1 = R - center,
  v2 = D - center;
  
  // calculate the corner and the
  // u and v vectors of the cutting plane
  corner = (center - v1) - v2;
  u = v1 * 2.0;
  v = v2 * 2.0;

  //int cptype = cutting_plane_type.get();
  
  // create the grid for the cutting plane
  double u_fac = widget->GetRatioR(),
  v_fac = widget->GetRatioD(),
      //scale_fac = scale.get(),
      //offset_fac = offset.get();
  
  u_num = (int) (u_fac * 500);
  v_num = (int) (v_fac * 500);

  // Get the scalar values and corresponding
  // colors to put in the cutting plane

  //Vector unorm=u.normal();
  //Vector vnorm=v.normal();
  //Vector N(Cross(unorm, vnorm));

  cerr << u_num << " " << v_num << " Grid start...\n";

  // now find out how many of these frames you have...

  int num_fields=0;

  ScalarFieldRG *tmp_grid = ingrid;

  while(tmp_grid) {
    tmp_grid = (ScalarFieldRG*)tmp_grid->next;
    num_fields++;
  }
  
  cerr << num_fields << " num grids\n";
  grid = scinew TimeGrid( num_fields,u_num, v_num, corner, u, v);

  grid->map = cmap;

  tmp_grid = ingrid;

  for(int whichf=0;whichf < num_fields;whichf++) {

    grid->set_active(whichf,whichf/(num_fields*1.0));
    
    ScalarFieldRG *me = tmp_grid;

    int ix = 0;
    for (int i = 0; i < u_num; i++) {
      for (int j = 0; j < v_num; j++) {
	Point p = corner + u * ((double) i/(u_num-1)) + 
	  v * ((double) j/(v_num-1));
	double sval;
	
	// get the color from cmap for p 	    
	MaterialHandle matl;
#if 0
	double alpha;
#endif

	if (me->interpolate( p, sval, ix) || (ix=0) || me->interpolate( p, sval, ix)) {
//	  matl = cmap->lookup( sval);
#if 0
	  alpha = 0.8;
#endif
	} else {
	  matl = outcolor;
	  sval = 0;
#if 0
	  alpha=0.0;
#endif
	}
	
	grid->set(i, j, matl,sval);
//	grid->set(i, j, matl,alpha);
      }
    }
    tmp_grid = (ScalarFieldRG*)tmp_grid->next;
  }



  // delete the old grid/cutting plane
  if (old_grid_id != 0)
    ogeom->delObj( old_grid_id );
  
  grid_id = ogeom->addObj(grid, "Cutting Plane");

  timer.stop();	
  cerr << "Cutting plane took: " << timer.time() << " seconds\n";
}

void CuttingPlaneTex::widget_moved(int last)
{
  if(last && !abort_flag)
    {
      abort_flag=1;
      want_to_execute();
    }
}


void CuttingPlaneTex::tcl_command(TCLArgs& args, void* userdata)
{
  if(args.count() < 2)
    {
      args.error("Streamline needs a minor command");
      return;
    }
  if(args[1] == "findxy")
    {
      need_find=1;
      want_to_execute();
    }
  else if(args[1] == "findyz")
    {
      need_find=2;
      want_to_execute();
    }
  else if(args[1] == "findxz")
    {
      need_find=3;
      want_to_execute();
    }
  else
    {
      Module::tcl_command(args, userdata);
    }
}

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.2  1999/08/17 06:37:47  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:58:11  mcq
// Initial commit
//
// Revision 1.2  1999/04/27 22:57:57  dav
// updates in Modules for CoreDatatypes
//
// Revision 1.1.1.1  1999/04/24 23:12:33  dav
// Import sources
//
//
