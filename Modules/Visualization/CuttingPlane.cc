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

#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Dataflow/ModuleList.h>
#include <Datatypes/GeometryPort.h>
#include <Datatypes/ScalarFieldPort.h>
#include <Datatypes/ColormapPort.h>
#include <Geom/Grid.h>
#include <Geometry/Point.h>
#include <TCL/TCLvar.h>

#include <Widgets/ScaledFrameWidget.h>
#include <iostream.h>

#define CP_PLANE 0
#define CP_SURFACE 1
#define CP_CONTOUR 2

class CuttingPlane : public Module {
   ScalarFieldIPort *inscalarfield;
   ColormapIPort *incolormap;
   GeometryOPort* ogeom;
   CrowdMonitor widget_lock;
   int init;
   int widget_id;
   ScaledFrameWidget *widget;
   virtual void widget_moved();
   TCLint cutting_plane_type;
   TCLdouble scale;
   TCLdouble offset;
   MaterialHandle outcolor;
   int grid_id;
   int need_find;

public:
   CuttingPlane(const clString& id);
   CuttingPlane(const CuttingPlane&, int deep);
   virtual ~CuttingPlane();
   virtual Module* clone(int deep);
   virtual void execute();

   virtual void tcl_command(TCLArgs&, void*);
};

static Module* make_CuttingPlane(const clString& id)
{
   return new CuttingPlane(id);
}

static RegisterModule db1("Fields", "CuttingPlane", make_CuttingPlane);
static RegisterModule db2("Visualization", "CuttingPlane", make_CuttingPlane);

static clString module_name("CuttingPlane");

CuttingPlane::CuttingPlane(const clString& id)
: Module("CuttingPlane", id, Filter), 
  cutting_plane_type("cutting_plane_type",id, this),
  scale("scale", id, this), offset("offset", id, this)
{
    // Create the input ports
    // Need a scalar field and a colormap
    inscalarfield = new ScalarFieldIPort( this, "Scalar Field",
					ScalarFieldIPort::Atomic);
    add_iport( inscalarfield);
    incolormap = new ColormapIPort( this, "Colormap",
				     ColormapIPort::Atomic);
    add_iport( incolormap);
					
    // Create the output port
    ogeom = new GeometryOPort(this, "Geometry", 
			      GeometryIPort::Atomic);
    add_oport(ogeom);
    init = 1;
    float INIT(.1);

    widget = new ScaledFrameWidget(this, &widget_lock, INIT);
    grid_id=0;

    need_find=1;
    
    outcolor=new Material(Color(0,0,0), Color(0,0,0), Color(0,0,0), 0);
}

CuttingPlane::CuttingPlane(const CuttingPlane& copy, int deep)
: Module(copy, deep), cutting_plane_type("cutting_plane_type",id, this),
  scale("scale", id, this), offset("offset", id, this)
{
   NOT_FINISHED("CuttingPlane::CuttingPlane");
}

CuttingPlane::~CuttingPlane()
{
}

Module* CuttingPlane::clone(int deep)
{
   return new CuttingPlane(*this, deep);
}

void CuttingPlane::execute()
{
    int old_grid_id = grid_id;

    // get the scalar field and colormap...if you can
    ScalarFieldHandle sfield;
    if (!inscalarfield->get( sfield ))
	return;
    ColormapHandle cmap;
    if (!incolormap->get( cmap ))
	return;

    if (init == 1) 
    {
	init = 0;
	GeomObj *w = widget->GetWidget() ;
	widget_id = ogeom->addObj( w, module_name, &widget_lock );
	widget->Connect( ogeom );
	widget->SetRatioR( 0.2 );
	widget->SetRatioD( 0.2 );
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
	widget->SetScale( max_scale/20. );
	need_find = 0;
    }

    // get the position of the frame widget
    Point 	corner, center, R, D;
    widget->GetPosition( center, R, D);
    Vector v1 = R - center,
           v2 = D - center;
         
    // calculate the corner and the
    // u and v vectors of the cutting plane
    corner = (center - v1) - v2;
    Vector u = v1 * 2.0,
           v = v2 * 2.0;

    int cptype = cutting_plane_type.get();
    
    // create the grid for the cutting plane
    double u_fac = widget->GetRatioR(),
           v_fac = widget->GetRatioD(),
           scale_fac = scale.get(),
           offset_fac = offset.get();

    int u_num = (int) (u_fac * 100),
        v_num = (int) (v_fac * 100);

    cout << "u fac = " << u_fac << "\nv fac = " << v_fac << endl;
    GeomGrid* grid = new GeomGrid( u_num, v_num, corner, u, v);

    // Get the scalar values and corresponding
    // colors to put in the cutting plane
    for (int i = 0; i < u_num; i++)
	for (int j = 0; j < v_num; j++)
	{
	    Point p = corner + u * ((double) i/(u_num-1)) + 
		               v * ((double) j/(v_num-1));
	    double sval;

	    // get the color from cmap for p 	    
	    MaterialHandle matl;
	    if (sfield->interpolate( p, sval))
		matl = cmap->lookup( sval);
	    else
	    {
		matl = outcolor;
		sval = 0;
	    }

	    // put the color into the cutting plane (grid) at i, j
	    if (cptype == CP_SURFACE)
	    {
		double h = sval;
		Vector normal(sfield->gradient(p));
		grid->set(i, j, ((h*scale_fac) + offset_fac), normal, matl);
	    }
	    else if (cptype == CP_PLANE)
	    {	
		grid->set(i, j, 0, matl);
	    }	
	    else
	    {
		grid->set( i, j, 0, matl);
	    }
	}
    grid_id = ogeom->addObj(grid, "Cutting Plane");
    
    // delete the old grid/cutting plane
    if (old_grid_id != 0)
	ogeom->delObj( old_grid_id );
}

void CuttingPlane::widget_moved()
{
    if(!abort_flag)
    {
	abort_flag=1;
	want_to_execute();
    }
}


void CuttingPlane::tcl_command(TCLArgs& args, void* userdata)
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
