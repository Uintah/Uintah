/*
 *  Hedgehog.cc:  
 *
 *  Written by:
 *   Colette Mullenhoff
 *   Department of Computer Science
 *   University of Utah
 *   May 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#include <Classlib/Array1.h>
#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Dataflow/ModuleList.h>
#include <Datatypes/ColormapPort.h>
#include <Datatypes/GeometryPort.h>
#include <Datatypes/ScalarFieldPort.h>
#include <Datatypes/VectorFieldPort.h>
#include <Geom/Arrows.h>
#include <Geom/Group.h>
#include <Geom/Line.h>
#include <Geom/Material.h>
#include <Geometry/Point.h>
#include <Math/MinMax.h>
#include <Malloc/Allocator.h>
#include <TCL/TCLvar.h>

#include <Widgets/ScaledFrameWidget.h>
#include <iostream.h>

#define CP_PLANE 0
#define CP_SURFACE 1
#define CP_CONTOUR 2

class Hedgehog : public Module {
   VectorFieldIPort *invectorfield;
   ScalarFieldIPort* inscalarfield;
   ColormapIPort *incolormap;
   GeometryOPort* ogeom;
   CrowdMonitor widget_lock;
   int init;
   int widget_id;
   ScaledFrameWidget *widget;
   virtual void widget_moved(int last);
   TCLdouble length_scale;
   TCLdouble width_scale;
   MaterialHandle outcolor;
   int grid_id;
   int need_find;

public:
   Hedgehog(const clString& id);
   Hedgehog(const Hedgehog&, int deep);
   virtual ~Hedgehog();
   virtual Module* clone(int deep);
   virtual void execute();

   virtual void tcl_command(TCLArgs&, void*);
};

static Module* make_Hedgehog(const clString& id)
{
   return scinew Hedgehog(id);
}

static RegisterModule db1("Fields", "Hedgehog", make_Hedgehog);
static RegisterModule db2("Visualization", "Hedgehog", make_Hedgehog);

static clString module_name("Hedgehog");
static clString widget_name("Hedgehog Widget");

Hedgehog::Hedgehog(const clString& id)
: Module("Hedgehog", id, Filter), 
  length_scale("length_scale", id, this),
  width_scale("width_scale", id, this)
{
    // Create the input ports
    // Need a scalar field and a colormap
    invectorfield = scinew VectorFieldIPort( this, "Vector Field",
					     VectorFieldIPort::Atomic);
    add_iport( invectorfield);
    inscalarfield = scinew ScalarFieldIPort( this, "Scalar Field",
					ScalarFieldIPort::Atomic);
    add_iport( inscalarfield);
    incolormap = scinew ColormapIPort( this, "Colormap",
				     ColormapIPort::Atomic);
    add_iport( incolormap);
					
    // Create the output port
    ogeom = scinew GeometryOPort(this, "Geometry", 
			      GeometryIPort::Atomic);
    add_oport(ogeom);
    init = 1;
    float INIT(.1);

    widget = scinew ScaledFrameWidget(this, &widget_lock, INIT);
    grid_id=0;

    need_find=1;
    
    outcolor=scinew Material(Color(0,0,0), Color(0,0,0), Color(0,0,0), 0);
}

Hedgehog::Hedgehog(const Hedgehog& copy, int deep)
: Module(copy, deep), length_scale("length_scale", id, this),
  width_scale("width_scale", id, this)
{
   NOT_FINISHED("Hedgehog::Hedgehog");
}

Hedgehog::~Hedgehog()
{
}

Module* Hedgehog::clone(int deep)
{
   return scinew Hedgehog(*this, deep);
}

void Hedgehog::execute()
{
    int old_grid_id = grid_id;

    // get the scalar field and colormap...if you can
    VectorFieldHandle vfield;
    if (!invectorfield->get( vfield ))
	return;
    ScalarFieldHandle ssfield;
    int have_sfield=inscalarfield->get( ssfield );
    ColormapHandle cmap;
    int have_cmap=incolormap->get( cmap );
    if(!have_cmap)
	have_sfield=0;

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
	vfield->get_bounds( min, max );
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

    // create the grid for the cutting plane
    double u_fac = widget->GetRatioR(),
           v_fac = widget->GetRatioD(),
           lenscale = length_scale.get(),
           widscale = width_scale.get();

    int u_num = (int) (u_fac * 100),
        v_num = (int) (v_fac * 100);

    cout << "u fac = " << u_fac << "\nv fac = " << v_fac << endl;

    double ld=vfield->longest_dimension();
    GeomArrows* arrows = new GeomArrows(widscale*ld);
    Vector unorm=u.normal();
    Vector vnorm=v.normal();
    Vector N(Cross(unorm, vnorm));
    for (int i = 0; i < u_num; i++)
	for (int j = 0; j < v_num; j++)
	{
	    Point p = corner + u * ((double) i/(u_num-1)) + 
		v * ((double) j/(v_num-1));

	    // Query the vector field...
	    Vector v;
	    if (vfield->interpolate( p, v)){
		if(have_sfield){
		    // get the color from cmap for p 	    
		    MaterialHandle matl;
		    double sval;
		    if (ssfield->interpolate( p, sval))
			matl = cmap->lookup( sval);
		    else
		    {
			matl = outcolor;
		    }
		    arrows->add(p, v*lenscale, matl, matl, matl);
		} else {
		    arrows->add(p, v*lenscale);
		}
	    }
	}
    grid_id = ogeom->addObj(arrows, module_name);

    // delete the old grid/cutting plane
    if (old_grid_id != 0)
	ogeom->delObj( old_grid_id );
}

void Hedgehog::widget_moved(int last)
{
    if(last && !abort_flag)
    {
	abort_flag=1;
	want_to_execute();
    }
}


void Hedgehog::tcl_command(TCLArgs& args, void* userdata)
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
