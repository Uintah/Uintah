/*
 *  Hedgehog.cc:  
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   June 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#include <Classlib/Array1.h>
#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Datatypes/ColorMapPort.h>
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

#include <Widgets/ScaledBoxWidget.h>
#include <Widgets/ScaledFrameWidget.h>
#include <iostream.h>

#define CP_PLANE 0
#define CP_SURFACE 1
#define CP_CONTOUR 2

class Hedgehog : public Module {
   VectorFieldIPort *invectorfield;
   ScalarFieldIPort* inscalarfield;
   ColorMapIPort *inColorMap;
   GeometryOPort* ogeom;
   CrowdMonitor widget_lock;
   int init;
   int widget_id;
   ScaledBoxWidget* widget3d;
   ScaledFrameWidget *widget2d;
   virtual void widget_moved(int last);
   TCLdouble length_scale;
   TCLdouble width_scale;
   TCLdouble head_length;
   TCLstring type;
   TCLint exhaustive_flag;
   TCLint drawcylinders;
   TCLdouble shaft_rad;
   MaterialHandle outcolor;
   int grid_id;
   int need_find2d;
   int need_find3d;

public:
   Hedgehog(const clString& id);
   Hedgehog(const Hedgehog&, int deep);
   virtual ~Hedgehog();
   virtual Module* clone(int deep);
   virtual void execute();

   virtual void tcl_command(TCLArgs&, void*);
};

extern "C" {
Module* make_Hedgehog(const clString& id)
{
   return scinew Hedgehog(id);
}
}

static clString module_name("Hedgehog");
static clString widget_name("Hedgehog Widget");

Hedgehog::Hedgehog(const clString& id)
: Module("Hedgehog", id, Filter), 
  length_scale("length_scale", id, this),
  width_scale("width_scale", id, this),
  head_length("head_length", id, this),
  type("type", id, this),
  drawcylinders("drawcylinders", id, this),
  shaft_rad("shaft_rad", id, this),
  exhaustive_flag("exhaustive_flag", id, this)
{
    // Create the input ports
    // Need a scalar field and a ColorMap
    invectorfield = scinew VectorFieldIPort( this, "Vector Field",
					     VectorFieldIPort::Atomic);
    add_iport( invectorfield);
    inscalarfield = scinew ScalarFieldIPort( this, "Scalar Field",
					ScalarFieldIPort::Atomic);
    add_iport( inscalarfield);
    inColorMap = scinew ColorMapIPort( this, "ColorMap",
				     ColorMapIPort::Atomic);
    add_iport( inColorMap);
					
    // Create the output port
    ogeom = scinew GeometryOPort(this, "Geometry", 
			      GeometryIPort::Atomic);
    add_oport(ogeom);
    init = 1;
    float INIT(.1);

    widget2d = scinew ScaledFrameWidget(this, &widget_lock, INIT);
    widget3d = scinew ScaledBoxWidget(this, &widget_lock, INIT);
    grid_id=0;

    need_find2d=1;
    need_find3d=1;

    drawcylinders.set(0);
    outcolor=scinew Material(Color(0,0,0), Color(0,0,0), Color(0,0,0), 0);
}

Hedgehog::Hedgehog(const Hedgehog& copy, int deep)
: Module(copy, deep), length_scale("length_scale", id, this),
  width_scale("width_scale", id, this),
  head_length("head_length", id, this),
  type("type", id, this),
  drawcylinders("drawcylinders", id, this),
  shaft_rad("shaft_rad", id, this),
  exhaustive_flag("exhaustive_flag", id, this)
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

    // get the scalar field and ColorMap...if you can
    VectorFieldHandle vfield;
    if (!invectorfield->get( vfield ))
	return;
    ScalarFieldHandle ssfield;
    int have_sfield=inscalarfield->get( ssfield );
    ColorMapHandle cmap;
    int have_cmap=inColorMap->get( cmap );
    if(!have_cmap)
	have_sfield=0;

    if (init == 1) 
    {
	init = 0;
	GeomObj *w2d = widget2d->GetWidget() ;
	GeomObj *w3d = widget3d->GetWidget() ;
	GeomGroup* w = new GeomGroup;
	w->add(w2d);
	w->add(w3d);
	widget_id = ogeom->addObj( w, widget_name, &widget_lock );

	widget2d->Connect( ogeom );
	widget2d->SetRatioR( 0.2 );
	widget2d->SetRatioD( 0.2 );

	widget3d->Connect( ogeom );
	widget3d->SetRatioR( 0.2 );
	widget3d->SetRatioD( 0.2 );
	widget3d->SetRatioI( 0.2 );
    }
    int do_3d=1;
    if(type.get() == "2D")
	do_3d=0;

    widget2d->SetState(!do_3d);
    widget3d->SetState(do_3d);
    double ld=vfield->longest_dimension();
    if (do_3d){
	if(need_find3d != 0){
	    Point min, max;
	    vfield->get_bounds( min, max );
	    Point center = min + (max-min)/2.0;
	    Point right( max.x(), center.y(), center.z());
	    Point down( center.x(), min.y(), center.z());
	    Point in( center.x(), center.y(), min.z());
	    widget3d->SetPosition( center, right, down, in);
	    widget3d->SetScale( ld/20. );
	}
	need_find3d = 0;
    } else {
	if (need_find2d != 0){
	    Point min, max;
	    vfield->get_bounds( min, max );
	    Point center = min + (max-min)/2.0;
	    double max_scale;
	    if (need_find2d == 1) {
		// Find the field and put in optimal place
		// in xy plane with reasonable frame thickness
		Point right( max.x(), center.y(), center.z());
		Point down( center.x(), min.y(), center.z());
		widget2d->SetPosition( center, right, down);
		max_scale = Max( (max.x() - min.x()), (max.y() - min.y()) );
	    } else if (need_find2d == 2) {
		// Find the field and put in optimal place
		// in yz plane with reasonable frame thickness
		Point right( center.x(), center.y(), max.z());
		Point down( center.x(), min.y(), center.z());	    
		widget2d->SetPosition( center, right, down);
		max_scale = Max( (max.z() - min.z()), (max.y() - min.y()) );
	    } else {
		// Find the field and put in optimal place
		// in xz plane with reasonable frame thickness
		Point right( max.x(), center.y(), center.z());
		Point down( center.x(), center.y(), min.z());	    
		widget2d->SetPosition( center, right, down);
		max_scale = Max( (max.x() - min.x()), (max.z() - min.z()) );
	    }
	    widget2d->SetScale( max_scale/20. );
	    need_find2d = 0;
	}
    }
    
    // get the position of the frame widget
    Point 	center, R, D, I;
    int u_num, v_num, w_num;
    if(do_3d){
	widget3d->GetPosition( center, R, D, I);
	double u_fac = widget3d->GetRatioR();
	double v_fac = widget3d->GetRatioD();
	double w_fac = widget3d->GetRatioI();
	u_num = (int)(u_fac*100);
	v_num = (int)(v_fac*100);
	w_num = (int)(w_fac*100);
    } else {
	widget2d->GetPosition( center, R, D);
	I = center;
	double u_fac = widget2d->GetRatioR();
	double v_fac = widget2d->GetRatioD();
	u_num = (int)(u_fac*100);
	v_num = (int)(v_fac*100);
	w_num = 2;
    }
    Vector v1 = R - center,
    v2 = D - center,
    v3 = I - center;
    
    cerr << "unum = "<<u_num<<"  vnum="<<v_num<<"  wnum="<<w_num<<"\n";
//    u_num=v_num=w_num=4;

    // calculate the corner and the
    // u and v vectors of the cutting plane
    Point corner = center - v1 - v2 - v3;
    Vector u = v1 * 2.0,
    v = v2 * 2.0,
    w = v3 * 2.0;
    
    // create the grid for the cutting plane
    double lenscale = length_scale.get(),
    widscale = width_scale.get(),
    headlen = head_length.get();
    int exhaustive = exhaustive_flag.get();
    GeomArrows* arrows = new GeomArrows(widscale, 1.0-headlen, drawcylinders.get(), shaft_rad.get() );
    for (int i = 0; i < u_num; i++)
	for (int j = 0; j < v_num; j++)
	    for(int k = 0; k < w_num; k++)
		{
		    Point p = corner + u * ((double) i/(u_num-1)) + 
			v * ((double) j/(v_num-1)) +
			    w * ((double) k/(w_num-1));
		    
		    // Query the vector field...
		    Vector vv;
		    int ii=0;
		    if (vfield->interpolate( p, vv, ii, exhaustive)){
			if(have_sfield){
			    // get the color from cmap for p 	    
			    MaterialHandle matl;
			    double sval;
//			    ii=0;
			    if (ssfield->interpolate( p, sval, ii, exhaustive))
				matl = cmap->lookup( sval);
			    else
				{
				    matl = outcolor;
				}

			    if(vv.length2()*lenscale > 1.e-3)
			      arrows->add(p, vv*lenscale, matl, matl, matl);
			} else {
			  if(vv.length2()*lenscale > 1.e-3)
			    arrows->add(p, vv*lenscale);
			  else
			      cerr << "vv.length2()="<<vv.length2()<<"\n";
			}
		    }
		}

    // delete the old grid/cutting plane
    if (old_grid_id != 0)
	ogeom->delObj( old_grid_id );

    grid_id = ogeom->addObj(arrows, module_name);
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
	    if(type.get() == "2D")
		need_find2d=1;
	    else
		need_find3d=1;
	    want_to_execute();
	}
    else if(args[1] == "findyz")
	{
	    if(type.get() == "2D")
		need_find2d=2;
	    else
		need_find3d=1;
	    want_to_execute();
	}
    else if(args[1] == "findxz")
	{
	    if(type.get() == "2D")
		need_find2d=3;
	    else
		need_find3d=1;
	    want_to_execute();
	}
    else
	{
	    Module::tcl_command(args, userdata);
	}
}
