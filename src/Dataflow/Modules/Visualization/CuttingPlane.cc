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
#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/GeometryPort.h>
#include <SCICore/Datatypes/Mesh.h>
#include <PSECore/Datatypes/ScalarFieldPort.h>
#include <SCICore/Datatypes/ScalarFieldRGBase.h>
#include <SCICore/Datatypes/ScalarFieldUG.h>
#include <PSECore/Datatypes/ColorMapPort.h>
#include <SCICore/Geom/GeomGrid.h>
#include <SCICore/Geom/GeomGroup.h>
#include <SCICore/Geom/GeomLine.h>
#include <SCICore/Geom/Material.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Math/MinMax.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLvar.h>

#include <PSECore/Widgets/ScaledFrameWidget.h>
#include <iostream.h>

#define CP_PLANE 0
#define CP_SURFACE 1
#define CP_CONTOUR 2

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace PSECore::Widgets;
using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;
using namespace SCICore::Geometry;
using namespace SCICore::Math;

#define EPS 1.e-6

/**************************************
CLASS
   CuttingPlane
       CuttingPlane interpolates a planar slice through the
       unstructured data and maps data values to colors on a
       semitransparent surface.

GENERAL INFORMATION

   CuttingPlane
  
   Author:  Colette Mullenhoff<br>
            Department of Computer Science<br>
            University of Utah
   Date:    June 1995
   
   C-SAFE
   
   Copyright <C> 1995 SCI Group

KEYWORDS
   Visualization, Widget, GenColormap

DESCRIPTION
   CuttingPlane interpolates a planar slice through the
   unstructured data and maps data values to colors on a
   semitransparent surface.  The plane can be manipulated with a
   3D widget to allow the user to look at different
   cross sections of the electric potential.

WARNING
   None

****************************************/

class CuttingPlane : public Module {
   ScalarFieldIPort *inscalarfield;
   ColorMapIPort *inColorMap;
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
   TCLdouble where;
   TCLint need_find;
   TCLint localMinMaxTCL;
   TCLint fullRezTCL;
   TCLint exhaustiveTCL;
   MaterialHandle outcolor;
   int grid_id;
   clString msg;
   Mesh* m;
public:
 
        // GROUP:  Constructors:
        ///////////////////////////
        //
        // Constructs an instance of class CuttingPlane
        //
        // Constructor taking
        //    [in] id as an identifier
        //
   CuttingPlane(const clString& id);

        // GROUP:  Destructor:
        ///////////////////////////
        // Destructor
   virtual ~CuttingPlane();

        // GROUP:  Access functions:
        ///////////////////////////
        //
        // execute() - execution scheduled by scheduler
   virtual void execute();


        //////////////////////////
        //
        // tcl_commands - overides tcl_command in base class Module, takes:
        //                                  findxy, findyz, findxz, 
        //                                  plusx, minusx, plusy, minusy, plusz, minusz,
        //                                  connectivity
   virtual void tcl_command(TCLArgs&, void*);
};

Module* make_CuttingPlane(const clString& id) {
  return new CuttingPlane(id);
}

//static clString module_name("CuttingPlane");
static clString widget_name("CuttingPlane Widget");

CuttingPlane::CuttingPlane(const clString& id)
: Module("CuttingPlane", id, Filter), 
  cutting_plane_type("cutting_plane_type",id, this),
  need_find("need_find",id,this),
  scale("scale", id, this), offset("offset", id, this),
  num_contours("num_contours", id, this), where("where", id, this),
  localMinMaxTCL("localMinMaxTCL", id, this), 
  fullRezTCL("fullRezTCL", id, this), exhaustiveTCL("exhaustiveTCL", id, this)
{
    // Create the input ports
    // Need a scalar field and a ColorMap
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

    widget = scinew ScaledFrameWidget(this, &widget_lock, INIT);
    grid_id=0;

    need_find.set(1);
    
    outcolor=scinew Material(Color(0,0,0), Color(0,0,0), Color(0,0,0), 0);
    msg="";
    m=0;
}

CuttingPlane::~CuttingPlane()
{
}

void CuttingPlane::execute()
{
    int old_grid_id = grid_id;
    static int find = -1;
    int cmapmin, cmapmax;

    // get the scalar field and ColorMap...if you can
    ScalarFieldHandle sfield;
    if (!inscalarfield->get( sfield ))
	return;
    ColorMapHandle cmap;
    if (!inColorMap->get( cmap ))
	return;
    cmapmin=cmap->getMin();
    cmapmax=cmap->getMax();
    ScalarFieldUG* sfug=sfield->getUG();
    if (sfug) m=sfug->mesh.get_rep(); else m=0;
    if (init == 1) 
    {
	init = 0;
	GeomObj *w = widget->GetWidget() ;
	widget_id = ogeom->addObj( w, widget_name, &widget_lock );
	widget->Connect( ogeom );
	widget->SetRatioR( 0.4 );
	widget->SetRatioD( 0.4 );
    }
    if (need_find.get() != find)
    {
	Point min, max;
	sfield->get_bounds( min, max );
	Point center = min + (max-min)/2.0;
	double max_scale;
	double wh=where.get();
	if (need_find.get() == 1)
	{   // Find the field and put in optimal place
	    // in xy plane with reasonable frame thickness
	    center.z(min.z()*wh+max.z()*(1-wh));
	    Point right( max.x(), center.y(), center.z());
	    Point down( center.x(), min.y(), center.z());
	    max_scale = Max( (max.x() - min.x()), (max.y() - min.y()) );
	    widget->SetScale( max_scale/30. );
	    widget->SetPosition( center, right, down);
	}
	else if (need_find.get() == 2)
	{   // Find the field and put in optimal place
	    // in yz plane with reasonable frame thickness
	    center.x(min.x()*wh+max.x()*(1-wh));
	    Point right( center.x(), center.y(), max.z());
	    Point down( center.x(), min.y(), center.z());	    
	    max_scale = Max( (max.z() - min.z()), (max.y() - min.y()) );
	    widget->SetScale( max_scale/30. );
	    widget->SetPosition( center, right, down);
	}
	else
	{   // Find the field and put in optimal place
	    // in xz plane with reasonable frame thickness
	    center.y(min.y()*wh+max.y()*(1-wh));
	    Point right( max.x(), center.y(), center.z());
	    Point down( center.x(), center.y(), min.z());	    
	    max_scale = Max( (max.x() - min.x()), (max.z() - min.z()) );
	    widget->SetScale( max_scale/30. );
	    widget->SetPosition( center, right, down);
	}
	find = need_find.get();
    }

    // advance or decrement along x, y, or z
    if (msg != "") {
	ScalarFieldRGBase *sfb=sfield->getRGBase();
	if (!sfb) {
	    cerr << "Error - not a regular grid... can't increment/decrement!\n";
	    msg="";
	    return;
	}
	Point center, right, down;
	widget->GetPosition(center, right, down);
	Point min, max;
	Vector diag;
	sfield->get_bounds(min, max);
	diag=max-min;
	diag.x(diag.x()/(sfb->nx-1));
	diag.y(diag.y()/(sfb->ny-1));
	diag.z(diag.z()/(sfb->nz-1));
	if (msg=="plusx") {
	    center.x(center.x()+diag.x());
	    right.x(right.x()+diag.x());
	    down.x(down.x()+diag.x());
	} else if (msg=="minusx") {
	    center.x(center.x()-diag.x());
	    right.x(right.x()-diag.x());
	    down.x(down.x()-diag.x());
	} else if (msg=="plusy") {
	    center.y(center.y()+diag.y());
	    right.y(right.y()+diag.y());
	    down.y(down.y()+diag.y());
	} else if (msg=="minusy") {
	    center.y(center.y()-diag.y());
	    right.y(right.y()-diag.y());
	    down.y(down.y()-diag.y());
	} else if (msg=="plusz") {
	    center.z(center.z()+diag.z());
	    right.z(right.z()+diag.z());
	    down.z(down.z()+diag.z());
	} else { // if (msg=="minusz")
	    center.z(center.z()-diag.z());
	    right.z(right.z()-diag.z());
	    down.z(down.z()-diag.z());
	}
	widget->SetPosition( center, right, down);
	msg="";
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

    int u_num = (int) (u_fac * 500),
        v_num = (int) (v_fac * 500);

    if (fullRezTCL.get()) {
	ScalarFieldRGBase *sfb=sfield->getRGBase();
	if (!sfb) {
	    cerr << "Error - not a regular grid... can't use Full Resolution!\n";
	}
	int most=Max(Max(sfb->nx, sfb->ny), sfb->nz);
	u_num=v_num=most;
    }
    //    cout << "u fac = " << u_fac << "\nv fac = " << v_fac << endl;
    
    int localMinMax=localMinMaxTCL.get();

    int exhaustive=exhaustiveTCL.get();
    // Get the scalar values and corresponding
    // colors to put in the cutting plane
    if (cptype != CP_CONTOUR)
      {
#if 0
	double alpha=1.0;
#endif
	//	GeomGrid::Format format=GeomGrid::WithMaterials;
	//	if (cptype == CP_SURFACE)
	//	    format=GeomGrid::WithNormAndMatl;
	GeomGrid* grid = new GeomGrid( u_num, v_num, corner, u, v,
				       0);
        Vector unorm=u.normal();
        Vector vnorm=v.normal();
        Vector N(Cross(unorm, vnorm));
#if 0
	if (cptype != CP_SURFACE)
	  grid->texture();
#endif

	int ix = 0;
	int i, j;

	int haveval=0;
	double min, max, invrange;

	if (localMinMax) {
	    // get the min and max values from this slice
	    for (i = 0; i < u_num; i++)
		for (j = 0; j < v_num; j++) {
		    Point p = corner + u * ((double) i/(u_num-1)) + 
			v * ((double) j/(v_num-1));
		    double sval;
		    if (sfield->interpolate( p, sval, ix) || (ix=0) || sfield->interpolate( p, sval, ix, EPS, EPS, exhaustive)) {
			if (!haveval) { min=max=sval; haveval=1; }
			else if (sval<min) min=sval;
			else if (sval>max) max=sval;
		    }
		}
	    invrange=(cmapmax-cmapmin)/(max-min);
	}

	for (i = 0; i < u_num; i++)
	  for (j = 0; j < v_num; j++) {
	    Point p = corner + u * ((double) i/(u_num-1)) + 
	      v * ((double) j/(v_num-1));
	    double sval;
	    
	    // get the color from cmap for p 	    
	    MaterialHandle matl;
	    if (sfield->interpolate( p, sval, ix) || (ix=0) || sfield->interpolate( p, sval, ix, EPS, EPS, exhaustive)) {
		if (localMinMax)	// use local min/max to scale
		    sval=(sval-min)*invrange+cmapmin;
		matl = cmap->lookup( sval);
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
	    
	    // put the color into the cutting plane (grid) at i, j
	    if (cptype == CP_SURFACE)
	      {
		double h = sval;
		Vector G(sfield->gradient(p));
		double umag=Dot(unorm, G)*scale_fac;
		double vmag=Dot(vnorm, G)*scale_fac;
		Vector normal(N-unorm*umag-vnorm*vmag);
		grid->set(i, j, ((h*scale_fac) + offset_fac), normal, matl/*,alpha*/);
	      }
	    else  			// if (cptype == CP_PLANE)
	      grid->set(i, j, 0, matl/*,alpha*/);
	  }
	// delete the old grid/cutting plane
	if (old_grid_id != 0) {
	  ogeom->delObj( old_grid_id );
	}
	grid_id = ogeom->addObj(grid, "Cutting Plane");
	old_grid_id = grid_id;
	
      }

    else
    {
	double min, max, invrange;
	sfield->get_minmax( min, max );

	if (localMinMax) {
	    // get the min and max values from this slice
	    int ix = 0;
	    int i, j;
	    int haveval=0;
 
	    for (i = 0; i < u_num; i++)
		for (j = 0; j < v_num; j++) {
		    Point p = corner + u * ((double) i/(u_num-1)) + 
			v * ((double) j/(v_num-1));
		    double sval;
		    if (sfield->interpolate( p, sval, ix) || (ix=0) || sfield->interpolate( p, sval, ix, EPS, EPS, exhaustive)) {
			if (!haveval) { min=max=sval; haveval=1; }
			else if (sval<min) min=sval;
			else if (sval>max) max=sval;
		    }
		}
	    invrange=(cmapmax-cmapmin)/(max-min);
	} else {
	    // get the min and max values from the field
	    sfield->get_minmax( min, max );
	}
	// get the number of contours
	int contours = num_contours.get();
	if (contours >= 2)
	{
	    GeomGroup *cs = new GeomGroup;
	    Array1<GeomLines *> col_group( contours);
	    Array1<GeomMaterial *> colrs( contours);
	    Array1<double> values( contours);

	    // get the "contour" number of values from 
	    // the field, find corresponding colors,	
	    // and add a Material and it's group to the tree
	    int i;
	    for (i = 0; i < contours; i++)
	    {
		if (localMinMax) values[i]=i/(contours-1.)*
				     (cmapmax-cmapmin)+cmapmin;
		else values[i]=((double)i/(contours - 1)) * (max - min) + min;
		MaterialHandle matl;
		matl = cmap->lookup( values[i] );
		col_group[i] = new GeomLines;
		colrs[i] = new GeomMaterial( col_group[i], matl );
		cs->add( colrs[i] );
	    }
	    int ix=0;
	    // look at areas in the plane to find the contours
	    for (i = 0; i < u_num-1; i++)
		for (int j = 0; j < v_num-1; j++)
		{
		    Point p1 = corner + u * ((double) i/(u_num-1)) + 
			                v * ((double) j/(v_num-1));
		    Point p2 = corner + u * ((double) (i+1)/(u_num-1)) + 
			                v * ((double) j/(v_num-1));
		    Point p3 = corner + u * ((double) (i+1)/(u_num-1)) + 
			                v * ((double) (j+1)/(v_num-1));
		    Point p4 = corner + u * ((double) i/(u_num-1)) + 
			                v * ((double) (j+1)/(v_num-1));
		    double sval1;
		    double sval2;
		    double sval3;
		    double sval4;

		    // get the value from the field for each point	    
		    if ( (sfield->interpolate( p1,sval1,ix) || (ix=0) || 
			  sfield->interpolate( p1, sval1, ix, EPS, EPS, exhaustive)) && 
			(sfield->interpolate( p2,sval2,ix) || (ix=0) || 
			 sfield->interpolate( p2, sval2, ix, EPS, EPS, exhaustive)) && 
			(sfield->interpolate( p3,sval3, ix) || (ix=0) || 
			 sfield->interpolate( p3, sval3, ix, EPS, EPS, exhaustive)) && 
			(sfield->interpolate( p4, sval4,ix) || (ix=0) || 
			 sfield->interpolate( p4, sval4, ix, EPS, EPS, exhaustive)))
		    {
			if (localMinMax) {	// use local min/max to scale
			    sval1=(sval1-min)*invrange+cmapmin;
			    sval2=(sval2-min)*invrange+cmapmin;
			    sval3=(sval3-min)*invrange+cmapmin;
			    sval4=(sval4-min)*invrange+cmapmin;
			}			    
			// find the indices of the values array between smin & smax
			double smin = Min( Min( sval1, sval2), Min( sval3, sval4)),
			       smax = Max( Max( sval1, sval2), Max( sval3, sval4));
			int i1, i2;
			if (localMinMax) {
			    i1 = RoundUp((smin-cmapmin)*(contours - 1)/(cmapmax-cmapmin));
			    i2 = RoundDown((smax-cmapmin)*(contours - 1)/(cmapmax-cmapmin));
			} else {
			    i1=RoundUp((smin-min)*(contours - 1)/(max - min));
			    i2=RoundDown((smax-min)*(contours-1)/(max - min));
			}
			if(!localMinMax && (smin < min || smax > max))
			{
			    cerr << "OOPS: " << endl;
			    cerr << "smin, smax=" << smin << " " << smax << endl;
			    cerr << "min, max=" << min << " " << max << endl;
			    continue;
			}
			
			// find and add the contour lines if they exist in this area
			for (int k = i1; k <= i2; k++)
			{
			    int found = 0;
			    Point x1, x2;

			    if ((sval1 <= values[k] && values[k] < sval2) ||
				(sval2 < values[k] && values[k] <= sval1))
			    {
				x1 = Interpolate( p1, p2, (values[k]-sval1)/(sval2-sval1));
				++found;
			    }
			    if ((sval2 <= values[k] && values[k] < sval3) ||
				(sval3 < values[k] && values[k] <= sval2))
			    {
				x2 = Interpolate( p2, p3, (values[k]-sval2)/(sval3-sval2));
				if (!found)
				    x1 = x2;
				++found;
			    }
			    if (((sval3 <= values[k] && values[k] < sval4) ||
				 (sval4 < values[k] && values[k] <= sval3)) && found < 2)
			    {
				x2 = Interpolate( p3, p4, (values[k]-sval3)/(sval4-sval3));
				if (!found)
				    x1 = x2;
				++found;
			    }
			    if (((sval1 < values[k] && values[k] <= sval4) ||
				 (sval4 <= values[k] && values[k] < sval1)) && found < 2)
			    {
				x2 = Interpolate( p1, p4, (values[k]-sval1)/(sval4-sval1));
				++found;
			    }
			    // did we find two points to draw a line with?
			    if (found == 2)
				col_group[k]->add(x1, x2);
			}
		    }
		}
    // delete the old grid/cutting plane
	    if (old_grid_id != 0) {
	      ogeom->delObj( old_grid_id );

	    }
	    grid_id =  ogeom->addObj( cs, "Contour Plane");
	    old_grid_id = grid_id;
		      
	}
	
    }

}

void CuttingPlane::widget_moved(int last)
{
    if(last && !abort_flag)
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
	need_find.set(1);
	want_to_execute();
    }
    else if(args[1] == "findyz")
    {
	need_find.set(2);
	want_to_execute();
    }
    else if(args[1] == "findxz")
    {
	need_find.set(3);
	want_to_execute();
    }
    else if(args[1] == "plusx")
    {
	msg="plusx";
	want_to_execute();
    }
    else if(args[1] == "minusx")
    {
	msg="minusx";
	want_to_execute();
    }
    else if(args[1] == "plusy")
    {
	msg="plusy";
	want_to_execute();
    }
    else if(args[1] == "minusy")
    {
	msg="minusy";
	want_to_execute();
    }
    else if(args[1] == "plusz")
    {
	msg="plusz";
	want_to_execute();
    }
    else if(args[1] == "minusz")
    {
	msg="minusz";
	want_to_execute();
    }
    else if(args[1] == "connectivity")
    {
	
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
// Revision 1.5  1999/08/25 03:48:05  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.4  1999/08/19 23:17:56  sparker
// Removed a bunch of #include <SCICore/Util/NotFinished.h> statements
// from files that did not need them.
//
// Revision 1.3  1999/08/18 20:20:04  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
// Revision 1.2  1999/08/17 06:37:47  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:58:11  mcq
// Initial commit
//
// Revision 1.3  1999/05/06 20:09:07  dav
// added .h files back
//
// Revision 1.2  1999/05/05 22:11:39  dav
// updated CuttingPlane
//
// Revision 1.1.1.1  1999/04/24 23:12:33  dav
// Import sources
//
//
