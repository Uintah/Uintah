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

#include <Classlib/Array1.h>
#include <Classlib/NotFinished.h>
#include <Dataflow/Module.h>
#include <Datatypes/GeometryPort.h>
#include <Datatypes/ScalarFieldPort.h>
#include <Datatypes/ColorMapPort.h>
#include <Geom/Grid.h>
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
   MaterialHandle outcolor;
   int grid_id;

public:
   CuttingPlane(const clString& id);
   CuttingPlane(const CuttingPlane&, int deep);
   virtual ~CuttingPlane();
   virtual Module* clone(int deep);
   virtual void execute();

   virtual void tcl_command(TCLArgs&, void*);
};

extern "C" {
Module* make_CuttingPlane(const clString& id)
{
   return scinew CuttingPlane(id);
}
}

//static clString module_name("CuttingPlane");
static clString widget_name("CuttingPlane Widget");

CuttingPlane::CuttingPlane(const clString& id)
: Module("CuttingPlane", id, Filter), 
  cutting_plane_type("cutting_plane_type",id, this),
  need_find("need_find",id,this),
  scale("scale", id, this), offset("offset", id, this),
  num_contours("num_contours", id, this), where("where", id, this)
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
}

CuttingPlane::CuttingPlane(const CuttingPlane& copy, int deep)
: Module(copy, deep), cutting_plane_type("cutting_plane_type",id, this),
  need_find("need_find",id,this),
  scale("scale", id, this), offset("offset", id, this),
   num_contours("num_contours", id, this), where("where", id, this)
{
   NOT_FINISHED("CuttingPlane::CuttingPlane");
}

CuttingPlane::~CuttingPlane()
{
}

Module* CuttingPlane::clone(int deep)
{
   return scinew CuttingPlane(*this, deep);
}

void CuttingPlane::execute()
{
    int old_grid_id = grid_id;
    static int find = -1;

    // get the scalar field and ColorMap...if you can
    ScalarFieldHandle sfield;
    if (!inscalarfield->get( sfield ))
	return;
    ColorMapHandle cmap;
    if (!inColorMap->get( cmap ))
	return;

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

    //    cout << "u fac = " << u_fac << "\nv fac = " << v_fac << endl;
    
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
	for (int i = 0; i < u_num; i++)
	  for (int j = 0; j < v_num; j++) {
	    Point p = corner + u * ((double) i/(u_num-1)) + 
	      v * ((double) j/(v_num-1));
	    double sval;
	    
	    // get the color from cmap for p 	    
	    MaterialHandle matl;
	    if (sfield->interpolate( p, sval, ix) || (ix=0) || sfield->interpolate( p, sval, ix)) {
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
	// get the min and max values from the field
	double min, max;
	sfield->get_minmax( min, max );

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
		values[i] = ((double)i/(contours - 1)) * (max - min) + min;
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
		    if ( (sfield->interpolate( p1, sval1,ix) || 
			  (ix=0) || sfield->interpolate( p1, sval1,ix)) && 
			(sfield->interpolate( p2, sval2,ix) || 
			 (ix=0) || sfield->interpolate( p2, sval2,ix)) && 
			(sfield->interpolate( p3, sval3,ix) || 
			 (ix=0) || sfield->interpolate( p3, sval3,ix)) && 
			(sfield->interpolate( p4, sval4,ix) || 
			 (ix=0) || sfield->interpolate( p4, sval4,ix)))
		    {
			// find the indices of the values array between smin & smax
			double smin = Min( Min( sval1, sval2), Min( sval3, sval4)),
			       smax = Max( Max( sval1, sval2), Max( sval3, sval4));
			int i1 = RoundUp( (smin - min)*(contours - 1)/(max - min)),
			    i2 = RoundDown( (smax - min)*(contours - 1)/(max - min));

			if(smin < min || smax > max)
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
    else
    {
	Module::tcl_command(args, userdata);
    }
}
