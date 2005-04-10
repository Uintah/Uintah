//static char *id="@(#) $Id$";

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

#include <SCICore/Containers/Array1.h>
#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/ColorMapPort.h>
#include <PSECore/Datatypes/GeometryPort.h>
#include <PSECore/Datatypes/ScalarFieldPort.h>
#include <PSECore/Datatypes/VectorFieldPort.h>
#include <SCICore/Geom/GeomArrows.h>
#include <SCICore/Geom/GeomGroup.h>
#include <SCICore/Geom/GeomLine.h>
#include <SCICore/Geom/Material.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Math/MinMax.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <SCICore/Thread/CrowdMonitor.h>

#include <PSECore/Widgets/ScaledBoxWidget.h>
#include <PSECore/Widgets/ScaledFrameWidget.h>
#include <iostream>
using std::cerr;

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
using namespace SCICore::Math;
using namespace SCICore::Containers;

/**************************************
CLASS
   Hedgehog
      Hedgehog produces arrows at sample points in a vector field.


GENERAL INFORMATION

   Hedgehog
  
   Author:  Steven G. Parker (sparker@cs.utah.edu)
            
            Department of Computer Science
            
            University of Utah
   
   Date:    June 1995
   
   C-SAFE
   
   Copyright <C> 1995 SCI Group

KEYWORDS
   Visualization, vector_field, GenColormap

DESCRIPTION
   Hedgehog produces arrows at sample points in a vector
   field.  The length of the arrow indicates the magnitude of the
   field at that point, and the orientation indicates the
   direction of the field.  In addition, the shaft of the arrow
   is mapped to a scalar field value using a colormap produced by
   GenColormap.

WARNING
   None



****************************************/

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
 
        // GROUP:  Widgets:
        //////////////////////
        //
        // widget_moved -  
   virtual void widget_moved(int last);
   TCLdouble length_scale;
   TCLdouble width_scale;
   TCLdouble head_length;
   TCLstring type;
   TCLint exhaustive_flag;
   TCLint vector_default_color;
   TCLint drawcylinders;
   TCLdouble shaft_rad;
   MaterialHandle outcolor;
  MaterialHandle shaft;
  MaterialHandle head;
  MaterialHandle back;
   int grid_id;
   int need_find2d;
   int need_find3d;

public:
 
        // GROUP:  Constructors:
        ///////////////////////////
        //
        // Constructs an instance of class Hedgehog
        //
        // Constructor taking
        //    [in] id as an identifier
        //
   Hedgehog(const clString& id);
       
        // GROUP:  Destructor:
        ///////////////////////////
        // Destructor
   virtual ~Hedgehog();
  
        // GROUP:  Access functions:
        ///////////////////////////
        //
        // execute() - execution scheduled by scheduler
   virtual void execute();

        //////////////////////////
        //
        // tcl_commands - overides tcl_command in base class Module, takes:
        //                                  findxy,
        //                                  findyz,
        //                                  findxz
   virtual void tcl_command(TCLArgs&, void*);
};

static clString module_name("Hedgehog");
static clString widget_name("Hedgehog Widget");

extern "C" Module* make_Hedgehog(const clString& id) {
  return new Hedgehog(id);
}       

Hedgehog::Hedgehog(const clString& id)
: Module("Hedgehog", id, Filter), widget_lock("Hedgehog widget lock"),
  length_scale("length_scale", id, this),
  width_scale("width_scale", id, this),
  head_length("head_length", id, this),
  type("type", id, this),
  drawcylinders("drawcylinders", id, this),
  shaft_rad("shaft_rad", id, this),
  vector_default_color("vector_default_color", id, this),
  exhaustive_flag("exhaustive_flag", id, this),
  shaft(new Material(Color(0,0,0), Color(.6, .6, .6),
		     Color(.6, .6, .6), 10)),
  head(new Material(Color(0,0,0), Color(1,1,1), Color(.6, .6, .6), 10)),
  back(new Material(Color(0,0,0), Color(.6, .6, .6), Color(.6, .6, .6), 10))
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

Hedgehog::~Hedgehog()
{
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
    if(!have_cmap){
	have_sfield=0;
	if(vector_default_color.get() == 0){
	  *(shaft.get_rep()) = Material(Color(0,0,0), Color(.8, .8, .8),
			   Color(.6, .6, .6), 10);
	  *(head.get_rep()) = Material(Color(0,0,0), Color(1,1,1),
			  Color(.6, .6, .6), 10);
	  *(back.get_rep()) = Material(Color(0,0,0), Color(.8, .8, .8),
			  Color(.6, .6, .6), 10);
	} else if (vector_default_color.get() == 1) {
	  *(shaft.get_rep()) = Material(Color(0,0,0), Color(.4, .4, .4),
			   Color(.6, .6, .6), 10);
	  *(head.get_rep()) = Material(Color(0,0,0), Color(.4,.4,.4),
			  Color(.6, .6, .6), 10);
	  *(back.get_rep()) = Material(Color(0,0,0), Color(.4, .4, .4),
			  Color(.6, .6, .6), 10);
	} else {
	  *(shaft.get_rep()) = Material(Color(0,0,0), Color(.1, .1, .1),
			   Color(.6, .6, .6), 10);
	  *(head.get_rep()) = Material(Color(0,0,0), Color(.1,.1,.1),
			  Color(.6, .6, .6), 10);
	  *(back.get_rep()) = Material(Color(0,0,0), Color(.1, .1, .1),
			  Color(.6, .6, .6), 10);
	}
    }

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

			    if(vv.length2()*lenscale > 1.e-5)
			      arrows->add(p, vv*lenscale, matl, matl, matl);
			} else {
			  if(vv.length2()*lenscale > 1.e-5)
			    arrows->add(p, vv*lenscale, shaft, back, head);
			  //else
			  //    cerr << "vv.length2()="<<vv.length2()<<"\n";
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

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.10  2001/02/21 16:45:58  kuzimmer
// changed draw routine so that the vector will draw if its length is greater than 1e-5,  was 1e-3.
//
// Revision 1.9  2001/02/21 16:19:16  kuzimmer
// Added the ability to set the vector grayscale  when a colormap is not being used.
//
// Revision 1.8  2000/03/17 09:27:32  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.7  1999/10/07 02:07:06  sparker
// use standard iostreams and complex type
//
// Revision 1.6  1999/08/29 00:46:46  sparker
// Integrated new thread library
// using statement tweaks to compile with both MipsPRO and g++
// Thread library bug fixes
//
// Revision 1.5  1999/08/25 03:48:07  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.4  1999/08/19 23:17:57  sparker
// Removed a bunch of #include <SCICore/Util/NotFinished.h> statements
// from files that did not need them.
//
// Revision 1.3  1999/08/18 20:20:07  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
// Revision 1.2  1999/08/17 06:37:49  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:58:13  mcq
// Initial commit
//
// Revision 1.3  1999/06/21 23:52:52  dav
// updated makefiles.main
//
// Revision 1.2  1999/05/11 19:48:03  dav
// updated Hedgehog
//
// Revision 1.1.1.1  1999/04/24 23:12:33  dav
// Import sources
//
//
