/*
 *  HedgehogLitLines.cc:  
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
#include <Datatypes/VoidStar.h> // ljd added
#include <Datatypes/VoidStarPort.h> // ljd added
#include <Geom/BBoxCache.h>
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

class HedgehogLitLines : public Module {
   VectorFieldIPort *invectorfield;
   ScalarFieldIPort* inscalarfield;
   ColorMapIPort *inColorMap;
   GeometryOPort* ogeom;
   CrowdMonitor widget_lock;
   int locMode;
   int init;
   int widget_id;
   ScaledBoxWidget* widget3d;
   ScaledFrameWidget *widget2d;
   virtual void widget_moved(int last);
   TCLdouble length_scale;
   TCLdouble width_scale;
   TCLdouble head_length;
   TCLstring type;
   TCLint locus_size;
   TCLint use_locus;
   TCLint exhaustive_flag;
   Colorub outcolorub, graycolorub, defaultcolorub;
   int line_id1;
   int line_id2;
   int line_id3;
   int line_id4;
   TexGeomLines *lines1, *lines2, *lines3, *lines4;
   int need_find2d;
   int need_find3d;
   VoidStarIPort * inpoint;  // ljd added
   PhantomXYZ * xyz; // ljd added 
   int haveXYZ; //ljd
   BBox bbox;
public:
   HedgehogLitLines(const clString& id);
   HedgehogLitLines(const HedgehogLitLines&, int deep);
   virtual ~HedgehogLitLines();
   virtual Module* clone(int deep);
   virtual void execute();
   virtual void tcl_command(TCLArgs&, void*);
};

extern "C" {
Module* make_HedgehogLitLines(const clString& id)
{
   return scinew HedgehogLitLines(id);
}
}

static clString module_name("HedgehogLitLines");
static clString widget_name("HedgehogLitLines Widget");

HedgehogLitLines::HedgehogLitLines(const clString& id)
: Module("HedgehogLitLines", id, Filter), 
  length_scale("length_scale", id, this),
  width_scale("width_scale", id, this),
  head_length("head_length", id, this),
  locus_size("locus_size", id, this),
  use_locus("use_locus", id, this),
  type("type", id, this), haveXYZ(0),
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
			
    inpoint = scinew VoidStarIPort(this, "PhantomEndpoint", 
                                   VoidStarIPort::Atomic); // ljd added	
    add_iport(inpoint); // ljd added
	
    // Create the output port
    ogeom = scinew GeometryOPort(this, "Geometry", 
			      GeometryIPort::Atomic);
    add_oport(ogeom);
    init = 1;
    float INIT(.1);

    widget2d = scinew ScaledFrameWidget(this, &widget_lock, INIT);
    widget3d = scinew ScaledBoxWidget(this, &widget_lock, INIT);

    line_id1=0;
    line_id2=0;
    line_id3=0;
    line_id4=0;

    need_find2d=1;
    need_find3d=1;
    
    cerr << 9 << endl;
    Color outc(0,0,0);
    Color defc(.8,0,0);
    Color grayc(.2,.2,.2);
    outcolorub=Colorub(outc);
    defaultcolorub=Colorub(defc);
    graycolorub=Colorub(grayc);

    // for "default" lines (red)
    lines1 = new TexGeomLines;

    // for colormapped lines
    lines2 = new TexGeomLines;

    // for background lines
    lines3 = new TexGeomLines;

    // for the "center of the locus" lines
    lines4 = new TexGeomLines;
}

HedgehogLitLines::HedgehogLitLines(const HedgehogLitLines& copy, int deep)
: Module(copy, deep), length_scale("length_scale", id, this),
  width_scale("width_scale", id, this),
  head_length("head_length", id, this),
  use_locus("use_locus", id, this),
  locus_size("locus_size", id, this),
  type("type", id, this), haveXYZ(0),
  exhaustive_flag("exhaustive_flag", id, this)
{
   NOT_FINISHED("HedgehogLitLines::HedgehogLitLines");
}

HedgehogLitLines::~HedgehogLitLines()
{
}

Module* HedgehogLitLines::clone(int deep)
{
   return scinew HedgehogLitLines(*this, deep);
}

void HedgehogLitLines::execute()
{

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
// ljd added
    // get the current phantom endpoint position if passed
    if (!haveXYZ) {
      VoidStarHandle rmHandle;
      inpoint->get(rmHandle);
      if (rmHandle.get_rep()) { 
         if ((xyz = rmHandle->getPhantomXYZ())) haveXYZ = 1; // check for contents and
           //assign to xyz
      }
    } 
// end ljd added
 
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

	Point min, max;
	vfield->get_bounds( min, max );
	bbox.extend(min); bbox.extend(max);
    }
    
    double lsOld=-1;
    Vector locPOld;
    double lenscaleOld=0;
    double widscaleOld=0;
    double headlenOld=0;
    clString typeStrOld=type.get();

    locMode = (haveXYZ && use_locus.get() == 1);
    do {
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
	if(do_3d == 1){
	    widget3d->GetPosition( center, R, D, I);
	    double u_fac = widget3d->GetRatioR();
	    double v_fac = widget3d->GetRatioD();
	    double w_fac = widget3d->GetRatioI();
	    u_num = (int)(u_fac*100);
	    v_num = (int)(v_fac*100);
	    w_num = (int)(w_fac*100);
	} else if (do_3d == 0) {
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
	
	// calculate the corner and the
	// u and v vectors of the cutting plane
	Point corner = center - v1 - v2 - v3;
	Vector u = v1 * 2.0,
	    v = v2 * 2.0,
	    w = v3 * 2.0;
	
	Vector locP;
	double dist;
	// create the grid for the cutting plane
	double lenscale = length_scale.get();
	double widscale = width_scale.get();
	double headlen = head_length.get();
	int exhaustive = exhaustive_flag.get();
	clString typeStr = type.get();
	if (locMode) {
	    // look at xyz point value,
	    // with locks,
	    xyz->updateLock.read_lock();
	    locP = xyz->position;
	    xyz->updateLock.read_unlock();
	    
	    // grab the size from the tcl interface       
	    double ls = locus_size.get();
	    Point min, max;
	    vfield->get_bounds( min, max );
	    
	    // scale by size of field
	    dist = ls/100. * (max-min).length();
	    
	    if (locP == locPOld && ls == lsOld &&
		lenscale == lenscaleOld && widscale == widscaleOld &&
		headlen == headlenOld && typeStr == typeStrOld) {
		reset_vars();
		locMode = (haveXYZ && (use_locus.get() == 1));
		continue;
	    }
	    lsOld=ls;
	    locPOld=locP;
	    widscaleOld = widscale;
	    lenscaleOld = lenscale;
	    headlenOld = headlen;
	    typeStrOld = typeStr;
	}

	lines1->mutex.lock();
	lines1->pts.resize(0);
	lines1->tangents.resize(0);
	lines1->colors.resize(0);
	lines1->mutex.unlock();

	lines2->mutex.lock();
	lines2->pts.resize(0);
	lines2->tangents.resize(0);
	lines2->colors.resize(0);
	lines2->mutex.unlock();

	lines3->mutex.lock();
	lines3->pts.resize(0);
	lines3->tangents.resize(0);
	lines3->colors.resize(0);
	lines3->mutex.unlock();

	lines4->mutex.lock();
	lines4->pts.resize(0);
	lines4->tangents.resize(0);
	lines4->colors.resize(0);
	lines4->mutex.unlock();

	for (int i = 0; i < u_num; i++)
	    for (int j = 0; j < v_num; j++)
		for(int k = 0; k < w_num; k++)
		    {
			Point p = corner + u * ((double) i/(u_num-1)) + 
			    v * ((double) j/(v_num-1)) +
			    w * ((double) k/(w_num-1));
			
			// display only those vectors around that locus
			Vector vv;
			int ii=0;
			if (vfield->interpolate( p, vv, ii, exhaustive)){
			    if (locMode && (p-locP).vector().length()>dist ) {
				if(vv.length2()*lenscale > 1.e-3) {
				    lines3->mutex.lock();
				    lines3->add(p, vv*lenscale, graycolorub);
				    lines3->mutex.unlock();
				}
			    } else {
				// Query the vector field...
				if(have_sfield){
				    // get the color from cmap for p 	    
				    Colorub clrub;
				    double sval;
				    ii=0;
				    if (ssfield->interpolate( p, sval, ii, exhaustive))
					clrub = cmap->lookup(sval)->diffuse;
				    else
					{
					    clrub = outcolorub;
					}
				    
				    if(vv.length2()*lenscale > 1.e-3) {
					lines2->mutex.lock();
					lines2->add(p, vv*lenscale, clrub);
					lines2->mutex.unlock();
				    }
				} else {
				    if(vv.length2()*lenscale > 1.e-3) {
					lines1->mutex.lock();
					lines1->add(p, vv*lenscale, defaultcolorub);
					lines1->mutex.unlock();
				    }
				}
			    }
			}
		    }
	Vector vv;
	int ii=0;
	if (locMode && vfield->interpolate( locP.point(), vv, ii, exhaustive)){
	    if (have_sfield) {
		// get the color from cmap for p 	    
		Colorub clrub;
		double sval;
		ii=0;
		if (ssfield->interpolate( locP.point(), sval, ii, exhaustive))
		    clrub = cmap->lookup(sval)->diffuse;
		else
		    {
			clrub = outcolorub;
		    }
		
		if(vv.length2()*lenscale > 1.e-3) {
		    lines4->mutex.lock();
		    lines4->add(locP.point(), vv*lenscale, clrub);
		    lines4->mutex.unlock();
		}
	    } else {
		if(vv.length2()*lenscale > 1.e-3) {
		    lines4->mutex.lock();
		    lines4->add(locP.point(), vv*lenscale, defaultcolorub);
		    lines4->mutex.unlock();
		}
	    }
	}
	// delete the old grid/cutting plane
	
	if (line_id1 == 0) {
	    GeomBBoxCache *bb = scinew GeomBBoxCache(lines1, bbox);
	    line_id1 = ogeom->addObj(bb, module_name+"Default");
	}
	if (line_id2 == 0) {
	    GeomBBoxCache *bb = scinew GeomBBoxCache(lines2, bbox);
	    line_id2 = ogeom->addObj(bb, module_name+"ColorMapped");
	}
	if (line_id3 == 0) {
	    GeomBBoxCache *bb = scinew GeomBBoxCache(lines3, bbox);
	    line_id3 = ogeom->addObj(bb, module_name+"Background");
	}
	if (line_id4 == 0) {
	    GeomBBoxCache *bb = scinew GeomBBoxCache(lines4, bbox);
	    line_id4 = ogeom->addObj(bb, module_name+"Center");
	}
	ogeom->flushViews();
	reset_vars();  // calls tcl to resynch with tcl interface

	locMode = (haveXYZ && (use_locus.get() == 1));
	// ljd. Add new display technique.
	
    } while(locMode);
}

void HedgehogLitLines::widget_moved(int last)
{
    if(last && !abort_flag)
	{
	    abort_flag=1;
	    want_to_execute();
	}
}


void HedgehogLitLines::tcl_command(TCLArgs& args, void* userdata)
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
	    if (!locMode) want_to_execute();
	}
    else if(args[1] == "findyz")
	{
	    if(type.get() == "2D")
		need_find2d=2;
	    else
		need_find3d=1;
	    if (!locMode) want_to_execute();
	}
    else if(args[1] == "findxz")
	{
	    if(type.get() == "2D")
		need_find2d=3;
	    else
		need_find3d=1;
	    if (!locMode) want_to_execute();
	}
    else
	{
	    Module::tcl_command(args, userdata);
	}
}
