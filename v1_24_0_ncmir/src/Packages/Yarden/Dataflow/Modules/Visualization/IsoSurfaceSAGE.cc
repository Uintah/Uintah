/*
 *  IsoSurfaceSAGE.cc 
 *      View Depended Iso Surface Extraction
 *      for Structures Grids (Bricks)
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Dec 1996
 *
 *  Copyright (C) 1996 SCI Group
 */


#include <stdio.h>
#include <unistd.h>

#include <Core/Thread/Time.h>
#include <Core/Containers/String.h>

#include <Core/Datatypes/ScalarFieldRG.h>

#include <Core/Thread/Thread.h>

#include <Core/Datatypes/Color.h>
#include <Core/Geom/Material.h>
#include <Core/Geom/GeomTriangles.h>
#include <Core/Geom/View.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/GeomObj.h>
#include <Core/Geom/GeomTri.h>
#include <Core/Geom/GeomLine.h>
#include <Core/Geom/GeomBox.h>
#include <Core/Geom/Pt.h>
#include <Core/Geom/GeomTransform.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Transform.h>
#include <Core/Geom/BBoxCache.h>
#include <Core/Geom/GeomDL.h>

#include <Core/Geometry/Vector.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/Trig.h>

#include <Core/GuiInterface/TCLTask.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/GuiInterface/TCL.h>

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/ScalarFieldPort.h>
#include <Dataflow/Ports/ColorMapPort.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/GeometryComm.h>
#include <Dataflow/Ports/SurfacePort.h>
#include <Dataflow/Ports/PathPort.h>

#include <Packages/Yarden/Core/Algorithms/Visualization/Sage.h>

#include <tcl.h>
#include <tk.h>
#include <math.h>
#include <iostream>
#include <sstream>
#include <values.h>



namespace Yarden {

using namespace SCIRun;    
    
    
    SysTime::SysClock extract_timer, vis_timer;

    int bbox_count;
    int debug = 1;
    int scan_yes;
    int scan_no;
  

    GeometryData *gd;

    // IsoSurfaceSAGE

    class IsoSurfaceSAGE : public Module 
    {
      // input
      ScalarFieldIPort* infield;  // input scalar fields (bricks)
      ScalarFieldIPort* incolorfield;
      ColorMapIPort* incolormap;
      PathIPort* icam_view;

      // ouput
      GeometryOPort* ogeom;       // input from salmon - view point

      // UI variables
      GuiDouble isoval;
      GuiDouble isoval_min, isoval_max;
      GuiInt tcl_bbox, tcl_value,tcl_visibility, tcl_scan;
      GuiInt tcl_reduce, tcl_all;
      GuiInt tcl_min_size, tcl_poll;
      GuiInt tcl_dl;
  
      int value, bbox_visibility, visibility;
      int scan, count_values, extract_all;

      int box_id;
      int surface_id;
      int points_id;

      Path       camv;
      bool has_camera_view;

      MaterialHandle bone;
      MaterialHandle flesh;
      MaterialHandle matl;
      MaterialHandle box_matl;
      MaterialHandle points_matl;
  
      GeomPts* points;
      GeomGroup* group;
      GeomObj* topobj;
  
      GeomTrianglesP *triangles;
      GeomMaterial *surface;

      SageBase<Module> *sage;
      Screen *screen;

      Point bmin, bmax;
      int dx, dy, dz, dim;
      int field_generation;
      double iso_value;

      int reduce;
  
      double bbox_limit_x, bbox_limit_y;
      double bbox_limit_x1, bbox_limit_y1;
      double left, right, top, bottom;

      GeometryData *default_gd;
      int xres, yres;

      //int busy;
      //int portid;

    public:
      IsoSurfaceSAGE( const clString& id);
      virtual ~IsoSurfaceSAGE();
      
      virtual void execute();
      //virtual void do_execute();

    private:
      void search();
      int set_view( const View &);
    };
    
    extern "C" Module* make_IsoSurfaceSAGE(const clString& id)
    {
      return scinew IsoSurfaceSAGE(id);
    }
    
    static clString module_name("IsoSurfaceSAGE");
    static clString box_name("SageBox");
    static clString surface_name("Sage");
    
    IsoSurfaceSAGE::IsoSurfaceSAGE(const clString& id)
      : Module("IsoSurfaceSAGE", id, Filter ), 
	isoval("isoval", id, this),
	isoval_min("isoval_min", id, this), 
	isoval_max("isoval_max", id, this),
	tcl_bbox("bbox", id, this), 
	tcl_value("value", id, this), 
	tcl_visibility("visibility", id, this),
	tcl_scan("scan", id, this),  
	tcl_reduce("reduce",id,this), 
	tcl_all("all",id,this),
	tcl_min_size("min_size", id, this),
	tcl_poll("poll", id, this),
	tcl_dl("dl", id, this)
    {
      // input ports
      infield=scinew ScalarFieldIPort(this, "Field",ScalarFieldIPort::Atomic);
      add_iport(infield);
      
      incolorfield=scinew ScalarFieldIPort(this, "Color Field",
					   ScalarFieldIPort::Atomic);
      add_iport(incolorfield);

      incolormap=scinew ColorMapIPort(this,"Color Map",ColorMapIPort::Atomic);
      add_iport(incolormap);
    
      icam_view=scinew PathIPort(this, "Camera View",  
				       PathIPort::Atomic);
      add_iport(icam_view);

      // output port
      ogeom=scinew GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
      add_oport(ogeom);
  
      // for handling our own input geom
      //busy = 0;
      //have_own_dispatch = 1;

      Color Flesh = Color(1.0000, 0.4900, 0.2500);
      Color Bone = Color(0.9608, 0.8706, 0.7020);
      flesh = scinew Material( Flesh*.1, Flesh*.6, Flesh*.6, 20 );
      bone = scinew Material( Bone*.1, Bone*.6, Bone*.6, 20 );
      box_matl=scinew Material(Color(0.3,0.3,0.3), 
			       Color(.8,.8,.8), 
			       Color(.7,.7,.7), 
			       20);
      points_matl=scinew Material(Color(0.3,0,0), 
				  Color(.8,0,0), 
				  Color(.7,.7,.7), 
				  20);
      surface_id = 0;
      points_id = 0;
      box_id = 0;
      field_generation = -1;

      default_gd = scinew GeometryData;
      default_gd->xres = 512;
      default_gd->yres = 512;
      default_gd->znear = 1;
      default_gd->zfar = 2;
      default_gd->view = scinew View( Point(0.65, 0.5, -4.5),
				      Point(0.5,0.5,0.5),
				      Vector(0,1,0),
				      17 );
      
      screen = new Screen;
      screen->setup( 512, 512 );

      sage = 0;
      xres = yres = 512;
    }

    IsoSurfaceSAGE::~IsoSurfaceSAGE()
    {
    }

    void
    IsoSurfaceSAGE::execute()
    {
      SysTime::SysClock start = SysTime::currentTicks();

      ScalarFieldHandle scalar_field;

      if(!infield->get(scalar_field)) {
	error("No input field\n");
	return;
      }
      
      if ( scalar_field->generation !=  field_generation ) {
	// save a handle to this field
	
	// create a new Sage algorithm
	SageBase<Module> *tmp = SageBase<Module>::make( scalar_field, this );
	if ( !tmp) {
	  error( "Can not work with this type of ScalarField");
	  return;
	}
	
	if ( sage ) delete sage;
	sage = tmp;
	sage->setScreen ( screen );
	
	// save the current generation
 	field_generation = scalar_field->generation;
	
	// reset the tcl
	double min, max;
	scalar_field->get_minmax( min, max );
	isoval_min.set(min);
	isoval_max.set(max);
	isoval.set((min+max)/2);
	reset_vars();

	// send the bbox to Salmon
	scalar_field->get_bounds( bmin, bmax );
	GeomBox *box = scinew GeomBox( bmin, bmax, 1 );
	GeomObj *bbox= scinew GeomMaterial( box, box_matl);
	box_id = ogeom->addObj( bbox, box_name );

	if ( points_id ) {
	  ogeom->delObj(points_id);
	  points_id = 0;
	}
	
	if(surface_id ) {
	  ogeom->delObj(surface_id);
	  surface_id = 0;
	}
	
	return;
      }

      // Get View information 
      
      // first, check if we got a view in the input port
      PathHandle camera;
      if ( !icam_view->get( camera ) || !camera.get_rep() || !camera->keyViews.size() || !set_view( camera->keyViews[0] )) {
	cerr << "using Salmon" << endl;
	// no. get the view from salmon
	gd = ogeom->getData(0, GEOM_VIEW);
	if ( !gd ) {
	  cerr << "using default view" << endl;
	  gd = default_gd;
	}
	
	sage->setView( *gd->view, 
		       gd->znear, gd->zfar, gd->xres, gd->yres );
	has_camera_view = false;
      }

      search();

      if ( has_camera_view ) 
	ogeom->setView( 0, camera->keyViews[0] );
      else if ( tcl_poll.get() ) { 
	  GeometryData *tmp_gd = ogeom->getData(0, GEOM_VIEW);
	  if ((*gd->view == *tmp_gd->view) ) 
	    usleep( 50000 );
	  want_to_execute();
	}
    
    
      SysTime::SysClock end = SysTime::currentTicks();
      printf("Exec Timer: %.3f\n\n", (end-(long long)start) *SysTime::secondsPerTick() );
    }
   

    void
    IsoSurfaceSAGE::search()
    {
      //  SysTime::SysClock start = SysTime::currentTicks();
      scan_yes = scan_no = 0;
  
      iso_value = isoval.get();
      value = tcl_value.get();
      scan = tcl_scan.get();
      visibility = tcl_visibility.get();
      bbox_visibility = tcl_bbox.get();
      reduce =  tcl_reduce.get();
      extract_all = tcl_all.get();
      int min_size = tcl_min_size.get() ? 2 : 1;

      sage->setParameters( scan, bbox_visibility, reduce, extract_all,
			   min_size);

      points = scinew GeomPts(2000);
      group=scinew GeomGroup;
      topobj=group;
  
      ScalarFieldHandle colorfield;
      ColorMapHandle cmap;
      int have_colorfield=incolorfield->get(colorfield);
      int have_colormap=incolormap->get(cmap);
      if(have_colormap && !have_colorfield){
	// Paint entire surface based on colormap
	topobj=scinew GeomMaterial(group, cmap->lookup(iso_value));
      } else if(have_colormap && have_colorfield){
	// Nothing - done per vertex
      } else {
	// Default material
	topobj=scinew GeomMaterial(group, iso_value < 800 ? flesh : bone);
      }
      

      // SEARCH 
      screen->clear();
      sage->search(iso_value, group, points);

      // OUTPUT
      if ( points_id ) {
	ogeom->delObj(points_id);
	points_id = 0;
      }
  
      if(surface_id ) 
	ogeom->delObj(surface_id);
  
      if ( group->size() == 0 && points->pts.size() == 0 ) {
	if ( !box_id ) {
	  GeomBox *box = scinew GeomBox( bmin, bmax, 1 );
	  GeomObj *bbox= scinew GeomMaterial( box, box_matl);
	  box_id = ogeom->addObj( bbox, box_name );
	}
      }
      else if ( box_id ) {
	ogeom->delObj(box_id);
	box_id = 0;
      }
      
      if ( group->size() == 0 ) {
	delete group;
	surface_id=0;
	//surface_id2 = 0;
      } else {
	if ( tcl_dl.get() )
	  topobj = scinew GeomDL( topobj );
	GeomBBoxCache *bbc = scinew GeomBBoxCache( topobj,
						   BBox( bmin, bmax) );
	surface_id=ogeom->addObj( bbc, surface_name );
      }
      if ( points->pts.size() > 0 ) {
	if ( points->pts.size() > 2000 )
	  printf("NOTE: there are %d points! \n",points->pts.size() );
	points_id =ogeom->addObj( scinew GeomMaterial( points, 
						       iso_value < 800 ? flesh 
						       : bone ),
				  "SAGE points");
      }
    }
    
    //  SysTime::SysClock end = SysTime::currentTicks();			
    //  printf("Scan: %d cells\n", statistics.extracted );
    //   printf("Scan : %d %d\n", scan_yes, scan_no );	
    
    //   printf(" Search Timers: \n\tinit %.3f  \n"
    // 	 "\tsearch %.3f (%.3f  %.3f) \n"
    // 	 "\tall %.3f\n ",
    //    	 (end-start -(end1-start1))*cycleval*1e-9,
    //    	 (end1-start1)*cycleval*1e-9, 
    //    	 vis_timer*cycleval*1e-9, extract_timer*cycleval*1e-9, 
    //    	 (end-start)*cycleval*1e-9);
    

int
IsoSurfaceSAGE::set_view( const View &view )
{
  double znear=MAXDOUBLE;
  double zfar=-MAXDOUBLE;

  Point eyep(view.eyep());
  Vector dir(view.lookat()-eyep);
  if(dir.length2() < 1.e-6) {
    printf("dir error in set view\n");
    return 0;
  }
  dir.normalize();
  double d=-Dot(eyep, dir);
  for(int ix=0;ix<2;ix++){
    for(int iy=0;iy<2;iy++){
      for(int iz=0;iz<2;iz++){
	Point p(ix?bmax.x():bmin.x(),
		iy?bmax.y():bmin.y(),
		iz?bmax.z():bmin.z());
	double dist=Dot(p, dir)+d;
	znear=Min(znear, dist);
	zfar=Max(zfar, dist);
      }
    }
  }

  if(znear <= 0){
    if(zfar <= 0){
      // Everything is behind us - it doesn't matter what we do
      znear=1.0;
      zfar=2.0;
    } else {
      znear=zfar*.001;
    }
  }

  sage->setView( view, znear, zfar, xres, yres );
  has_camera_view = true;

  return 1;
}


} // End namespace Yarden
