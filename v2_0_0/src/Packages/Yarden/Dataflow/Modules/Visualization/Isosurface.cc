/*
 *  Isosurface.cc 
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

#include <Core/Containers/Array1.h>
#include <Core/Containers/Array2.h>

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


// SAGE
#include <Packages/Yarden/Core/Algorithms/Visualization/Sage.h>

// SpanSpace and Noise
#include <Packages/Yarden/Core/Datatypes/SpanSpace.h>
#include <Packages/Yarden/Core/Algorithms/Visualization/MCRGScan.h>
#include <Packages/Yarden/Core/Algorithms/Visualization/MCUG.h>
#include <Packages/Yarden/Core/Algorithms/Visualization/Noise.h>

#include <tcl.h>
#include <tk.h>
#include <math.h>
#include <iostream>
#include <sstream>
#include <strstream>
#include <values.h>


namespace Yarden {
    
using namespace SCIRun;
    
    SysTime::SysClock extract_timer, vis_timer;

    int bbox_count;
    int debug = 1;
    int scan_yes;
    int scan_no;
  

    GeometryData *gd;

    class NoAbort {
    private:
      Module *m;
    public:
      NoAbort( Module *m) : m(m) {}

      void update_progress(double p) { m->update_progress(p); }
      bool get_abort() { return false; }
    };
      
    //typedef SageBase<Module> SageAlg;
    typedef SageBase<NoAbort> SageAlg;
    
    typedef NoiseBase<Module> NoiseAlg;

    enum Algorithm {SAGE, NOISE };

    // Isosurface

    class Isosurface : public Module 
    {
      // input
      ScalarFieldIPort* infield;  // input scalar fields (bricks)
      ScalarFieldIPort* incolorfield;
      ColorMapIPort* incolormap;
      
      // input for Sage
      PathIPort* icam_view;

      // ouput
      GeometryOPort* ogeom;       // input from salmon - view point

      // UI variables
      GuiDouble isoval;
      GuiDouble isoval_min, isoval_max;
      GuiDouble tcl_red, tcl_green, tcl_blue;
      GuiInt tcl_dl;
      GuiInt tcl_bbox;
      GuiInt tcl_algorithm;

      // UI for Spanspace
      GuiInt tcl_span_region_set;
      GuiInt tcl_span_region_x0, tcl_span_region_y0;
      GuiInt tcl_span_region_x1, tcl_span_region_y1;
      GuiInt tcl_span_width, tcl_span_height;

      // UI for SAGE
      GuiInt tcl_value,tcl_visibility, tcl_scan;
      GuiInt tcl_reduce, tcl_all;
      GuiInt tcl_min_size, tcl_poll;
  
      // UI for NOISE
      /* none */


      // Global variables
      int field_generation;
      int box_id;
      int surface_id;

      MaterialHandle bone;
      MaterialHandle flesh;
      MaterialHandle matl;
  
      GeomGroup* group;
      GeomObj* topobj;

      Point bmin, bmax;
      int dx, dy, dz, dim;
      int sage_generation, noise_generation;
      double iso_value;

      GeomTrianglesP *triangles;
      GeomMaterial *surface;

      // SpanSpace
      int span_region_set;
      double span_region_min0, span_region_min1;
      double span_region_max0, span_region_max1;

      // Sage
      int value, bbox_visibility, visibility;
      int scan, count_values, extract_all;

      int points_id;
      MaterialHandle box_matl;
      MaterialHandle points_matl;

      Path       camv;
      bool has_camera_view;

      GeomPts* points;

      SageAlg *sage;
      Screen *screen;

      GeometryData *default_gd;
      int xres, yres;

      int reduce;

      // NOISE

      typedef unsigned char Byte;
      NoiseAlg *noise;
      Byte cmap[256][3];
      
    public:
      Isosurface( const clString& id);
      virtual ~Isosurface();
      
      virtual void execute();
      //virtual void do_execute();

    private:
      // SAGE
      void sage_init( ScalarFieldHandle );
      void run_sage();

      // NOISE
      void noise_init( ScalarFieldHandle );
      void run_noise();
      template <class T, class F> NoiseAlg *makeNoise( F *);

      void make_spanspace_image( ScalarFieldHandle & );
      int  make_span_hist( Array2<int> &, ScalarFieldHandle &);
      template <class T> int compute_span_hist(Array2<int> &, 
					       Array3<T> &, 
					       double );
      void init_cmap();

      void forward();
      int set_view( const View &);
    };
    
    extern "C" Module* make_Isosurface(const clString& id)
    {
      return scinew Isosurface(id);
    }
    
    static clString module_name("Isosurface");
    clString box_name("SageBox");
    clString surface_name;;
    
    Isosurface::Isosurface(const clString& id)
      : Module("Isosurface", id, Filter ), 
	isoval("isoval", id, this),
	isoval_min("isoval_min", id, this), 
	isoval_max("isoval_max", id, this),
	tcl_red("clr-r", id, this),
	tcl_green("clr-g", id, this),
	tcl_blue("clr-b", id, this),
	tcl_dl("dl", id, this),
	tcl_bbox("bbox", id, this), 
	tcl_algorithm("alg", id, this), 

	// SpanSpace UI
	
	tcl_span_region_set("span-region-set", id, this),
	tcl_span_region_x0("span-region-x0", id, this),
	tcl_span_region_y0("span-region-y0", id, this),
	tcl_span_region_x1("span-region-x1", id, this),
	tcl_span_region_y1("span-region-y1", id, this),
	tcl_span_width("span-width", id, this),
	tcl_span_height("span-height", id, this),

	// Sage UI
	tcl_value("value", id, this), 
	tcl_visibility("visibility", id, this),
	tcl_scan("scan", id, this),  
	tcl_reduce("reduce",id,this), 
	tcl_all("all",id,this),
	tcl_min_size("min_size", id, this),
	tcl_poll("poll", id, this)
      
	// NOISE UI
    {
      cerr << "ID = " << id << endl;
      char name[20];
      sscanf( id(), "Dataflow_Visualization_%s",name );
      surface_name = clString(name);
      box_name = surface_name;
      cerr << "name = " << surface_name << endl;

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
      field_generation = -1;

      // SAGE
      points_id = 0;
      box_id = 0;
      sage_generation = -1;

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

      // Noise
      noise = 0;
      noise_generation = -1;
      init_cmap();
    }

    Isosurface::~Isosurface()
    {
    }

    void
    Isosurface::execute()
    {
      SysTime::SysClock start = SysTime::currentTicks();

      ScalarFieldHandle scalar_field;

      if(!infield->get(scalar_field)) {
	error("No input field\n");
	return;
      }

      if ( scalar_field->generation != field_generation ) {
	make_spanspace_image( scalar_field);
	field_generation = scalar_field->generation;
      }

      span_region_set =  tcl_span_region_set.get();
      if ( span_region_set ) {
	double base = isoval_min.get();
	double factor = (isoval_max.get() - base) / tcl_span_width.get();
	cerr << "span region_set is on" << endl;
	span_region_min0 = base + factor*tcl_span_region_x0.get();
	span_region_min1 = base + factor*tcl_span_region_x1.get();
	if ( span_region_min0 > span_region_min1 ) 
	  swap( span_region_min0, span_region_min1);

	int h = tcl_span_height.get();
	span_region_max0 = base + factor*( h - tcl_span_region_y0.get());
	span_region_max1 = base + factor*( h - tcl_span_region_y1.get());
	if ( span_region_max0 > span_region_max1 ) 
	  swap( span_region_max0, span_region_max1);

	cerr << span_region_min0 << " <-> " << span_region_min1 << " x " 
	     << span_region_max0 << " <-> " << span_region_max1 << endl;
      }
      
      switch ( tcl_algorithm.get() ) {
      case 0:
	error("MC algorithm not implemented yet");
	break;
      case 1:
	cerr << "Using NOISE\n";
	if ( scalar_field->generation !=  noise_generation ) 
	  noise_init( scalar_field );
	else 
	  run_noise();
	break;
      case 2:
	cerr << "Using SAGE\n";
	if ( scalar_field->generation !=  sage_generation ) 
	  sage_init( scalar_field );
	else 
	  run_sage();
	break;
      default:
	error("Unknow Algorithm requested\n");
	break;
      }

      SysTime::SysClock end = SysTime::currentTicks();
      printf("Exec Timer: %.3f\n\n", (end-(long long)start) *SysTime::secondsPerTick() );
    }
   

    void
    Isosurface::sage_init ( ScalarFieldHandle scalar_field )
    {
      // create a new Sage algorithm
      NoAbort *fake = scinew NoAbort(this);
      SageAlg *tmp = SageAlg::make( scalar_field, fake );
      if ( !tmp) {
	error( "Can not work with this type of ScalarField");
	return;
      }
      
      if ( sage ) delete sage;
      sage = tmp;
      sage->setScreen ( screen );
      
      // save the current generation
      sage_generation = scalar_field->generation;
      
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

    void
    Isosurface::run_sage()
    {
      // Get View information 
	
      // first, check if we got a view in the input port
      PathHandle camera;
      double dummy;
      View view;
      if ( !icam_view->get( camera ) || !camera.get_rep() || !camera->get_keyF(0, view, dummy) || !set_view(view) ) {
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
      if ( span_region_set )
	sage->setRegion( span_region_min0, span_region_min1, 
			  span_region_max0, span_region_max1 );
      else
	sage->setRegion( false );

      points = scinew GeomPts(2000);
      group=scinew GeomGroup;
      
      screen->clear();
      sage->search(iso_value, group, points);

      forward();
      
      if ( has_camera_view ) 
	ogeom->setView( 0, view );
      else if ( tcl_poll.get() ) { 
	GeometryData *tmp_gd = ogeom->getData(0, GEOM_VIEW);
	if ((*gd->view == *tmp_gd->view) ) 
	  usleep( 50000 );
	want_to_execute();
      }
    }

    void
    Isosurface::forward()
    {
      //  SysTime::SysClock start = SysTime::currentTicks();
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
	Color c( tcl_red.get(), tcl_green.get(), tcl_blue.get());
	matl = scinew Material(c/6, c, c, 20 );    
	
	cerr << "TCL COLOR = " << tcl_red.get() << " " << tcl_green.get() 
	     << " " << tcl_blue.get() << endl;
	topobj=scinew GeomMaterial(group, matl );
	//topobj=scinew GeomMaterial(group, iso_value < 800 ? flesh : bone);
      }
      
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
      if ( points && points->pts.size() > 0 ) {
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
    Isosurface::set_view( const View &view )
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
    

    void 
    Isosurface::noise_init( ScalarFieldHandle field )
    {
      NoiseAlg *tmp = NULL;

      ScalarFieldRGBase *base = field->getRGBase();
      if ( base ) {
	if ( base->getRGDouble() ) 
	  tmp = makeNoise<double, ScalarFieldRGdouble>(base->getRGDouble());
	else if ( base->getRGFloat() ) 
	  tmp = makeNoise<float, ScalarFieldRGfloat>(base->getRGFloat());
	else if ( base->getRGInt() ) 
	  tmp = makeNoise<int, ScalarFieldRGint>(base->getRGInt());
	else if ( base->getRGShort() ) 
	  tmp = makeNoise<short, ScalarFieldRGshort>(base->getRGShort());
	else if ( base->getRGChar() ) 
	  tmp = makeNoise<char, ScalarFieldRGchar>(base->getRGChar());
	else if ( base->getRGUchar() ) 
	  tmp = makeNoise<uchar, ScalarFieldRGuchar>(base->getRGUchar());
	else {
	  error( "Can not work with this RG scalar field");
	  return;
	}
      }
      else if ( field->getUG() ) {
	// create a MC interpolant
	MCUG *mc =  new MCUG( field->getUG() );
	
	// create SpanSpace
	SpanSpace<double> *span = new SpanSpaceBuildUG( field->getUG() );
	
	// create Noise
	tmp = new Noise<double,MCUG,Module> (span, mc,this);
      }
      else {
	error("Unknow scalar field type"); 
	return;
      }
	 
       
      if (noise) delete noise;
      noise = tmp;

      // set the GUI variables
      double min, max;
      field->get_minmax( min, max );
      isoval_max.set(max);
      reset_vars();    
      
      // new noise is ready
      noise_generation =  field->generation;
    }

    void 
    Isosurface::run_noise()
    {
      double v = isoval.get() + 1e-6;
      cerr << "Run NOISE: " << v << endl;

      group = noise->extract( v );
      
      forward();
    }
    
    template <class T, class F>
    NoiseAlg *
    Isosurface::makeNoise( F *f )
    {
      // create a MC interpolant
      MCRGScan<F> *mc =  scinew MCRGScan<F>( f );
      
      // create SpanSpace
      SpanSpace<T> *span = scinew SpanSpaceBuild<T, F>( f );
      
      // create Noise
      return scinew Noise<T,MCRGScan<F>,Module> (span, mc,this);
    }

    void
    Isosurface::make_spanspace_image( ScalarFieldHandle &field)
    {
      float exponent = 0.15;

      clString filename = field->get_filename() + ".ppm";
      cerr << "make_spanspace_image: " << filename << endl;
      FILE *ppm = fopen (filename(), "r" );
      if ( !ppm ) {
	cerr << "make image" << endl;
	ppm = fopen (filename(), "w" );
	if ( !ppm ) {
	  error("can not open spanspace.ppm for writing");
	  return;
	}
	
	// make and save the spanspace image
	int w = tcl_span_width.get()/2;
	int h = tcl_span_height.get()/2;
	//      int mindim = min(w,h);
	
	Array2<int> hist( w, h );
	hist.initialize(0);
	
	int max = make_span_hist( hist, field );
	
	// create and save the image
	
	Array2<int> pixels(w,h);
	pixels.initialize(0);
	
	cerr << "max hist " << max << endl;
	float denom = 1.0 / (float)max;
	
	// find color for each point in the image
	for( int y = 0; y < h; y++ ) {
	  for( int x = 0; x < w; x++ ) {
	    if( hist(x,y) == 0 ) {
	      // do nothing, don't overwrite y=x line
	    } else {
	      // calculate index into colormap
	      int val = (int)(255.0 * powf( hist(x,y) * denom, exponent ));
	      pixels(x,y) = val;
	    }
	  }
	}
	
	
	// write the pixels to a ppm file
	printf("writing ppm file...\n");
	fprintf(ppm,"P6\n");
	fprintf(ppm,"%d %d 255\n", 2*w, 2*h);
	
	Byte *buffer = scinew Byte[3*2*w];
	
	for( int y = h-1; y >=0; y-- ) {
	  int i=0;
	  for( int x = 0; x < w; x++ ) {
	    if( y < 16 && x >= w-128 ) {
	      // draw colormap key
	      int pos = 2*(x+128 - w);
	      buffer[i++] = cmap[pos][0];
	      buffer[i++] = cmap[pos][1];
	      buffer[i++] = cmap[pos][2];
	      pos++;
	      buffer[i++] = cmap[pos][0];
	      buffer[i++] = cmap[pos][1];
	      buffer[i++] = cmap[pos][2];
	    } else {
	      // draw pixel
	      buffer[i++] = cmap[pixels(x,y)][0];
	      buffer[i++] = cmap[pixels(x,y)][1];
	      buffer[i++] = cmap[pixels(x,y)][2];
	      
	      buffer[i++] = cmap[pixels(x,y)][0];
	      buffer[i++] = cmap[pixels(x,y)][1];
	      buffer[i++] = cmap[pixels(x,y)][2];
	    }
	  }
	  fwrite( buffer, i, 1, ppm );
	  fwrite( buffer, i, 1, ppm );
	}
	fclose(ppm);
	
	cerr << "image done" << endl;
	delete buffer;
      }
      
      //std::strstream string;
      //string << id << " span-read-image "<< filename << '\0';
      //clString str (string.str().c_str());
      char *str = new char[strlen(filename())+strlen(id())+25];
      sprintf(str,"%s span-read-image %s",id(),filename());
      TCL::execute(str);
      delete[] str;
    }

    int
    Isosurface::make_span_hist( Array2<int> &hist, ScalarFieldHandle &field )
    {
      int max = 0;
      double fmin, fmax;
      field->get_minmax( fmin, fmax );

      cerr << "field min/max = " << fmin << " " << fmax << endl;
      ScalarFieldRGBase *base = field->getRGBase();
      if ( base ) {
	if ( base->getRGDouble() ) 
	  max = compute_span_hist<double>( hist, 
					   base->getRGDouble()->grid, 
					   fmin);
	else if ( base->getRGFloat() ) 
	  max = compute_span_hist<float>( hist, 
					  base->getRGFloat()->grid,
					  fmin);
	else if ( base->getRGInt() ) 
	  max = compute_span_hist<int>( hist, 
					base->getRGInt()->grid, 
					fmin);
	else if ( base->getRGShort() ) 
	  max = compute_span_hist<short>( hist, 
					  base->getRGShort()->grid, 
					  fmin);
	else if ( base->getRGUchar() ) 
	  max = compute_span_hist<unsigned char>( hist, 
						  base->getRGUchar()->grid,
						  fmin);
	else if ( base->getRGChar() ) 
	  max = compute_span_hist<char>( hist, 
					 base->getRGChar()->grid,
					 fmin);
	else {
	  error( "Can not work with this RG scalar field");
	  return -1;
	}
      }
      else if ( field->getUG() ) {
	error( "Can not work with this RG scalar field");
	return -1;
      }

      return max;
    }

    template <class T>
    int
    Isosurface::compute_span_hist( Array2<int> &hist, 
				   Array3<T> &grid, 
				   double gmin)
    {
      int w = hist.dim1();
      int h = hist.dim2();
      int len = max(w,h);

      int nx = grid.dim1();
      int ny = grid.dim2();
      int nz = grid.dim3();
      
      double factor = 255.0/len;
      int hmax = 0;

      cerr << "Compute hist" << endl;
      for( int z = 0; z < nz-1; z++ ) {
	for( int y = 0; y < ny-1; y++ ) {
	  for( int x = 0; x < nx-1; x++ ) {
	    // find min and max of this cell
	    T min = grid(x,y,z);
	    T max = min;
	    
	    for (int i=0; i<2; i++)
	      for (int j=0; j<2; j++)
		for (int k=0; k<2; k++) {
		  T v = grid(x+i,y+j,z+k);
		  if ( v < min ) min = v;
		  else if (v > max ) max = v;
		}
	    
	    // increment histogram
	    int minidx = (int)(factor * (min-gmin));
	    int maxidx = (int)(factor * (max-gmin));
	    
	    if ( minidx < w && maxidx < h ) {
	      int v = hist(minidx,maxidx)++;
	      if ( v > hmax ) hmax = v;
	    }
	  }
	}
      }
      return hmax+1;
    }

    void
    Isosurface::init_cmap()
    {
      int i;
      float deltar, deltag, deltab;
      float currr, currg, currb;
      bzero( cmap, 256*3*sizeof(Byte) );

      // black -> blue
      cmap[0][0] = cmap[0][1] = cmap[0][2] = Byte(0);
      deltab = .75 / 63.0;
      for( i = 1; i < 64; i++ ) {
	cmap[i][2] = Byte(255*i * deltab);
      }
      currr = 0.0;
      currg = 0.0;
      currb = 0.75;
      
      // blue -> red
      deltar = 1.0 / 63.0;
      deltag = .25 / 63.0;
      deltab = -.50 / 63.0;
      for( i = 0; i < 64; i++ ) {
	cmap[i+64][0] = Byte(255*(currr + i * deltar));
	cmap[i+64][1] = Byte(255*(currg + i * deltag));
	cmap[i+64][2] = Byte(255*(currb + i * deltab));
      }
      currr = 1.0;
      currg = .25;
      currb = .25;
      
      // red -> yellow
      deltag = .75 / 63.0;
      for( i = 0; i < 64; i++ ) {
	cmap[i+128][0] = Byte(255*currr);
	cmap[i+128][1] = Byte(255*(currg + i * deltag));
	cmap[i+128][2] = Byte(255*currb);
      }
      currg = 1.0;
      
      // yellow -> white
      deltab = .75 / 63.0;
      for( i = 0; i < 64; i++ ) {
	cmap[i+192][0] = Byte(255*currr);
	cmap[i+192][1] = Byte(255*currg);
	cmap[i+192][2] = Byte(255*(currb + i * deltab));
      }
    }

    

} // End namespace Yarden
