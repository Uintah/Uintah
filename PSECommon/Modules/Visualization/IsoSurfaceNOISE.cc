/*
 *  IsoSurfaceNOISE.cc:  The NOISE Module
 *
 *  Written by:
 *   Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *   Nov. 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#include <stdio.h>
#include <time.h>
#include <SCICore/Containers/Queue.h>  
#include <SCICore/Persistent/Pstreams.h>          
#include <SCICore/Geometry/BBox.h>
#include <PSECore/Datatypes/SpanSpace.h>
#include <SCICore/Datatypes/TriSurface.h>
#include <SCICore/Datatypes/ColorMap.h>
#include <SCICore/Geom/BBoxCache.h>
#include <SCICore/Geom/GeomGroup.h>
#include <SCICore/Geom/Material.h>
#include <SCICore/Malloc/Allocator.h>
//#include <SCICore/Math/Expon.h>
//#include <SCICore/Math/MiscMath.h>
#include <SCICore/Thread/Parallel.h>
#include <SCICore/Thread/Thread.h>

#include <SCICore/Datatypes/ScalarField.h>

#include <PSECore/Datatypes/ScalarFieldPort.h>
#include <PSECore/Datatypes/SurfacePort.h>
#include <PSECore/Datatypes/ColorMapPort.h>
#include <PSECore/Datatypes/GeometryPort.h>

//#include <SCICore/TclInterface/TCLTask.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <SCICore/TclInterface/TCL.h>
//#include <tcl.h>
//#include <tk.h>

#include <iostream>

#include <PSECore/Datatypes/SpanSpace.h>
#include <PSECommon/Algorithms/Visualization/MCRGScan.h>
#include <PSECommon/Algorithms/Visualization/MCUG.h>
#include <PSECommon/Algorithms/Visualization/Noise.h>



namespace PSECommon {
  namespace Modules {
    
    using namespace SCICore::TclInterface;
    using namespace SCICore::Containers;
    using namespace SCICore::GeomSpace;
    using namespace SCICore::Geometry;
    using namespace PSECore::Dataflow;
    using namespace PSECore::Datatypes;
    using namespace PSECommon::Algorithms;
    
    class IsoSurfaceNOISE : public Module 
    {
      // input
      ScalarFieldIPort* infield;  // input scalar fields (bricks)
      ScalarFieldIPort* incolorfield;
      ColorMapIPort* incolormap;

      // ouput
      GeometryOPort* ogeom;       // input from salmon - view point

      // UI variables
      double v; // isovalue
      TCLdouble isoval;
      TCLdouble isoval_min, isoval_max;
      TCLint    tcl_bbox;
      TCLint    tcl_np;

      NoiseBase<Module> *noise;

      int field_generation;
      int surface_id;
      
      ScalarFieldHandle field;
      MaterialHandle matl;
      GeomGroup *group;

    public:
      IsoSurfaceNOISE(const clString& id);
      virtual ~IsoSurfaceNOISE();

      void execute();
      void new_field();
      void extract();
      template <class T, class F> NoiseBase<Module> *makeNoise( F *);
    };
      
    
    extern "C" Module* make_IsoSurfaceNOISE(const clString& id)
    {
      return scinew IsoSurfaceNOISE(id);
    }

    const double epsilon = 1.e-8;

    static clString module_name("IsoSurfaceNOISE");
    static clString surface_name("NoiseSurface");

    IsoSurfaceNOISE::IsoSurfaceNOISE(const clString& id) :
      Module("IsoSurfaceNOISE", id, Filter),
      isoval("isoval", id, this ),
      isoval_min("isoval_min", id, this ),
      isoval_max("isoval_max", id, this ),
      tcl_bbox("bbox", id, this),
      tcl_np("np",id,this)
    {
      // Create input ports
      infield=scinew ScalarFieldIPort(this, "Field", ScalarFieldIPort::Atomic);
      add_iport(infield);
      
      incolorfield=scinew ScalarFieldIPort(this, "Color Field", 
					   ScalarFieldIPort::Atomic);
      add_iport(incolorfield);
      
      incolormap=scinew ColorMapIPort(this, "Color Map",ColorMapIPort::Atomic);
      add_iport(incolormap);
      
      
      // Create output port
      ogeom = scinew GeometryOPort( this, "Geometry", GeometryIPort::Atomic);
      add_oport(ogeom);
      
      matl=scinew Material(Color(0,0,0), Color(0,.8,0), Color(.7,.7,.7), 20);
      
      surface_id = 0;
      field_generation = -1;
    }
    


    IsoSurfaceNOISE::~IsoSurfaceNOISE()
    {
    }
    

    void
    IsoSurfaceNOISE::execute()
    {
      if(!infield->get(field)) {
	printf("IsoSurfaceNoise: no field\n");
	return;
      }
      
      if ( field->generation !=  field_generation ) 
	new_field();
      else 
	extract();
    }

    void
    IsoSurfaceNOISE::new_field() 
    {
      NoiseBase<Module> *tmp = NULL;

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
      field_generation =  field->generation;
    }

    void
    IsoSurfaceNOISE::extract()
    {
      double v = isoval.get() + epsilon;

      GeomGroup *group = noise->extract( v );

      if ( surface_id ) {
	ogeom->delObj(surface_id);
      }

      if ( group->size() == 0 ) {
	printf("empty group\n");
	delete group;
	surface_id = 0;
      }
      else {
	GeomObj *surface=group;

	ScalarFieldHandle colorfield;
	int have_colorfield=incolorfield->get(colorfield);
	ColorMapHandle cmap;
	int have_colormap=incolormap->get(cmap);
      
	if(have_colormap && !have_colorfield){
	  // Paint entire surface based on colormap
	  surface = scinew GeomMaterial( group, cmap->lookup(v));
	} else if(have_colormap && have_colorfield){
	  // Nothing - done per vertex
	} else {
	  // Default material
	  surface = scinew GeomMaterial( group, matl );
	}
	

	// send the bbox to Salmon
 	Point bmin, bmax;
 	field->get_bounds( bmin, bmax );
  	surface_id = ogeom->addObj( scinew GeomBBoxCache( surface, 
  						       BBox( bmin, bmax )),
  				    "NOISE");
      }
      
      ogeom->flushViews();
    }


  template <class T, class F>
  NoiseBase<Module> *
  IsoSurfaceNOISE::makeNoise( F *f )
  {
    // create a MC interpolant
    MCRGScan<F> *mc =  new MCRGScan<F>( f );
    
    // create SpanSpace
    SpanSpace<T> *span = new SpanSpaceBuild<T, F>( f );
    
    // create Noise
    return new Noise<T,MCRGScan<F>,Module> (span, mc,this);
  }

  } // namespace Modules
} // namespace PSECommon
