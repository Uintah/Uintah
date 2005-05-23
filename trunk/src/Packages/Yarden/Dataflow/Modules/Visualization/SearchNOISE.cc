/*
 *  SearchNOISE.cc:  Search using the NOISE Module
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
#include <Core/Persistent/Pstreams.h>          
#include <Core/Geometry/BBox.h>
#include <Packages/Yarden/Core/Datatypes/SpanSpace.h>
#include <Core/Datatypes/TriSurfFieldace.h>
#include <Core/Geom/ColorMap.h>
#include <Core/Geom/BBoxCache.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/Material.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Thread/Parallel.h>
#include <Core/Thread/Thread.h>

#include <Core/Datatypes/ScalarField.h>

#include <Dataflow/Ports/ScalarFieldPort.h>
#include <Packages/Yarden/Dataflow/Ports/SpanPort.h>
#include <Dataflow/Ports/SurfacePort.h>
#include <Dataflow/Ports/ColorMapPort.h>
#include <Dataflow/Ports/GeometryPort.h>

//#include <Core/GuiInterface/TCLTask.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/GuiInterface/TCL.h>
//#include <tcl.h>
//#include <tk.h>

#include <iostream>

#include <Packages/Yarden/Core/Datatypes/SpanSpace.h>
#include <Packages/Yarden/Core/Algorithms/Visualization/MCRGScan.h>
#include <Packages/Yarden/Core/Algorithms/Visualization/MCUG.h>
#include <Packages/Yarden/Core/Algorithms/Visualization/Noise.h>



namespace Yarden {
    
using namespace SCIRun;

    class SearchNOISE : public Module 
    {
      // input
      ScalarFieldIPort *infield;
      SpanUniverseIPort *inuniverse;
      ScalarFieldIPort* incolorfield;
      ColorMapIPort* incolormap;

      // ouput
      GeometryOPort* ogeom;       // input from salmon - view point

      // UI variables
      double v; // isovalue
      GuiDouble isoval;
      GuiDouble isoval_min, isoval_max;
      GuiInt    tcl_bbox;
      GuiInt    tcl_np;

      NoiseBase<Module> *noise;

      int field_generation, universe_generation;
      int surface_id;
      
      ScalarFieldHandle field;
      SpanUniverseHandle universe;
      MaterialHandle matl;
      GeomGroup *group;

    public:
      SearchNOISE(const clString& id);
      virtual ~SearchNOISE();

      void execute();
      void new_field();
      void extract();
      template <class T, class F> NoiseBase<Module> *makeNoise( F *);
    };
      
    
    extern "C" Module* make_SearchNOISE(const clString& id)
    {
      return scinew SearchNOISE(id);
    }

    const double epsilon = 1.e-8;

    static clString module_name("SearchNOISE");
    static clString surface_name("NoiseSurface");

    SearchNOISE::SearchNOISE(const clString& id) :
      Module("SearchNOISE", id, Filter),
      isoval("isoval", id, this ),
      isoval_min("isoval_min", id, this ),
      isoval_max("isoval_max", id, this ),
      tcl_bbox("bbox", id, this),
      tcl_np("np",id,this)
    {
      // Create input ports
      infield=scinew ScalarFieldIPort(this, "ScalarField",
				      ScalarFieldIPort::Atomic);
      add_iport(infield);

      inuniverse=scinew SpanUniverseIPort(this, "SpanUniverse",
				      SpanUniverseIPort::Atomic);
      add_iport(inuniverse);
      
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
      universe_generation = -1;
    }
    


    SearchNOISE::~SearchNOISE()
    {
    }
    

    void
    SearchNOISE::execute()
    {
      if(!infield->get(field)) {
	printf("SearchNoise: no field\n");
	return;
      }

      if(!inuniverse->get(universe)) {
	printf("SearchNoise: no span universe\n");
	return;
      }
      
      if ( field->generation != field_generation || 
	   universe->generation !=  universe_generation )
	new_field();
      else 
	extract();
    }


    void
    SearchNOISE::new_field() 
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
	
	// select SpanSpace
	SpanSpace<double> *span = (SpanSpace<double> *) universe->space[0];

	// create Noise
	tmp = new Noise<double,MCUG,Module> (span, mc, this);
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
      universe_generation =  universe->generation;
    }

    void
    SearchNOISE::extract()
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
  SearchNOISE::makeNoise( F *f )
  {
    // create a MC interpolant
    MCRGScan<F> *mc =  scinew MCRGScan<F>( f );
    
    // select SpanSpace
    SpanSpace<T> *span = (SpanSpace<T> *) universe->space[0];
    
    // create Noise
    return scinew Noise<T,MCRGScan<F>,Module> (span, mc,this);
  }

} // End namespace Yarden
