/*
 *  Span.cc:  The NOISE Module
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
#include <SCICore/Thread/Parallel.h>
#include <SCICore/Thread/Thread.h>

#include <PSECore/Datatypes/SpanSpace.h>
#include <SCICore/Datatypes/ScalarField.h>
#include <SCICore/Datatypes/ScalarFieldRGdouble.h> 
#include <SCICore/Datatypes/ScalarFieldRGfloat.h> 
#include <SCICore/Datatypes/ScalarFieldRGint.h> 
#include <SCICore/Datatypes/ScalarFieldRGshort.h> 
#include <SCICore/Datatypes/ScalarFieldRGushort.h> 
#include <SCICore/Datatypes/ScalarFieldRGchar.h> 
#include <SCICore/Datatypes/ScalarFieldRGuchar.h> 

#include <PSECore/Datatypes/ScalarFieldPort.h>
#include <PSECore/Datatypes/SpanPort.h>

#include <SCICore/TclInterface/TCLvar.h>
#include <SCICore/TclInterface/TCL.h>

#include <iostream>


namespace PSECommon {
  namespace Modules {
    
    using namespace SCICore::TclInterface;
    using namespace SCICore::Containers;
    using namespace SCICore::GeomSpace;
    using namespace SCICore::Geometry;
    using namespace PSECore::Dataflow;
    using namespace PSECore::Datatypes;
    //using namespace PSECommon::Algorithms;
    
    class Span : public Module 
    {
      // input
      ScalarFieldIPort* infield;  // input scalar fields (bricks)

      // output
      SpanUniverseOPort* ospan;       // input from salmon - view point

      
      ScalarFieldHandle field;
      SpanUniverseHandle universe;

      int field_generation;

    public:
      Span(const clString& id);
      virtual ~Span();

      void execute();
      void new_field();
    };
      
    
    extern "C" Module* make_Span(const clString& id)
    {
      return scinew Span(id);
    }

    static clString module_name("Span");

    Span::Span(const clString& id) :
      Module("Span", id, Filter)
    {
      // Create input ports
      infield=scinew ScalarFieldIPort(this, "Field", ScalarFieldIPort::Atomic);
      add_iport(infield);
      
      
      
      // Create output port
      ospan = scinew SpanUniverseOPort( this, "SpanUniverse", 
					SpanUniverseIPort::Atomic);
      add_oport(ospan);
      
      field_generation = -1;
    }
    


    Span::~Span()
    {
    }
    

    void
    Span::execute()
    {
      if(!infield->get(field)) {
	printf("Span: no field\n");
	return;
      }
      
      if ( field->generation !=  field_generation ) 
	new_field();
      ospan->send( universe );
    }

    void
    Span::new_field() 
    {
      SpanUniverseHandle tmp  = scinew SpanUniverse( field );
      
      ScalarFieldRGBase *base = field->getRGBase();
      if ( base ) {
	if ( base->getRGDouble() ) 
	  tmp->add( scinew SpanSpaceBuild<double, ScalarFieldRGdouble>
		    ( base->getRGDouble() ));
	else if ( base->getRGFloat() ) 
	  tmp->add( scinew SpanSpaceBuild<float, ScalarFieldRGfloat>
		    (base->getRGFloat()));
	else if ( base->getRGInt() ) 
	  tmp->add( scinew SpanSpaceBuild<int, ScalarFieldRGint>
		    (base->getRGInt()));
	else if ( base->getRGShort() ) 
	  tmp->add( scinew SpanSpaceBuild<short, ScalarFieldRGshort>
		    (base->getRGShort()));
	else if ( base->getRGChar() ) 
	  tmp->add( scinew SpanSpaceBuild<char, ScalarFieldRGchar>
		    (base->getRGChar()));
	else if ( base->getRGUchar() ) 
	  tmp->add( scinew SpanSpaceBuild<uchar, ScalarFieldRGuchar>
		    (base->getRGUchar()));
	else {
	  error( "Can not work with this RG scalar field");
	  return;
	}
      }
      else if ( field->getUG() ) {
	tmp->add( scinew SpanSpaceBuildUG( field->getUG() ));
      }
      else {
	error("Unknow scalar field type"); 
	return;
      }
	 
      universe = tmp;

      // new universe is ready
      field_generation =  field->generation;
    }


  } // namespace Modules
} // namespace PSECommon
