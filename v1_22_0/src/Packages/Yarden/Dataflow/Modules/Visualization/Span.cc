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

#include <Packages/Yarden/Core/Datatypes/SpanSpace.h>
#include <Core/Datatypes/ScalarField.h>
#include <Core/Datatypes/ScalarFieldRG.h> 

#include <Dataflow/Ports/ScalarFieldPort.h>
#include <Packages/Yarden/Dataflow/Ports/SpanPort.h>

#include <Core/GuiInterface/GuiVar.h>
#include <Core/GuiInterface/TCL.h>

#include <iostream>


namespace Yarden {
    
using namespace SCIRun;

    //using namespace Dataflow::Algorithms;
    
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


} // End namespace SCIRun
