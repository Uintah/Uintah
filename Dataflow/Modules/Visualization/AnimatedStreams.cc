
/*
 * AnimatedStreams.cc
 *
 * Simple interface to volume rendering stuff
 */

#include "AnimatedStreams.h"

#include <Core/Containers/Array1.h>
#include <Dataflow/Network/Module.h>
#include <Core/Datatypes/ColorMap.h>
#include <Dataflow/Ports/ColorMapPort.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/Field.h>
#include <Core/Thread/CrowdMonitor.h>

#include <Dataflow/Widgets/PointWidget.h>

#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>

#include <iostream>
#include <algorithm>

using std::cerr;
using std::endl;
using std::hex;
using std::dec;

namespace Uintah{

using namespace SCIRun;

static string control_name("Stream Control Widget");
			 
extern "C" Module* make_AnimatedStreams( const string& id) {
  return new AnimatedStreams(id);
}


AnimatedStreams::AnimatedStreams(const string& id)
  : Module("AnimatedStreams", id, Filter, "Visualization", "Uintah"),
    generation(-1), timestep(-1),
    pause("pause",id, this),
    normals("normals", id, this),
    stepsize("stepsize", id, this),
    linewidth("linewidth", id, this),
    control_lock("AnimatedStreams position lock"),
    control_widget(0), control_id(-1),
    anistreams(0), vf(0),
    mutex("Animated Streams")
{

}

AnimatedStreams::~AnimatedStreams()
{

}
void AnimatedStreams::widget_moved(int)
{
  if( anistreams ){
      anistreams->SetWidgetLocation(control_widget->ReferencePoint());
    }
}

void AnimatedStreams::execute(void)
{
  // Create the input ports
  infield = (FieldIPort *) get_iport("Vector Field");
  incolormap= (ColorMapIPort *) get_iport("ColorMap");
  // Create the output port
  ogeom = (GeometryOPort *) get_oport("Geometry");

  FieldHandle field;
  if (!infield->get(field)) {
    return;
  }
  else if (!field.get_rep()) {
    return;
  }
  
  static ColorMapHandle map = 0;
  ColorMapHandle cmap;
  if( !incolormap->get(cmap)){
    return;
  }


  if(!control_widget){
    control_widget=scinew PointWidget(this, &control_lock, 0.2);
    
    BBox bb;
    Point Smin, Smax;
    bb = field->mesh()->get_bounding_box();
    Smin = bb.min(); Smax = bb.max();
    double max =  std::max(Smax.x() - Smin.x(), Smax.y() - Smin.y());
    max = std::max( max, Smax.z() - Smin.z());
    control_widget->SetPosition(Interpolate(Smin,Smax,0.5));
    control_widget->SetScale(max/80.0);
    GeomObj *w=control_widget->GetWidget();
    control_id = ogeom->addObj( w, control_name, &control_lock);
  }

  mutex.lock();

  if( !anistreams ){
    anistreams = new GLAnimatedStreams(0x12345676,
				       field,
				       cmap);
    ogeom->addObj( anistreams, "AnimatedStreams");
    vf = field;
    map = cmap;
  } else {
    // determine if the field has changed
    if( vf.get_rep() != field.get_rep() ){
      vf = field;
      // We need to figure out why the field has changed. If it changed because
      // the grid changed we need to call SetVectorField which will resets the
      // the streams.  If the grid didn't change, but the time step changed
      // then we can assume the field dimensions are the same and call
      // ChangeVectorField which doesn't reset the streams.  If we can't tell
      // what the field generation is, call SetVectorField, because it
      // probably didn't come from a Uintah::DataArchive.
      int new_generation = -1;
      if (vf->get("generation",new_generation)) {
	// generation property found, check now for timestep property
	int new_timestep = -1;
	if (vf->get("timestep",new_timestep)) {
	  // now check for sameness of generation and timestep
	  if (new_generation == generation) {
	    if (new_timestep != timestep) {
	      // same generation different timestep, swap out the vector field
	      anistreams->ChangeVectorField( field );
	      timestep = new_timestep;
	    } // else do nothing (same generation and timestep)
	  } else {
	    // new DataArchive, set the vector field
	    anistreams->SetVectorField( field );
	    generation = new_generation;
	    timestep = new_timestep;
	  }
	} else {
	  // this is weird, the generation PropertyManager was found, but
	  // not the timestep one.  Set the vector field, and print out
	  // a warning.
	  cerr << "WARNING:AnimatedStreams::execute:generation PropertyManager was found, but not the timestep.  Using new vector field.\n";
	  anistreams->SetVectorField( field );
	  // make sure these parameters get reset
	  timestep = -1;
	}    
      } else {
	// generation parameter not found, set the vector field
	anistreams->SetVectorField( field );
	// make sure these parameters get reset
	generation = timestep = -1;
      }
    } // else the fields are the same and do nothing

    if( cmap.get_rep() != map.get_rep() ){
      anistreams->SetColorMap( cmap );
      map = cmap;
    }
  }
 

  anistreams->Pause( (bool)(pause.get()) );
  anistreams->Normals( (bool)(normals.get()) );
  anistreams->SetStepSize( stepsize.get() );
  anistreams->SetLineWidth( linewidth.get());
  //anistreams->UseWidget(true);
  if( !pause.get() ) {
    want_to_execute();
    ogeom->flushViews();
    mutex.unlock();
    return;
  }

  anistreams->IncrementFlow();
    
  ogeom->flushViews();
  
  mutex.unlock();
}

void AnimatedStreams::tcl_command( TCLArgs& args, void* userdata) {
  if(args.count() < 2) {
    args.error("Streamline needs a minor command");
    return;
  }
  if(args[1] == "reset_streams") {
    anistreams->ResetStreams();
    want_to_execute();
  }
  else if(args[1] == "update_linewidth") {
    anistreams->SetLineWidth( linewidth.get());
    ((GeometryOPort *) get_oport("Geometry"))->flushViews();
  }
  else {
    Module::tcl_command(args, userdata);
  }
}

} // End namespace Uintah





