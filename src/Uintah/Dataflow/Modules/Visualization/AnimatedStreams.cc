/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/



/*
 * AnimatedStreams.cc
 *
 * Simple interface to volume rendering stuff
 */

#include "AnimatedStreams.h"

#include <Core/Containers/Array1.h>
#include <Core/Geom/ColorMap.h>
#include <Core/Datatypes/Field.h>
#include <Core/Thread/CrowdMonitor.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/Ports/ColorMapPort.h>
#include <Dataflow/Network/Ports/GeometryPort.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Dataflow/Widgets/PointWidget.h>


#include <iostream>
#include <algorithm>

using std::cerr;
using std::endl;
using std::hex;
using std::dec;

namespace Uintah{

using namespace SCIRun;

static string control_name("Stream Control Widget");
			 
  DECLARE_MAKER(AnimatedStreams)

AnimatedStreams::AnimatedStreams(GuiContext* ctx)
  : Module("AnimatedStreams", ctx, Filter, "Visualization", "Uintah"),
    vf(0), anistreams(0), 
    generation(-1), timestep(-1),
    pause(get_ctx()->subVar("pause")),
    normals(get_ctx()->subVar("normals")),
    lighting(get_ctx()->subVar("lighting")),
    normal_method(get_ctx()->subVar("normal_method")),
    use_deltat(get_ctx()->subVar("use_deltat")),
    stepsize(get_ctx()->subVar("stepsize")),
    linewidth(get_ctx()->subVar("linewidth")),
    control_lock("AnimatedStreams position lock"),
    control_widget(0), control_id(-1),
    mutex("Animated Streams")
{

}

AnimatedStreams::~AnimatedStreams()
{

}
void AnimatedStreams::widget_moved(bool,BaseWidget*)
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
    GeomHandle w=control_widget->GetWidget();
    control_id = ogeom->addObj( w, control_name, &control_lock);
  }

  mutex.lock();

  if( !anistreams ){
    anistreams = new GLAnimatedStreams( field, cmap);
    ogeom->addObj( anistreams, "AnimatedStreams");
    vf = field;
    map = cmap;
  } else {
    // determine if the field has changed
    if( vf.get_rep() != field.get_rep() ){
      vf = field;
      // We need to figure out why the field has changed. If it changed because
      // the grid or variable changed we need to call SetVectorField
      // which will resets the the streams.
      // If the grid didn't change, but the time step changed
      // then we can assume the field dimensions are the same and call
      // ChangeVectorField which doesn't reset the streams.  If we can't tell
      // what the field generation or variable are, call SetVectorField,
      // because it probably didn't come from a Uintah::DataArchive.
      string new_varname("This is not a valid name");
      int new_generation = -1;
      int new_timestep = -1;
      // check for the properties
      if (vf->get_property("varname",new_varname) &&
	  vf->get_property("generation",new_generation) &&
	  vf->get_property("timestep",new_timestep)) {
	// check to make sure the variable name and generation are the same
	if (new_varname == varname && new_generation == generation) {
	  if (new_timestep != timestep)
	    // same field, different timestep -> swap out the field
		anistreams->ChangeVectorField( field );
		timestep = new_timestep;
	} else {
	  // we have a different field
	  anistreams->SetVectorField( field );
	  varname = new_varname;
	  generation = new_generation;
	  timestep = new_timestep;
	}
      } else {
	// all the necessary properties don't exist
	anistreams->SetVectorField( field );
	// make sure these parameters get reset
	varname = string("This is not a valid name");
	generation = timestep = -1;
      }
    } // else the fields are the same and do nothing

    if( cmap.get_rep() != map.get_rep() ){
      anistreams->SetColorMap( cmap );
      map = cmap;
    }
  }

  double dt = -1;
  if (vf->get_property("delta_t",dt) && dt > 0) {
    cerr << "Delta_t = " << dt << "\n";
    anistreams->SetDeltaT(dt);
  } else {
    anistreams->SetDeltaT(-1);
    use_deltat.set(0);
  }

  anistreams->Pause( (bool)(pause.get()) );
  anistreams->Normals( (bool)(normals.get()) );
  anistreams->Lighting( (bool)(lighting.get()) );
  anistreams->SetStepSize( stepsize.get() );
  anistreams->SetLineWidth( linewidth.get());
  anistreams->SetNormalMethod( normal_method.get() );
  anistreams->UseDeltaT( (bool)use_deltat.get() );
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

void AnimatedStreams::tcl_command( GuiArgs& args, void* userdata) {
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





