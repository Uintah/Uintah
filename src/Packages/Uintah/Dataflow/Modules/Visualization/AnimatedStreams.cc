
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
    if( vf.get_rep() != field.get_rep() ){
      anistreams->SetVectorField( field );
      vf = field;
    }
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
      
    
  ogeom->flushViews();
  
  mutex.unlock();
}

} // End namespace Uintah


