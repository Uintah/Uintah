
/*
 * AnimatedStreams.cc
 *
 * Simple interface to volume rendering stuff
 */

#include "AnimatedStreams.h"

#include <SCICore/Containers/Array1.h>
#include <PSECore/Dataflow/Module.h>
#include <SCICore/Datatypes/ColorMap.h>
#include <PSECore/Datatypes/ColorMapPort.h>
#include <PSECore/Datatypes/GeometryPort.h>
#include <PSECore/Datatypes/VectorFieldPort.h>
#include <SCICore/Datatypes/VectorField.h>
#include <SCICore/Thread/CrowdMonitor.h>

#include <PSECore/Widgets/PointWidget.h>
#include <SCICore/Geom/GeomTriangles.h>

#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLvar.h>

#include <iostream>
#include <ios>
#include <algorithm>

using std::cerr;
using std::endl;
using std::hex;
using std::dec;

namespace Kurt {
namespace Modules {


using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;

static clString control_name("Stream Control Widget");
			 
extern "C" Module* make_AnimatedStreams( const clString& id) {
  return new AnimatedStreams(id);
}


AnimatedStreams::AnimatedStreams(const clString& id)
  : Module("AnimatedStreams", id, Filter),
    pause("pause",id, this),
    normals("normals", id, this),
    stepsize("stepsize", id, this),
    linewidth("linewidth", id, this),
    control_lock("AnimatedStreams position lock"),
    control_widget(0), control_id(-1),
    anistreams(0), vf(0),
    mutex("Animated Streams")
{
  // Create the input ports
  infield = scinew VectorFieldIPort( this, "VectorField",
				     VectorFieldIPort::Atomic);
  add_iport(infield);
  incolormap=scinew  
    ColorMapIPort(this, "ColorMap", ColorMapIPort::Atomic);
    
  add_iport(incolormap);
					
  // Create the output port
  ogeom = scinew GeometryOPort(this, "Geometry", 
			       GeometryIPort::Atomic);
  add_oport(ogeom);

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
  VectorFieldHandle field;
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
    
    Point Smin, Smax;
    field->get_bounds(Smin, Smax);

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

} // End namespace Modules
} // End namespace Uintah


