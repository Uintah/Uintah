
/*
 * TexCuttingPlanes.cc
 *
 * Simple interface to volume rendering stuff
 */

#include "TexCuttingPlanes.h"

#include <SCICore/Containers/Array1.h>
#include <PSECore/Dataflow/Module.h>
#include <SCICore/Datatypes/ColorMap.h>
#include <PSECore/Datatypes/ColorMapPort.h>
#include <PSECore/Datatypes/GeometryPort.h>
#include <SCICore/Datatypes/ScalarFieldRGuchar.h>

#include <SCICore/Geom/GeomTriangles.h>

#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <SCICore/Thread/CrowdMonitor.h>

#include <PSECore/Widgets/PointWidget.h>
#include <iostream>
#include <algorithm>
#include <Kurt/Datatypes/VolumeUtils.h>



namespace Kurt {
namespace Modules {

using namespace Kurt::Datatypes;
using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace PSECore::Widgets;
using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;
using namespace SCICore::Geometry;
using namespace SCICore::Math;
using SCICore::Datatypes::ScalarFieldRGuchar;
using std::cerr;


static clString control_name("Control Widget");
			 
extern "C" Module* make_TexCuttingPlanes( const clString& id) {
  return new TexCuttingPlanes(id);
}


TexCuttingPlanes::TexCuttingPlanes(const clString& id)
  : Module("TexCuttingPlanes", id, Filter), 
  drawX("drawX", id, this),
  drawY("drawY", id, this),
  drawZ("drawZ", id, this),
  drawView("drawView", id, this),
  control_lock("TexCuttingPlanes resolution lock"),
  control_widget(0), control_id(-1),
  volren(0), tex(0)
{
  // Create the input ports
  intexture = scinew GLTexture3DIPort( this, "GL Texture",
				     GLTexture3DIPort::Atomic);
  add_iport(intexture);
  incolormap=scinew  
    ColorMapIPort(this, "Color Map", ColorMapIPort::Atomic);
    
  add_iport(incolormap);
					
  // Create the output port
  ogeom = scinew GeometryOPort(this, "Geometry", 
			       GeometryIPort::Atomic);
  add_oport(ogeom);

}

TexCuttingPlanes::~TexCuttingPlanes()
{

}
void 
TexCuttingPlanes::tcl_command( TCLArgs& args, void* userdata)
{
  if (args[1] == "MoveWidget") {
      if (!control_widget) return;
      Point w(control_widget->ReferencePoint());
      if (args[2] == "xplus") {
	w+=Vector(ddv.x(), 0, 0);
      } else if (args[2] == "xminus") {
	w-=Vector(ddv.x(), 0, 0);
      } else if (args[2] == "yplus") {
	w+=Vector(0, ddv.y(), 0);
      } else if (args[2] == "yminus") {
	w-=Vector(0, ddv.y(), 0);
      } else if (args[2] == "zplus") {
	w+=Vector(0, 0, ddv.z());
      } else if (args[2] == "zminus"){
	w-=Vector(0, 0, ddv.z());
      } //else if (args[2] == "vplus"){
      control_widget->SetPosition(w);
      widget_moved(1);
      cerr<< "MoveWidgit " << w << endl;
  } else {
    Module::tcl_command(args, userdata);
  }
}

void TexCuttingPlanes::widget_moved(int)
{
  if( volren ){
      volren->SetControlPoint(control_widget->ReferencePoint());
    }
}


void TexCuttingPlanes::execute(void)
{
  static GLTexture3DHandle oldtex = 0;
  if (!intexture->get(tex)) {
    return;
  }
  else if (!tex.get_rep()) {
    return;
  }
  
  ColorMapHandle cmap;
  if( !incolormap->get(cmap)){
    return;
  }


  if(!control_widget){
    control_widget=scinew PointWidget(this, &control_lock, 0.2);
    
    Point Smin(tex->min());
    Point Smax(tex->max());
    Vector dv(Smax - Smin);
    ScalarFieldRGuchar *sf = tex->getField();
    ddv.x(dv.x()/(sf->nz - 1));
    ddv.y(dv.y()/(sf->ny - 1));
    ddv.z(dv.z()/(sf->nx - 1));

    double max =  std::max(Smax.x() - Smin.x(), Smax.y() - Smin.y());
    max = std::max( max, Smax.z() - Smin.z());
    control_widget->SetPosition(Interpolate(Smin,Smax,0.5));
    control_widget->SetScale(max/80.0);
    GeomObj *w=control_widget->GetWidget();
    control_id = ogeom->addObj( w, control_name, &control_lock);
  }


  if( !volren ){
    volren = new GLVolumeRenderer(0x12345676,
				  tex,
				  cmap);

    ogeom->addObj( volren, "Volume Renderer");
    volren->GLPlanes();
    volren->DrawPlanes();
  } else {
    if( tex.get_rep() != oldtex.get_rep() ){
      oldtex = tex;
      Point Smin(tex->min());
      Point Smax(tex->max());
      Vector dv(Smax - Smin);
      ScalarFieldRGuchar *sf = tex->getField();
      ddv.x(dv.x()/(sf->nz - 1));
      ddv.y(dv.y()/(sf->ny - 1));
      ddv.z(dv.z()/(sf->nx - 1));
      volren->SetVol( tex.get_rep() );
    }
    volren->SetColorMap( cmap.get_rep() );
  }
 
  volren->SetX(drawX.get());
  volren->SetY(drawY.get());
  volren->SetZ(drawZ.get());
  volren->SetView(drawView.get());


  ogeom->flushViews();				  
}

} // End namespace Modules
} // End namespace Uintah


