/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


/*
 * TexCuttingPlanes.cc
 *
 * Simple interface to volume rendering stuff
 */

#include "TexCuttingPlanes.h"

#include <Dataflow/Network/Module.h>
#include <Core/Datatypes/ColorMap.h>
#include <Dataflow/Ports/ColorMapPort.h>
#include <Dataflow/Ports/GeometryPort.h>

#include <Core/Geom/GeomTriangles.h>
#include <Core/Geom/View.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Thread/CrowdMonitor.h>

#include <Dataflow/Widgets/PointWidget.h>
#include <iostream>
#include <algorithm>
#include <Core/Datatypes/VolumeUtils.h>



namespace SCIRun {


static string control_name("Control Widget");
			 
extern "C" Module* make_TexCuttingPlanes( const string& id) {
  return scinew TexCuttingPlanes(id);
}


TexCuttingPlanes::TexCuttingPlanes(const string& id)
  : Module("TexCuttingPlanes", id, Filter, "Visualization", "SCIRun"), 
  tex(0),
  control_lock("TexCuttingPlanes resolution lock"),
  control_widget(0),
  control_id(-1),
  drawX("drawX", id, this),
  drawY("drawY", id, this),
  drawZ("drawZ", id, this),
  drawView("drawView", id, this),
  interp_mode("interp_mode", id, this),
  volren(0)
{
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
      } else if (args[2] == "vplus"){
	GeometryData* data = ogeom->getData( 0, 1);
	Vector view = data->view->lookat() - data->view->eyep();
	view.normalize();
	w += view*ddview;
      } else if (args[2] == "vminus"){
	GeometryData* data = ogeom->getData( 0, 1);
	Vector view = data->view->lookat() - data->view->eyep();
	view.normalize();
	w -= view*ddview;
      }
      control_widget->SetPosition(w);
      widget_moved(1);
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
  intexture = (GLTexture3DIPort *)get_iport("GL Texture");
  incolormap = (ColorMapIPort *)get_iport("Color Map");
  ogeom = (GeometryOPort *)get_oport("Geometry");

  if (!intexture) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  if (!incolormap) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  if (!ogeom) {
    postMessage("Unable to initialize "+name+"'s oport\n");
    return;
  }
  
  //AuditAllocator(default_allocator);
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
    int nx, ny, nz;
    tex->get_dimensions(nx,ny,nz);
    ddv.x(dv.x()/(nx - 1));
    ddv.y(dv.y()/(ny - 1));
    ddv.z(dv.z()/(nz - 1));
    ddview = (dv.length()/(std::max(nx, std::max(ny,nz)) -1));
    control_widget->SetPosition(Interpolate(Smin,Smax,0.5));
    control_widget->SetScale(dv.length()/80.0);
  }


  //AuditAllocator(default_allocator);
  if( !volren ){
    volren = scinew GLVolumeRenderer(0x12345676,
				  tex,
				  cmap);

    if(tex->CC()){
      volren->SetInterp(false);
      interp_mode.set(0);
    }

    ogeom->addObj( volren, "Volume Slicer");
    volren->_GLPlanes();
    volren->DrawPlanes();
  } else {    
    if( tex.get_rep() != oldtex.get_rep() ){
      oldtex = tex;
      Point Smin(tex->min());
      Point Smax(tex->max());
      Vector dv(Smax - Smin);
      int nx, ny, nz;
      tex->get_dimensions(nx,ny,nz);
      ddv.x(dv.x()/(nx - 1));
      ddv.y(dv.y()/(ny - 1));
      ddv.z(dv.z()/(nz - 1));
      ddview = (dv.length()/(std::max(nx, std::max(ny,nz)) -1));
      volren->SetVol( tex.get_rep() );
    }

    volren->SetInterp( bool(interp_mode.get()));
    volren->SetColorMap( cmap.get_rep() );
  }
 
  //AuditAllocator(default_allocator);
  if(drawX.get() || drawY.get() || drawZ.get()){
    if( control_id == -1 ){
      GeomObj *w=control_widget->GetWidget();
      control_id = ogeom->addObj( w, control_name, &control_lock);
    }
  } else {
    if( control_id != -1){
      ogeom->delObj( control_id, 0);
      control_id = -1;
    }
  }  

  volren->SetX(drawX.get());
  volren->SetY(drawY.get());
  volren->SetZ(drawZ.get());
  volren->SetView(drawView.get());
  //AuditAllocator(default_allocator);

  ogeom->flushViews();				  
  //AuditAllocator(default_allocator);
}

} // End namespace SCIRun


