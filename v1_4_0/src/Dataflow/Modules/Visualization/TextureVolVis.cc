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
 * TextureVolVis.cc
 *
 * Simple interface to volume rendering stuff
 */

#include "TextureVolVis.h"

#include <Dataflow/Network/Module.h>
#include <Core/Datatypes/ColorMap.h>
#include <Dataflow/Ports/ColorMapPort.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Geom/GeomTriangles.h>

#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Thread/CrowdMonitor.h>

#include <Dataflow/Widgets/PointWidget.h>
#include <iostream>
#ifdef __sgi
#include <ios>
#endif
#include <algorithm>
#include <Core/GLVolumeRenderer/VolumeUtils.h>

using std::hex;
using std::dec;

namespace SCIRun {


static string control_name("Control Widget");
			 
extern "C" Module* make_TextureVolVis( const string& id)
{
  return new TextureVolVis(id);
}


TextureVolVis::TextureVolVis(const string& id)
  : Module("TextureVolVis", id, Filter, "Visualization", "SCIRun"),
    tex(0),
    control_lock("TextureVolVis resolution lock"),
    control_widget(0),
    control_id(-1),
    num_slices("num_slices", id, this),
    draw_mode("draw_mode", id, this),
    render_style("render_style", id, this),
    alpha_scale("alpha_scale", id, this),
    interp_mode("interp_mode", id, this),
    volren(0)
{
}

TextureVolVis::~TextureVolVis()
{

}

void TextureVolVis::widget_moved(int)
{
  if( volren ){
      volren->SetControlPoint(control_widget->ReferencePoint());
    }
}


void TextureVolVis::execute(void)
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

    double max =  std::max(Smax.x() - Smin.x(), Smax.y() - Smin.y());
    max = std::max( max, Smax.z() - Smin.z());
    control_widget->SetPosition(Interpolate(Smin,Smax,0.5));
    control_widget->SetScale(max/80.0);
  }

  //AuditAllocator(default_allocator);
  if( !volren ){
    volren = new GLVolumeRenderer(0x12345676, tex, cmap);
    if(tex->CC()){
      volren->SetInterp(false);
      interp_mode.set(0);
    }
    //    ogeom->delAll();
    ogeom->addObj( volren, "VolumeRenderer TransParent");
  } else {
    volren->SetVol( tex );
    volren->SetColorMap( cmap );
  }
 
  //AuditAllocator(default_allocator);
  volren->SetInterp( bool(interp_mode.get()));
  //AuditAllocator(default_allocator);

  switch( render_style.get() ) {
  case 0:
    volren->_GLOverOp();
    break;
  case 1:
    volren->_GLMIP();
    break;
  case 2:
    volren->_GLAttenuate();
  }
  
  switch( draw_mode.get()) {
  case 0:
    volren->DrawFullRes();
    if( control_id >= 0){
      ogeom->delObj( control_id, 0);
      control_id = -1;
    }
    break;
  case 1:
    volren->DrawLOS();
    if( control_id == -1){
      GeomObj *w=control_widget->GetWidget();
      control_id = ogeom->addObj( w, control_name, &control_lock);
    }
    break;
  case 2:
    volren->DrawROI();
    if( control_id == -1){
      GeomObj *w=control_widget->GetWidget();
      control_id = ogeom->addObj( w, control_name, &control_lock);
   }
  }

  //AuditAllocator(default_allocator);
  volren->SetNSlices( num_slices.get() );
  volren->SetSliceAlpha( alpha_scale.get() );
  //AuditAllocator(default_allocator);
  ogeom->flushViews();				  
  //AuditAllocator(default_allocator);
}

} // End namespace SCIRun


