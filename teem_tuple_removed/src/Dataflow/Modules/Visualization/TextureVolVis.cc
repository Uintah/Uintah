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
#include <Core/Geom/ColorMap.h>
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
			 
DECLARE_MAKER(TextureVolVis)

TextureVolVis::TextureVolVis(GuiContext* ctx)
  : Module("TextureVolVis", ctx, Filter, "Visualization", "SCIRun"),
    tex(0),
    control_lock("TextureVolVis resolution lock"),
    control_widget(0),
    control_id(-1),
    num_slices(ctx->subVar("num_slices")),
    draw_mode(ctx->subVar("draw_mode")),
    render_style(ctx->subVar("render_style")),
    alpha_scale(ctx->subVar("alpha_scale")),
    interp_mode(ctx->subVar("interp_mode")),
    volren(0)
{
}

TextureVolVis::~TextureVolVis()
{

}

void TextureVolVis::widget_moved(bool)
{
  if( volren ){
      volren->SetControlPoint(control_widget->ReferencePoint());
    }
}


void TextureVolVis::execute(void)
{
  intexture = (GLTexture3DIPort *)get_iport("GL Texture");
  icmap = (ColorMapIPort *)get_iport("ColorMap");
  ogeom = (GeometryOPort *)get_oport("Geometry");
  ocmap = (ColorMapOPort *)get_oport("ColorMap");
  if (!intexture) {
    error("Unable to initialize iport 'GL Texture'.");
    return;
  }
  if (!icmap) {
    error("Unable to initialize iport 'ColorMap'.");
    return;
  }
  if (!ogeom) {
    error("Unable to initialize oport 'Geometry'.");
    return;
  }

  
  if (!intexture->get(tex)) {
    return;
  }
  else if (!tex.get_rep()) {
    return;
  }
  
  ColorMapHandle cmap;
  if( !icmap->get(cmap)){
    return;
  }

  if(!control_widget){
    control_widget=scinew PointWidget(this, &control_lock, 0.2);
    Transform trans = tex->get_field_transform();
    Point Smin(trans.project(tex->min()));
    Point Smax(trans.project(tex->max()));

    double max =  std::max(Smax.x() - Smin.x(), Smax.y() - Smin.y());
    max = std::max( max, Smax.z() - Smin.z());
    control_widget->SetPosition(Interpolate(Smin,Smax,0.5));
    control_widget->SetScale(max/80.0);
  }

  //AuditAllocator(default_allocator);
  if( !volren ){
    volren = new GLVolumeRenderer(tex, cmap);
//     if(tex->CC()){
//       volren->SetInterp(false);
//       interp_mode.set(0);
//     }
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
    volren->set_tex_ren_state(GLVolumeRenderer::TRS_GLOverOp);
    break;
  case 1:
    volren->set_tex_ren_state(GLVolumeRenderer::TRS_GLMIP);
    break;
  case 2:
    volren->set_tex_ren_state(GLVolumeRenderer::TRS_GLAttenuate);
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
      GeomHandle w = control_widget->GetWidget();
      control_id = ogeom->addObj( w, control_name, &control_lock);
    }
    break;
  case 2:
    volren->DrawROI();
    if( control_id == -1){
      GeomHandle w = control_widget->GetWidget();
      control_id = ogeom->addObj( w, control_name, &control_lock);
   }
  }

  //AuditAllocator(default_allocator);
  volren->SetNSlices( num_slices.get() );
  volren->SetSliceAlpha( alpha_scale.get() );
  //AuditAllocator(default_allocator);
  ogeom->flushViews();				  
  //AuditAllocator(default_allocator);

  if (!ocmap) {
    error("Unable to initialize oport 'Color Map'.");
    return;
  } else {
    ColorMapHandle outcmap;
    outcmap = new ColorMap(*cmap.get_rep()); 
    double min, max;
    tex->getminmax(min, max);
    outcmap->Scale(min, max);
    ocmap->send(outcmap);
  }    

}

} // End namespace SCIRun


