/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
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
 * TextureVolVis.cc
 *
 * Simple interface to volume rendering stuff
 */

#include <Dataflow/Modules/Visualization/TextureVolVis.h>

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/ColorMapPort.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Widgets/PointWidget.h>

#include <Core/Geom/ColorMap.h>
#include <Core/Geom/GeomTriangles.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/GLVolumeRenderer/VolumeUtils.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Thread/CrowdMonitor.h>

#include <iostream>

#ifdef __sgi
#  include <ios>
#endif
#include <algorithm>

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
    geom_lock_("TextureVolVis geometry lock"),
    volren(0)
{
}

TextureVolVis::~TextureVolVis()
{

}

void TextureVolVis::widget_moved(bool, BaseWidget*)
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
    ogeom->addObj( volren, "VolumeRenderer Transparent", &geom_lock_);
  } else {
    geom_lock_.writeLock();
    volren->SetVol( tex.get_rep() );
    volren->SetColorMap( cmap );
    geom_lock_.writeUnlock();
  }
 
  geom_lock_.writeLock();
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
  geom_lock_.writeUnlock();
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


