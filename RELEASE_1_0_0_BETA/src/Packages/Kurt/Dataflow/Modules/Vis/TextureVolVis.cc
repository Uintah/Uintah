
/*
 * TextureVolVis.cc
 *
 * Simple interface to volume rendering stuff
 */

#include "TextureVolVis.h"

#include <Core/Containers/Array1.h>
#include <Dataflow/Network/Module.h>
#include <Core/Datatypes/ColorMap.h>
#include <Dataflow/Ports/ColorMapPort.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/ScalarFieldPort.h>
#include <Core/Datatypes/ScalarFieldRG.h>

#include <Core/Geom/GeomTriangles.h>

#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Thread/CrowdMonitor.h>

#include <Dataflow/Widgets/PointWidget.h>
#include <iostream>
#include <ios>
#include <algorithm>
#include <Packages/Kurt/Core/Datatypes/VolumeUtils.h>

using std::cerr;
using std::endl;
using std::hex;
using std::dec;

namespace Kurt {

using namespace SCIRun;

static clString control_name("Control Widget");
			 
extern "C" Module* make_TextureVolVis( const clString& id) {
  return new TextureVolVis(id);
}


TextureVolVis::TextureVolVis(const clString& id)
  : Module("TextureVolVis", id, Filter), 
  alpha_scale("alpha_scale", id, this),
  alpha_gamma("alpha_gamma", id, this),
  num_slices("num_slices", id, this),
  draw_mode("draw_mode", id, this),
  render_style("render_style", id, this),
  control_lock("TextureVolVis resolution lock"),
  interp_mode("interp_mode", id, this),
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

TextureVolVis::~TextureVolVis()
{

}

void TextureVolVis::widget_moved(int)
{
  if( volren ){
      volren->SetControlPoint(control_widget->ReferencePoint());
    }
}


void TextureVolVis::SwapXZ( ScalarFieldHandle sfh )
{
  ScalarFieldRGuchar *ifu, *ofu;
  ifu = sfh->getRGBase()->getRGUchar();
  int nx=ifu->nx;
  int ny=ifu->ny;
  int nz=ifu->nz;
  Point min;
  Point max;
  sfh->get_bounds(min, max);

  ofu = scinew ScalarFieldRGuchar();
  ofu->resize(nz,ny,nx);
  ofu->set_bounds(min, max);
  for (int i=0, ii=0; i<nx; i++, ii++)
    for (int j=0, jj=0; j<ny; j++, jj++)
      for (int k=0, kk=0; k<nz; k++, kk++)
	ofu->grid(k,j,i)=ifu->grid(ii,jj,kk);

  sfh = ScalarFieldHandle( ofu );

}
  
void TextureVolVis::execute(void)
{

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
    volren = new GLVolumeRenderer(0x12345676,
				  tex,
				  cmap);
    if(tex->CC()){
      volren->SetInterp(false);
      interp_mode.set(0);
    }
    //    ogeom->delAll();
    ogeom->addObj( volren, "VolumeRenderer TransParent");
    if(control_widget)
      volren->SetControlPoint(control_widget->ReferencePoint());
  } else {
    volren->SetVol( tex );
    volren->SetColorMap( cmap );
  }
 
  //AuditAllocator(default_allocator);
  volren->SetInterp( bool(interp_mode.get()));
  //AuditAllocator(default_allocator);

  switch( render_style.get() ) {
  case 0:
    volren->GLOverOp();
    break;
  case 1:
    volren->GLMIP();
    break;
  case 2:
    volren->GLAttenuate();
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
  volren->SetSliceAlpha( alpha_gamma.get() );
  volren->SetScaleAlpha( alpha_scale.get() );
  volren->BuildTransferFunctions();
  //AuditAllocator(default_allocator);
  ogeom->flushViews();				  
  //AuditAllocator(default_allocator);
}
} // End namespace Kurt



