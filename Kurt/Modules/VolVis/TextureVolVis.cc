
/*
 * TextureVolVis.cc
 *
 * Simple interface to volume rendering stuff
 */

#include "TextureVolVis.h"

#include <SCICore/Containers/Array1.h>
#include <PSECore/Dataflow/Module.h>
#include <SCICore/Datatypes/ColorMap.h>
#include <PSECore/Datatypes/ColorMapPort.h>
#include <PSECore/Datatypes/GeometryPort.h>
#include <PSECore/Datatypes/ScalarFieldPort.h>
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
using std::cerr;


static clString control_name("Control Widget");
			 
extern "C" Module* make_TextureVolVis( const clString& id) {
  return new TextureVolVis(id);
}


TextureVolVis::TextureVolVis(const clString& id)
  : Module("TextureVolVis", id, Filter), 
  alpha_scale("alpha_scale", id, this),
  num_slices("num_slices", id, this),
  draw_mode("draw_mode", id, this),
  render_style("render_style", id, this),
  control_lock("TextureVolVis resolution lock"),
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

void TextureVolVis::widget_moved(int obj)
{
  //  cerr<<"the widget id is "<<obj<<endl;
  //  cerr<<"is brick set? "<< (( brick == 0)? "NO":"YES")<<endl;
  //  cerr<<"mode is "<<mode<<endl;
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
  GLTexture3DHandle tex;
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


  if( !volren ){
    volren = new GLVolumeRenderer(0x12345676,
				  tex.get_rep(),
				  cmap.get_rep());

    ogeom->addObj( volren, "Volume Renderer");
  } else {
    volren->SetVol( tex.get_rep() );
    volren->SetColorMap( cmap.get_rep() );
  }

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

  volren->SetNSlices( num_slices.get() );
  volren->SetSliceAlpha( alpha_scale.get() );

  ogeom->flushViews();				  
}

} // End namespace Modules
} // End namespace Uintah


