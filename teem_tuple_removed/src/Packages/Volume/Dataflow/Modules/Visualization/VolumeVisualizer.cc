/*
 *  VolumeVisualizer.cc:
 *
 *  Written by:
 *   kuzimmer
 *   TODAY'S DATE HERE
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Geom/ColorMap.h>
#include <Dataflow/Ports/ColorMapPort.h>
#include <Dataflow/Ports/GeometryPort.h>

#include <Dataflow/share/share.h>

#include <Core/GuiInterface/GuiVar.h>
#include <Core/Thread/CrowdMonitor.h>
#include <Dataflow/Widgets/PointWidget.h>

#include <Packages/Volume/Core/Geom/VolumeRenderer.h>
#include <Packages/Volume/Core/Datatypes/Texture.h>
#include <Packages/Volume/Dataflow/Ports/TexturePort.h>

#include <iostream>
#ifdef __sgi
#include <ios>
#endif
#include <algorithm>
#include <Packages/Volume/Core/Util/Utils.h>


namespace Volume {

using namespace SCIRun;

class PSECORESHARE VolumeVisualizer : public Module {
public:
  VolumeVisualizer(GuiContext*);

  virtual ~VolumeVisualizer();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);
private:
  
  TextureHandle tex;

  ColorMapIPort* icmap;
  TextureIPort* intexture;
  GeometryOPort* ogeom;
  ColorMapOPort* ocmap;
   
  CrowdMonitor control_lock; 
  PointWidget *control_widget;
  GeomID control_id;


  int cmap_id;  // id associated with color map...
  

  GuiInt gui_num_slices_;
  GuiInt gui_render_style_;
  GuiDouble gui_alpha_scale_;
  GuiInt gui_interp_mode_;
  VolumeRenderer *volren_;
};


DECLARE_MAKER(VolumeVisualizer)
VolumeVisualizer::VolumeVisualizer(GuiContext* ctx)
  : Module("VolumeVisualizer", ctx, Source, "Visualization", "Volume"),
    tex(0),
    control_lock("VolumeVisualizer resolution lock"),
    control_widget(0),
    control_id(-1),
    gui_num_slices_(ctx->subVar("num_slices")),
    gui_render_style_(ctx->subVar("render_style")),
    gui_alpha_scale_(ctx->subVar("alpha_scale")),
    gui_interp_mode_(ctx->subVar("interp_mode")),
    volren_(0)
{
}

VolumeVisualizer::~VolumeVisualizer(){
}

void
 VolumeVisualizer::execute(){

  static Point oldmin(0,0,0), oldmax(0,0,0);
  static int oldni = 0, oldnj = 0, oldnk = 0;
  static GeomID geomID  = 0;


  intexture = (TextureIPort *)get_iport("Texture");
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

//   if(!control_widget){
//     control_widget=scinew PointWidget(this, &control_lock, 0.2);
//     Transform trans = tex->get_field_transform();
//     Point Smin(trans.project(tex->min()));
//     Point Smax(trans.project(tex->max()));

//     double max =  std::max(Smax.x() - Smin.x(), Smax.y() - Smin.y());
//     max = std::max( max, Smax.z() - Smin.z());
//     control_widget->SetPosition(Interpolate(Smin,Smax,0.5));
//     control_widget->SetScale(max/80.0);
//   }

  //AuditAllocator(default_allocator);
  if( !volren_ ){
    volren_ = new VolumeRenderer(tex, cmap);
    oldmin = tex->min();
    oldmax = tex->max();
    tex->get_dimensions(oldni, oldnj, oldnk);

    //    ogeom->delAll();
    geomID = ogeom->addObj( volren_, "VolumeRenderer TransParent");
  } else {
    volren_->SetTexture( tex );
    volren_->SetColorMap( cmap );
    int ni, nj, nk;
    tex->get_dimensions( ni, nj, nk );
    if( oldmin != tex->min() || oldmax != tex->max() ||
	ni != oldni || nj != oldnj || nk != oldnk ){
      ogeom->delObj( geomID );
      geomID = ogeom->addObj( volren_, "VolumeRenderer TransParent");
      oldni = ni; oldnj = nj; oldnk = nk;
      oldmin = tex->min();
      oldmax = tex->max();
    }
  }
 
  //AuditAllocator(default_allocator);
  volren_->SetInterp( bool(gui_interp_mode_.get()));
  //AuditAllocator(default_allocator);

  switch( gui_render_style_.get() ) {
  case 0:
    volren_->SetRenderMode(VolumeRenderer::OVEROP);
    break;
  case 1:
    volren_->SetRenderMode(VolumeRenderer::MIP);
    break;
  case 2:
    volren_->SetRenderMode(VolumeRenderer::ATTENUATE);
  }
  
  //AuditAllocator(default_allocator);
  volren_->SetNSlices( gui_num_slices_.get() );
  volren_->SetSliceAlpha( gui_alpha_scale_.get() );
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
    tex->get_min_max(min, max);
    outcmap->Scale(min, max);
    ocmap->send(outcmap);
  }    

}

void
 VolumeVisualizer::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace Volume


