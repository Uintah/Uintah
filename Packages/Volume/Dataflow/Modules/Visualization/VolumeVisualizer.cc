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
#include <Packages/Volume/Dataflow/Ports/Colormap2Port.h>

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
  Colormap2IPort* icmap2;
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

  GuiInt gui_shading_;
  GuiDouble gui_ambient_;
  GuiDouble gui_diffuse_;
  GuiDouble gui_specular_;
  GuiDouble gui_shine_;
  GuiInt gui_light_;
  
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
    gui_shading_(ctx->subVar("shading")),
    gui_ambient_(ctx->subVar("ambient")),
    gui_diffuse_(ctx->subVar("diffuse")),
    gui_specular_(ctx->subVar("specular")),
    gui_shine_(ctx->subVar("shine")),
    gui_light_(ctx->subVar("light")),
    volren_(0)
{
}

VolumeVisualizer::~VolumeVisualizer(){
}

void
VolumeVisualizer::execute(){

  cerr << "VolumeVisualizer::execute" << endl;
  static Point oldmin(0,0,0), oldmax(0,0,0);
  static int oldni = 0, oldnj = 0, oldnk = 0;
  static GeomID geomID  = 0;
  
  intexture = (TextureIPort *)get_iport("Texture");
  icmap = (ColorMapIPort *)get_iport("ColorMap");
  icmap2 = (Colormap2IPort*)get_iport("ColorMap2");
  ogeom = (GeometryOPort *)get_oport("Geometry");
  ocmap = (ColorMapOPort *)get_oport("ColorMap");
  if (!intexture) {
    error("Unable to initialize iport 'GL Texture'.");
    return;
  }
  if (!icmap && !icmap2) {
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
  Colormap2Handle cmap2;
  bool c = icmap->get(cmap);
  bool c2 = icmap2->get(cmap2);
  if(!c && !c2){
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
    volren_ = new VolumeRenderer(tex, cmap, cmap2);
    oldmin = tex->min();
    oldmax = tex->max();
    tex->get_dimensions(oldni, oldnj, oldnk);

    //    ogeom->delAll();
    geomID = ogeom->addObj( volren_, "VolumeRenderer TransParent");
  } else {
    volren_->SetTexture(tex);
    volren_->SetColorMap(cmap);
    volren_->SetColormap2(cmap2);
    int ni, nj, nk;
    tex->get_dimensions(ni, nj, nk);
    if( oldmin != tex->min() || oldmax != tex->max() ||
	ni != oldni || nj != oldnj || nk != oldnk ){
      ogeom->delObj(geomID);
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
  }
  
  //AuditAllocator(default_allocator);
  volren_->SetNSlices( gui_num_slices_.get() );
  volren_->SetSliceAlpha( gui_alpha_scale_.get() );

  volren_->setShading(gui_shading_.get());
  volren_->setMaterial(gui_ambient_.get(), gui_diffuse_.get(),
                       gui_specular_.get(), gui_shine_.get());
  volren_->setLight(gui_light_.get());
  
  //AuditAllocator(default_allocator);
  ogeom->flushViews();				  
  //AuditAllocator(default_allocator);

  if (!ocmap) {
    error("Unable to initialize oport 'Color Map'.");
    return;
  } else {
    if(c) {
      ColorMapHandle outcmap;
      outcmap = new ColorMap(*cmap.get_rep()); 
      double vmin, vmax, gmin, gmax;
      tex->get_min_max(vmin, vmax, gmin, gmax);
      outcmap->Scale(vmin, vmax);
      ocmap->send(outcmap);
    }
  }    
}

void
 VolumeVisualizer::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace Volume


