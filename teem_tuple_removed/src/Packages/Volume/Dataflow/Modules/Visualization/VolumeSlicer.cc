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
#include <Core/Geom/View.h>
#include <Packages/Volume/Core/Geom/SliceRenderer.h>
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

class PSECORESHARE VolumeSlicer : public Module {
public:
  VolumeSlicer(GuiContext*);

  virtual ~VolumeSlicer();

  virtual void execute();
  virtual void widget_moved(bool last);    
  virtual void tcl_command(GuiArgs&, void*);
private:
  
  TextureHandle tex_;

  ColorMapIPort* icmap_;
  TextureIPort* intexture_;
  GeometryOPort* ogeom_;
  ColorMapOPort* ocmap_;
   
  CrowdMonitor control_lock_; 
  PointWidget *control_widget_;
  GeomID control_id_;


  GuiInt                  control_pos_saved_;
  GuiDouble               control_x_;
  GuiDouble               control_y_;
  GuiDouble               control_z_;
  GuiInt                  drawX_;
  GuiInt                  drawY_;
  GuiInt                  drawZ_;
  GuiInt                  drawView_;
  GuiInt                  interp_mode_;
  GuiInt                  draw_phi0_;
  GuiInt                  draw_phi1_;
  GuiDouble		  phi0_;
  GuiDouble		  phi1_;
  GuiInt                  cyl_active_;

  SliceRenderer          *slice_ren_;
  Point                   dmin_;
  Vector                  ddx_;
  Vector                  ddy_;
  Vector                  ddz_;
  double                  ddview_;
};


static string control_name("Control Widget");
			 
DECLARE_MAKER(VolumeSlicer)
VolumeSlicer::VolumeSlicer(GuiContext* ctx)
  : Module("VolumeSlicer", ctx, Source, "Visualization", "Volume"),
    tex_(0),
    control_lock_("VolumeSlicer resolution lock"),
    control_widget_(0),
    control_id_(-1),
    control_pos_saved_(ctx->subVar("control_pos_saved")),
    control_x_(ctx->subVar("control_x")),
    control_y_(ctx->subVar("control_y")),
    control_z_(ctx->subVar("control_z")),
    drawX_(ctx->subVar("drawX")),
    drawY_(ctx->subVar("drawY")),
    drawZ_(ctx->subVar("drawZ")),
    drawView_(ctx->subVar("drawView")),
    interp_mode_(ctx->subVar("interp_mode")),
    draw_phi0_(ctx->subVar("draw_phi_0")),
    draw_phi1_(ctx->subVar("draw_phi_1")),
    phi0_(ctx->subVar("phi_0")),
    phi1_(ctx->subVar("phi_1")),
    cyl_active_(ctx->subVar("cyl_active")),
    slice_ren_(0)
{
}

VolumeSlicer::~VolumeSlicer(){
}

void
 VolumeSlicer::execute(){
  intexture_ = (TextureIPort *)get_iport("Texture");
  icmap_ = (ColorMapIPort *)get_iport("ColorMap");
  ogeom_ = (GeometryOPort *)get_oport("Geometry");
  ocmap_ = (ColorMapOPort *)get_oport("ColorMap");
  if (!intexture_) {
    error("Unable to initialize iport 'GL Texture'.");
    return;
  }
  if (!icmap_) {
    error("Unable to initialize iport 'ColorMap'.");
    return;
  }
  if (!ogeom_) {
    error("Unable to initialize oport 'Geometry'.");
    return;
  }

  
  static TextureHandle oldtex = 0;
  if (!intexture_->get(tex_)) {
    return;
  } else if (!tex_.get_rep()) {
    return;
  }
  
  ColorMapHandle cmap;
  if( !icmap_->get(cmap)){
    return;
  }

  if(!control_widget_){
    control_widget_=scinew PointWidget(this, &control_lock_, 0.2);
    control_widget_->Connect(ogeom_);
    
    BBox b;
    tex_->get_bounds(b);
    Vector dv(b.diagonal());
    if( control_pos_saved_.get() ) {
      control_widget_->SetPosition(Point(control_x_.get(),
					 control_y_.get(),
					 control_z_.get()));
      control_widget_->SetScale(dv.length()/80.0);
    } else {
      int nx, ny, nz;
      Transform t(tex_->get_field_transform());
      dmin_=t.project(Point(0,0,0));
      ddx_=t.project(Point(1,0,0))-dmin_;
      ddy_=t.project(Point(0,1,0))-dmin_;
      ddz_=t.project(Point(0,0,1))-dmin_;
      tex_->get_dimensions(nx,ny,nz);
      ddview_ = (dv.length()/(std::max(nx, std::max(ny,nz)) -1));
      control_widget_->SetPosition(Interpolate(b.min(), b.max(), 0.5));
      control_widget_->SetScale(dv.length()/80.0);
    }
  }

  //AuditAllocator(default_allocator);
  if( !slice_ren_ ){
    slice_ren_ = new SliceRenderer(tex_, cmap);
    slice_ren_->SetControlPoint(tex_->get_field_transform().unproject(control_widget_->ReferencePoint()));
    //    ogeom->delAll();
    ogeom_->addObj( slice_ren_, "Volume Slicer");
    cyl_active_.reset();
    draw_phi0_.reset();
    phi0_.reset();
    draw_phi1_.reset();
    phi1_.reset();
    slice_ren_->set_cylindrical(cyl_active_.get(), draw_phi0_.get(), phi0_.get(), 
			     draw_phi1_.get(), phi1_.get());
  } else {
    if( tex_.get_rep() != oldtex.get_rep() ){
      oldtex = tex_;
      BBox b;
      tex_->get_bounds(b);
      Vector dv(b.diagonal());
      int nx, ny, nz;
      Transform t(tex_->get_field_transform());
      dmin_=t.project(Point(0,0,0));
      ddx_=t.project(Point(1,0,0))-dmin_;
      ddy_=t.project(Point(0,1,0))-dmin_;
      ddz_=t.project(Point(0,0,1))-dmin_;
      tex_->get_dimensions(nx,ny,nz);
      ddview_ = (dv.length()/(std::max(nx, std::max(ny,nz)) -1));
      if (!b.inside(control_widget_->GetPosition())) {
	control_widget_->SetPosition(Interpolate(b.min(), b.max(), 0.5));
	control_widget_->SetScale(dv.length()/80.0);
      }
      slice_ren_->SetTexture( tex_ );
      slice_ren_->SetControlPoint(tex_->get_field_transform().unproject(control_widget_->ReferencePoint()));
      slice_ren_->SetColorMap( cmap );
    }
  }
 
  //AuditAllocator(default_allocator);
  slice_ren_->SetInterp( bool(interp_mode_.get()));
  //AuditAllocator(default_allocator);
  if(drawX_.get() || drawY_.get() || drawZ_.get()){
    if( control_id_ == -1 ){
      GeomHandle w = control_widget_->GetWidget();
      control_id_ = ogeom_->addObj( w, control_name, &control_lock_);
    }
  } else {
    if( control_id_ != -1){
      ogeom_->delObj( control_id_, 0);
      control_id_ = -1;
    }
  }  
  slice_ren_->SetX(drawX_.get());
  slice_ren_->SetY(drawY_.get());
  slice_ren_->SetZ(drawZ_.get());
  slice_ren_->SetView(drawView_.get());
  cyl_active_.reset();
  draw_phi0_.reset();
  phi0_.reset();
  draw_phi1_.reset();
  phi1_.reset();
  slice_ren_->set_cylindrical(cyl_active_.get(), draw_phi0_.get(), phi0_.get(), 
			   draw_phi1_.get(), phi1_.get());
  
  ogeom_->flushViews();				  
  //AuditAllocator(default_allocator);
  
  if (!ocmap_) {
    error("Unable to initialize oport 'Color Map'.");
    return;
  } else {
    ColorMapHandle outcmap;
    outcmap = new ColorMap(*cmap.get_rep()); 
    double min, max;
    tex_->get_min_max(min, max);
    outcmap->Scale(min, max);
    ocmap_->send(outcmap);
  }    
  
}

void
 VolumeSlicer::tcl_command(GuiArgs& args, void* userdata)
{
  if (args[1] == "MoveWidget") {
      if (!control_widget_) return;
      Point w(control_widget_->ReferencePoint());
      if (args[2] == "xplus") {
	w+=ddx_*atof(args[3].c_str());
      } else if (args[2] == "xat") {
	w=dmin_+ddx_*atof(args[3].c_str());
      } else if (args[2] == "yplus") {
	w+=ddy_*atof(args[3].c_str());
      } else if (args[2] == "yat") {
	w=dmin_+ddy_*atof(args[3].c_str());
      } else if (args[2] == "zplus") {
	w+=ddz_*atof(args[3].c_str());
      } else if (args[2] == "zat") {
	w=dmin_+ddz_*atof(args[3].c_str());
      } else if (args[2] == "vplus"){
	GeometryData* data = ogeom_->getData(0, 0, 1);
	Vector view = data->view->lookat() - data->view->eyep();
	view.normalize();
	w += view*ddview_*atof(args[3].c_str());
      }
      control_widget_->SetPosition(w);
      widget_moved(true);
      control_x_.set( w.x() );
      control_y_.set( w.y() );
      control_z_.set( w.z() );
      control_pos_saved_.set( 1 );
      ogeom_->flushViews();				  
  } else {
    Module::tcl_command(args, userdata);
  }
}




void
VolumeSlicer::widget_moved(bool)
{
  if( slice_ren_ ){
      slice_ren_->SetControlPoint(tex_->get_field_transform().unproject(control_widget_->ReferencePoint()));
    }
  Point w(control_widget_->ReferencePoint());
  control_x_.set( w.x() );
  control_y_.set( w.y() );
  control_z_.set( w.z() );
  control_pos_saved_.set( 1 );
}


} // End namespace Volume




