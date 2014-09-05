//  
//  For more information, please see: http://software.sci.utah.edu
//  
//  The MIT License
//  
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
//  University of Utah.
//  
//  License for the specific language governing rights and limitations under
//  Permission is hereby granted, free of charge, to any person obtaining a
//  copy of this software and associated documentation files (the "Software"),
//  to deal in the Software without restriction, including without limitation
//  the rights to use, copy, modify, merge, publish, distribute, sublicense,
//  and/or sell copies of the Software, and to permit persons to whom the
//  Software is furnished to do so, subject to the following conditions:
//  
//  The above copyright notice and this permission notice shall be included
//  in all copies or substantial portions of the Software.
//  
//  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
//  OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
//  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
//  THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
//  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
//  FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//  DEALINGS IN THE SOFTWARE.
//  
//    File   : VolumeSlicer.cc
//    Author : Milan Ikits
//    Author : Kurt Zimmerman
//    Date   : Sat Jul 10 21:55:08 2004

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
#include <Packages/Volume/Dataflow/Ports/Colormap2Port.h>

#include <iostream>
#include <algorithm>
#include <Packages/Volume/Core/Util/Utils.h>
#include <Packages/Volume/Core/Util/VideoCardInfo.h>

namespace Volume {

using namespace SCIRun;

class PSECORESHARE VolumeSlicer : public Module {
public:
  VolumeSlicer(GuiContext*);
  virtual ~VolumeSlicer();
  virtual void execute();
  virtual void widget_moved(bool last, BaseWidget*);
  virtual void tcl_command(GuiArgs&, void*);

private:
  TextureHandle tex_;

  ColorMapIPort* icmap1_;
  Colormap2IPort* icmap2_;
  TextureIPort* intexture_;
  GeometryOPort* ogeom_;
  ColorMapOPort* ocmap_;
  int cmap1_prevgen_;
  int cmap2_prevgen_;
  int card_mem_;
  
  CrowdMonitor control_lock_; 
  PointWidget *control_widget_;
  GeomID control_id_;

  GuiInt control_pos_saved_;
  GuiDouble control_x_;
  GuiDouble control_y_;
  GuiDouble control_z_;
  GuiInt draw_x_;
  GuiInt draw_y_;
  GuiInt draw_z_;
  GuiInt draw_view_;
  GuiInt interp_mode_;
  GuiInt draw_phi0_;
  GuiInt draw_phi1_;
  GuiDouble phi0_;
  GuiDouble phi1_;
  GuiInt cyl_active_;

  TextureHandle old_tex_;
  ColorMapHandle old_cmap_;
  Point old_min_, old_max_;
  GeomID geom_id_;
  SliceRenderer* slice_ren_;
  Point dmin_;
  Vector ddx_;
  Vector ddy_;
  Vector ddz_;
  double ddview_;
};


static string control_name("Control Widget");
			 
DECLARE_MAKER(VolumeSlicer)
VolumeSlicer::VolumeSlicer(GuiContext* ctx)
  : Module("VolumeSlicer", ctx, Source, "Visualization", "Volume"),
    tex_(0),
    cmap1_prevgen_(0),
    cmap2_prevgen_(0),
    card_mem_(video_card_memory_size()),
    control_lock_("VolumeSlicer resolution lock"),
    control_widget_(0),
    control_id_(-1),
    control_pos_saved_(ctx->subVar("control_pos_saved")),
    control_x_(ctx->subVar("control_x")),
    control_y_(ctx->subVar("control_y")),
    control_z_(ctx->subVar("control_z")),
    draw_x_(ctx->subVar("drawX")),
    draw_y_(ctx->subVar("drawY")),
    draw_z_(ctx->subVar("drawZ")),
    draw_view_(ctx->subVar("drawView")),
    interp_mode_(ctx->subVar("interp_mode")),
    draw_phi0_(ctx->subVar("draw_phi_0")),
    draw_phi1_(ctx->subVar("draw_phi_1")),
    phi0_(ctx->subVar("phi_0")),
    phi1_(ctx->subVar("phi_1")),
    cyl_active_(ctx->subVar("cyl_active")),
    old_tex_(0),
    old_cmap_(0),
    old_min_(Point(0,0,0)), old_max_(Point(0,0,0)),
    geom_id_(-1),
    slice_ren_(0)
{}

VolumeSlicer::~VolumeSlicer()
{}

void VolumeSlicer::execute()
{
  intexture_ = (TextureIPort*)get_iport("Texture");
  icmap1_ = (ColorMapIPort*)get_iport("ColorMap");
  icmap2_ = (Colormap2IPort*)get_iport("ColorMap2");
  ogeom_ = (GeometryOPort*)get_oport("Geometry");
  ocmap_ = (ColorMapOPort*)get_oport("ColorMap");
  if (!intexture_) {
    error("Unable to initialize iport 'GL Texture'.");
    return;
  }
  if (!icmap1_ && !icmap2_) {
    error("Unable to initialize iport 'ColorMap'.");
    return;
  }
  if (!ogeom_) {
    error("Unable to initialize oport 'Geometry'.");
    return;
  }
  
  if (!intexture_->get(tex_)) {
    return;
  } else if (!tex_.get_rep()) {
    return;
  }
  
  ColorMapHandle cmap1;
  Colormap2Handle cmap2;
  if(!icmap1_->get(cmap1) && !icmap2_->get(cmap2)) {
    return;
  }

  if(!control_widget_){
    control_widget_=scinew PointWidget(this, &control_lock_, 0.2);
    control_widget_->Connect(ogeom_);
    
    BBox b;
    tex_->get_bounds(b);
    Vector dv(b.diagonal());
    if(control_pos_saved_.get()) {
      control_widget_->SetPosition(Point(control_x_.get(),
					 control_y_.get(),
					 control_z_.get()));
      control_widget_->SetScale(dv.length()/80.0);
    } else {
      control_widget_->SetPosition(Interpolate(b.min(), b.max(), 0.5) );
      control_widget_->SetScale(dv.length()/80.0);
    }
    int nx = tex_->nx();
    int ny = tex_->ny();
    int nz = tex_->nz();
    Transform t(tex_->transform());
    dmin_=t.project(Point(0,0,0));
    ddx_= (t.project(Point(1,0,0))-dmin_) * (dv.x()/nx);
    ddy_= (t.project(Point(0,1,0))-dmin_) * (dv.y()/ny);
    ddz_= (t.project(Point(0,0,1))-dmin_) * (dv.z()/nz);
    ddview_ = (dv.length()/(std::max(nx, std::max(ny,nz)) -1));
  }

  //AuditAllocator(default_allocator);
  if(!slice_ren_) {
    slice_ren_ = new SliceRenderer(tex_, cmap1, cmap2, int(card_mem_*1024*1024*0.8));
    slice_ren_->set_control_point(tex_->transform().unproject(control_widget_->ReferencePoint()));
    //    ogeom->delAll();
    geom_id_ = ogeom_->addObj(slice_ren_, "Volume Slicer");
    cyl_active_.reset();
    draw_phi0_.reset();
    phi0_.reset();
    draw_phi1_.reset();
    phi1_.reset();
    slice_ren_->set_cylindrical(cyl_active_.get(), draw_phi0_.get(), phi0_.get(), 
                                draw_phi1_.get(), phi1_.get());
    old_tex_ = tex_;
    old_cmap_ = cmap1;
    BBox b;
    tex_->get_bounds(b);
    old_min_ = b.min();
    old_max_ = b.max();
  } else {
    BBox b;
    tex_->get_bounds(b);
    if(tex_.get_rep() != old_tex_.get_rep() ||
       b.min() != old_min_ || b.max() != old_max_) {
      old_tex_ = tex_;
      old_min_ = b.min();
      old_max_ = b.max();
      if(geom_id_ != -1) {
	ogeom_->delObj(geom_id_);
	geom_id_ = ogeom_->addObj(slice_ren_, "Volume Slicer");
      }
      Vector dv(b.diagonal());
      int nx = tex_->nx();
      int ny = tex_->ny();
      int nz = tex_->nz();
      Transform t(tex_->transform());
      dmin_=t.project(Point(0,0,0));
      ddx_= (t.project(Point(1,0,0))-dmin_) * (dv.x()/nx);
      ddy_= (t.project(Point(0,1,0))-dmin_) * (dv.y()/ny);
      ddz_= (t.project(Point(0,0,1))-dmin_) * (dv.z()/nz);
      ddview_ = (dv.length()/(std::max(nx, std::max(ny,nz)) -1));
      if(!b.inside(control_widget_->GetPosition())) {
	control_widget_->SetPosition(Interpolate(b.min(), b.max(), 0.5));
      }
      control_widget_->SetScale(dv.length()/80.0);
      slice_ren_->set_texture(tex_);
      slice_ren_->set_control_point(tex_->transform().unproject(control_widget_->ReferencePoint()));
    }

    if(cmap1 != old_cmap_) {
      slice_ren_->set_colormap1(cmap1);
      old_cmap_ = cmap1;
    }
  }
 
  //AuditAllocator(default_allocator);
  slice_ren_->set_interp(bool(interp_mode_.get()));
  //AuditAllocator(default_allocator);
  if(draw_x_.get() || draw_y_.get() || draw_z_.get()){
    if(control_id_ == -1) {
      GeomHandle w = control_widget_->GetWidget();
      control_id_ = ogeom_->addObj( w, control_name, &control_lock_);
    }
  } else {
    if(control_id_ != -1) {
      ogeom_->delObj(control_id_, 0);
      control_id_ = -1;
    }
  }  
  slice_ren_->set_x(draw_x_.get());
  slice_ren_->set_y(draw_y_.get());
  slice_ren_->set_z(draw_z_.get());
  slice_ren_->set_view(draw_view_.get());
  cyl_active_.reset();
  draw_phi0_.reset();
  phi0_.reset();
  draw_phi1_.reset();
  phi1_.reset();
  slice_ren_->set_cylindrical(cyl_active_.get(), draw_phi0_.get(), phi0_.get(), 
                              draw_phi1_.get(), phi1_.get());
  
  ogeom_->flushViews();		  
  //AuditAllocator(default_allocator);
  
  if(ocmap_ && cmap1.get_rep()) {
    ColorMapHandle outcmap;
    outcmap = new ColorMap(*cmap1.get_rep()); 
    double vmin = tex_->vmin();
    double vmax = tex_->vmax();
    outcmap->Scale(vmin, vmax);
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
    widget_moved(true, 0);
    control_x_.set(w.x());
    control_y_.set(w.y());
    control_z_.set(w.z());
    control_pos_saved_.set( 1 );
    ogeom_->flushViews();				  
  } else {
    Module::tcl_command(args, userdata);
  }
}

void
VolumeSlicer::widget_moved(bool,BaseWidget*)
{
  if(slice_ren_) {
    slice_ren_->set_control_point(tex_->transform().unproject(control_widget_->ReferencePoint()));
  }
  Point w(control_widget_->ReferencePoint());
  control_x_.set(w.x());
  control_y_.set(w.y());
  control_z_.set(w.z());
  control_pos_saved_.set(1);
}

} // namespace Volume
