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

#include <sci_defs/ogl_defs.h>
#include <Dataflow/Network/Module.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Geom/ColorMap.h>
#include <Dataflow/Ports/ColorMapPort.h>
#include <Dataflow/Ports/GeometryPort.h>

#include <Core/GuiInterface/GuiVar.h>
#include <Core/Thread/CrowdMonitor.h>
#include <Dataflow/Widgets/PointWidget.h>
#include <Core/Geom/View.h>
#include <Core/Volume/SliceRenderer.h>
#include <Dataflow/Ports/TexturePort.h>
#include <Core/Geom/ShaderProgramARB.h>
#include <Dataflow/Ports/Colormap2Port.h>

#include <iostream>
#include <algorithm>
#include <Core/Volume/VideoCardInfo.h>

namespace SCIRun {

class VolumeSlicer : public Module {
public:
  VolumeSlicer(GuiContext*);
  virtual ~VolumeSlicer();
  virtual void execute();
  virtual void widget_moved(bool last, BaseWidget*);
  virtual void tcl_command(GuiArgs&, void*);

private:
  TextureHandle tex_;

  ColorMapIPort* icmap1_;
  ColorMap2IPort* icmap2_;
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
  GuiInt gui_multi_level_;
  GuiInt gui_color_changed_;
  GuiInt gui_outline_levels_;
  GuiInt gui_use_stencil_;

  TextureHandle old_tex_;
  ColorMapHandle old_cmap1_;
  ColorMap2Handle old_cmap2_;
  Point old_min_, old_max_;
  GeomID geom_id_;
  SliceRenderer* slice_ren_;
  Point dmin_;
  Vector ddx_;
  Vector ddy_;
  Vector ddz_;
  double ddview_;

  int color_changed_;
};


static string control_name("Control Widget");
			 
DECLARE_MAKER(VolumeSlicer)
VolumeSlicer::VolumeSlicer(GuiContext* ctx)
  : Module("VolumeSlicer", ctx, Source, "Visualization", "SCIRun"),
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
    gui_multi_level_(ctx->subVar("multi_level")),
    gui_color_changed_(ctx->subVar("color_changed")),
    gui_outline_levels_(ctx->subVar("outline_levels")),
    gui_use_stencil_(ctx->subVar("use_stencil")),
    old_tex_(0),
    old_cmap1_(0),
    old_cmap2_(0),
    old_min_(Point(0,0,0)), old_max_(Point(0,0,0)),
    geom_id_(-1),
    slice_ren_(0),
    color_changed_(true)
{}

VolumeSlicer::~VolumeSlicer()
{}

void VolumeSlicer::execute()
{
  intexture_ = (TextureIPort*)get_iport("Texture");
  icmap1_ = (ColorMapIPort*)get_iport("ColorMap");
  icmap2_ = (ColorMap2IPort*)get_iport("ColorMap2");
  ogeom_ = (GeometryOPort*)get_oport("Geometry");
  ocmap_ = (ColorMapOPort*)get_oport("ColorMap");
  
  if (!(intexture_->get(tex_) && tex_.get_rep()))
  {
    warning("Required Texture input not found.");
    return;
  }
  
  ColorMapHandle cmap1(0);
  ColorMap2Handle cmap2(0);
  bool c1 = icmap1_->get(cmap1);
  bool c2 = icmap2_->get(cmap2);

  if (c2)
  {
    if (!ShaderProgramARB::shaders_supported())
    {
      warning("ColorMap2 usage is not supported by this machine.");
      cmap2 = 0;
      c2 = false;
    }
    else
    {
      if (tex_->nc() == 1)
      {
        warning("ColorMap2 requires gradient magnitude in the texture.");
        cmap2 = 0;
        c2 = false;
      }
    }
  }

  if (!c1 && !c2)
  {
    error("No colormap available to render.  Nothing drawn.");
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
    ddx_ = t.project(Vector(1.0/(nx-1), 0, 0));
    ddy_ = t.project(Vector(0, 1.0/(ny-1), 0));
    ddz_ = t.project(Vector(0, 0, 1.0/(nz-1)));
    ddview_ = (dv.length()/(std::max(nx, std::max(ny,nz)) -1));
  }

  string s;
  gui->eval(id + " hasUI", s);
  if( s == "0" )
    gui->execute(id + " buildTopLevel");

  if( tex_->nlevels() > 1 && gui_multi_level_.get() == 1){
    gui_multi_level_.set(tex_->nlevels());
    gui->execute(id + " build_multi_level");
  } else if(tex_->nlevels() == 1 && gui_multi_level_.get() > 1){
    gui_multi_level_.set(1);
    gui->execute(id + " destroy_multi_level");
  }

  if(!slice_ren_) {
    slice_ren_ = new SliceRenderer(tex_, cmap1, cmap2,
				   int(card_mem_*1024*1024*0.8));
    slice_ren_->
      set_control_point(tex_->transform().unproject(control_widget_->
						    ReferencePoint()));
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
    old_cmap1_ = cmap1;
    old_cmap2_ = cmap2;
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
      ddx_ = t.project(Vector(1.0/(nx-1), 0, 0));
      ddy_ = t.project(Vector(0, 1.0/(ny-1), 0));
      ddz_ = t.project(Vector(0, 0, 1.0/(nz-1)));
      ddview_ = (dv.length()/(std::max(nx, std::max(ny,nz)) -1));
      if(!b.inside(control_widget_->GetPosition())) {
	control_widget_->SetPosition(Interpolate(b.min(), b.max(), 0.5));
      }
      control_widget_->SetScale(dv.length()/80.0);
      slice_ren_->set_texture(tex_);
      slice_ren_->set_control_point(tex_->transform().unproject(control_widget_->ReferencePoint()));
    }

    if (cmap1 != old_cmap1_)
    {
      slice_ren_->set_colormap1(cmap1);
      old_cmap1_ = cmap1;
    }
    if (cmap2 != old_cmap2_)
    {
      slice_ren_->set_colormap2(cmap2);
      old_cmap2_ = cmap2;
    }
  }

  if(gui_multi_level_.get() > 1 && gui_color_changed_.get() == 1){
    gui_color_changed_.set(0);
    string outline_colors;
    gui->eval(id+" getOutlineColors", outline_colors);

    istringstream is( outline_colors );
    // Slurp in the rgb values.
    unsigned int rgbsize;
    is >> rgbsize;
    vector< Color > rgbs(rgbsize);
    for (int i = 0; i < rgbsize; i++)
    {
      double r, g, b;
      is >> r >> g >> b;
      rgbs[i] = Color(r, g, b);
    }
    slice_ren_->set_outline_colors( rgbs );
  }

  slice_ren_->set_interp(bool(interp_mode_.get()));
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
  slice_ren_->set_stencil( bool(gui_use_stencil_.get()) );
  slice_ren_->set_level_outline( bool (gui_outline_levels_.get()));
  cyl_active_.reset();
  draw_phi0_.reset();
  phi0_.reset();
  draw_phi1_.reset();
  phi1_.reset();
  slice_ren_->set_cylindrical(cyl_active_.get(), draw_phi0_.get(), phi0_.get(), 
                              draw_phi1_.get(), phi1_.get());

  if( tex_->nlevels() > 1 ){
    for(int i = 0; i < tex_->nlevels(); i++){
      string result;
      gui->eval(id + " isOn l" + to_string(i), result);
      if ( result == "0"){
  	slice_ren_->set_draw_level(tex_->nlevels()-1 -i, false);
      } else {
  	slice_ren_->set_draw_level(tex_->nlevels()-1 -i, true);
      }
    }
  }
  
  ogeom_->flushViews();		  
  
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
  } else if (args[1] == "color_changed") {
    color_changed_ = true;

//     def_color_r_.reset();
//     def_color_g_.reset();
//     def_color_b_.reset();
//     def_color_a_.reset();
//     def_material_->diffuse =
//       Color(def_color_r_.get(), def_color_g_.get(), def_color_b_.get());
//     def_material_->transparency = def_color_a_.get();
				  
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

} // namespace SCIRun
