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
//    File   : VolumeVisualizer.cc
//    Author : Milan Ikits
//    Author : Kurt Zimmerman
//    Date   : Sat Jul 10 21:55:34 2004

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

#include <Core/Volume/VolumeRenderer.h>
#include <Dataflow/Ports/TexturePort.h>
#include <Dataflow/Ports/Colormap2Port.h>
#include <Core/Volume/VideoCardInfo.h>
#include <Core/Geom/ShaderProgramARB.h>
#include <Core/Geom/GeomSticky.h>

#include <Dataflow/Widgets/ArrowWidget.h>
#include <Core/Thread/CrowdMonitor.h>

#include <iostream>
#include <sstream>
#include <string>
#ifdef __sgi
#include <ios>
#endif
#include <algorithm>
#include <Core/Volume/Utils.h>

namespace SCIRun {

class VolumeVisualizer : public Module {
public:
  VolumeVisualizer(GuiContext*);
  virtual ~VolumeVisualizer();
  virtual void execute();
  virtual void widget_moved(bool last, BaseWidget*);

private:
  TextureHandle texture_;

  TextureIPort* texture_iport_;
  ColorMapIPort* cmap1_iport_;
  GeometryOPort* geom_oport_;
  ColorMapOPort* cmap_oport_;
  int cmap1_prevgen_;
  vector<int> cmap2_prevgen_;
  vector<ColorMap2Handle> cmap2_;
  int tex_prevgen_;
  int card_mem_;
   



  CrowdMonitor widget_lock_;
  vector<int> widget_id_;
  vector<ArrowWidget*> widget_;
  vector<GeomHandle>       widget_switch_;
  vector<Plane>	clipping_planes_;
  map<ArrowWidget *, int> widget_plane_index_;


  GuiDouble gui_sampling_rate_hi_;
  GuiDouble gui_sampling_rate_lo_;
  GuiDouble gui_gradient_min_;
  GuiDouble gui_gradient_max_;;
  GuiInt gui_adaptive_;
  GuiInt gui_cmap_size_;
  GuiInt gui_sw_raster_;
  GuiInt gui_render_style_;
  GuiDouble gui_alpha_scale_;
  GuiInt gui_interp_mode_;
  GuiInt gui_shading_;
  GuiDouble gui_ambient_;
  GuiDouble gui_diffuse_;
  GuiDouble gui_specular_;
  GuiDouble gui_shine_;
  GuiInt gui_light_;
  GuiInt gui_blend_res_;
  GuiInt gui_multi_level_;
  GuiInt gui_use_stencil_;
  GuiInt gui_invert_opacity_;
  GuiInt gui_level_flag_;
  GuiInt gui_num_slices_;  // unused except for backwards compatability
  GuiInt gui_num_clipping_planes_;
  GuiInt gui_show_clipping_widgets_;
  GuiString gui_level_on_; // used for saving to net
  GuiString gui_level_vals_; // used for saving to net
  VolumeRenderer* volren_;
};


DECLARE_MAKER(VolumeVisualizer)
VolumeVisualizer::VolumeVisualizer(GuiContext* ctx)
  : Module("VolumeVisualizer", ctx, Source, "Visualization", "SCIRun"),
    texture_(0),
    texture_iport_((TextureIPort*)get_iport("Texture")),
    cmap1_iport_((ColorMapIPort*)get_iport("ColorMap")),
    geom_oport_((GeometryOPort*)get_oport("Geometry")),
    cmap_oport_((ColorMapOPort*)get_oport("ColorMap")),
    cmap1_prevgen_(0),
    cmap2_prevgen_(0),
    cmap2_(0),
    tex_prevgen_(0),
    card_mem_(video_card_memory_size()),
    widget_lock_("Clipping planes widget lock"),
    widget_id_(),
    widget_(),
    widget_switch_(),
    clipping_planes_(),
    widget_plane_index_(),
    gui_sampling_rate_hi_(ctx->subVar("sampling_rate_hi")),
    gui_sampling_rate_lo_(ctx->subVar("sampling_rate_lo")),
    gui_gradient_min_(ctx->subVar("gradient_min")),
    gui_gradient_max_(ctx->subVar("gradient_max")),
    gui_adaptive_(ctx->subVar("adaptive")),
    gui_cmap_size_(ctx->subVar("cmap_size")),
    gui_sw_raster_(ctx->subVar("sw_raster")),
    gui_render_style_(ctx->subVar("render_style")),
    gui_alpha_scale_(ctx->subVar("alpha_scale")),
    gui_interp_mode_(ctx->subVar("interp_mode")),
    gui_shading_(ctx->subVar("shading")),
    gui_ambient_(ctx->subVar("ambient")),
    gui_diffuse_(ctx->subVar("diffuse")),
    gui_specular_(ctx->subVar("specular")),
    gui_shine_(ctx->subVar("shine")),
    gui_light_(ctx->subVar("light")),
    gui_blend_res_(ctx->subVar("blend_res")),
    gui_multi_level_(ctx->subVar("multi_level")),
    gui_use_stencil_(ctx->subVar("use_stencil")),
    gui_invert_opacity_(ctx->subVar("invert_opacity")),
    gui_level_flag_(ctx->subVar("show_level_flag", false)),
    gui_num_slices_(ctx->subVar("num_slices", false)), // dont save
    gui_num_clipping_planes_(ctx->subVar("num_clipping_planes"), 2),
    gui_show_clipping_widgets_(ctx->subVar("show_clipping_widgets")),
    gui_level_on_(ctx->subVar("level_on")),
    gui_level_vals_(ctx->subVar("level_vals")),
    volren_(0)
{}

VolumeVisualizer::~VolumeVisualizer()
{}

void
VolumeVisualizer::widget_moved(bool release, BaseWidget *widget)
{
  ArrowWidget *arrow = dynamic_cast<ArrowWidget*>(widget);
  if (!arrow) return;
  map<ArrowWidget *, int>::iterator pos = widget_plane_index_.find(arrow);
  if (pos == widget_plane_index_.end()) return;
  const int idx = pos->second;
  if (idx < 0 || idx >= int(clipping_planes_.size())) return;
  Point up = texture_->transform().unproject(arrow->GetPosition());
  clipping_planes_[idx].ChangePlane(up,-arrow->GetDirection());
}
  

void
VolumeVisualizer::execute()
{
  static Point oldmin(0,0,0), oldmax(0,0,0);
  static int oldni = 0, oldnj = 0, oldnk = 0;
  static GeomID geomID  = 0;
  
  if (!texture_iport_->get(texture_)) {
    warning("No texture, nothing done.");
    return;
  }
  else if (!texture_.get_rep()) {
    warning("No texture, nothing done.");
    return;
  }

  bool shading_state = false;
  if (ShaderProgramARB::shaders_supported())
  {
    shading_state = (texture_->nb(0) == 1);
  }

  gui->execute(id + " change_shading_state " + (shading_state?"0":"1"));
  
  ColorMapHandle cmap1;
  
  bool c1 = (cmap1_iport_->get(cmap1) && cmap1.get_rep());

  vector<int> cmap2_generation;
  port_range_type range = get_iports("ColorMap2");
  port_map_type::iterator pi = range.first;
  vector<ColorMap2Handle> cmap2;
  while (pi != range.second) {
    ColorMap2IPort *cmap2_iport = 
      dynamic_cast<ColorMap2IPort*>(get_iport(pi->second));
    ColorMap2Handle cmap2H;
    if (cmap2_iport && cmap2_iport->get(cmap2H) && cmap2H.get_rep()) {
      cmap2_generation.push_back(cmap2H->generation);
      cmap2.push_back(cmap2H.get_rep());
    }
    ++pi;
  }

    
  bool c2 = !cmap2.empty();

  if (c2)
  {
    if (!ShaderProgramARB::shaders_supported())
    {
      warning("ColorMap2 usage is not supported by this machine.");
      cmap2_.clear();
      c2 = false;
    }
    // Section disabled because we can now copmpute gradient on GPU
    //    else
    //    {
    
    //      if (texture_->nc() == 1)
    //      {
    //        warning("ColorMap2 requires gradient magnitude in the texture.");
    //        cmap2_.clear();
    //        c2 = false;
    //      }
    //  }
  }

  if (!c1 && !c2)
  {
    error("No colormap available to render.  Nothing drawn.");
    return;
  }

  bool cmap1_dirty = false;
  if(c1 && (cmap1->generation != cmap1_prevgen_)) {
    cmap1_prevgen_ = cmap1->generation;
    cmap1_dirty = true;
  }

  bool cmap2_dirty = cmap2_generation.size() != cmap2_prevgen_.size();
  if(c2 && !cmap2_dirty) {
    for (unsigned int g = 0; g < cmap2_generation.size(); ++g) {
      cmap2_dirty = cmap2_generation[g] != cmap2_prevgen_[g];
      if (cmap2_dirty) break;
    }
  }
  cmap2_prevgen_ = cmap2_generation;

  bool tex_dirty = false;
  if (texture_.get_rep() && texture_->generation != tex_prevgen_) {
    tex_prevgen_ = texture_->generation;
    tex_dirty = true;
  }
   
  if (!cmap1_dirty && !cmap2_dirty && !tex_dirty && 
      !gui_sampling_rate_hi_.changed() && !gui_sampling_rate_lo_.changed() &&
      !gui_adaptive_.changed() && !gui_cmap_size_.changed() && 
      !gui_sw_raster_.changed() && !gui_render_style_.changed() &&
      !gui_alpha_scale_.changed() && !gui_interp_mode_.changed() &&
      !gui_shading_.changed() && !gui_ambient_.changed() &&
      !gui_diffuse_.changed() && !gui_specular_.changed() &&
      !gui_shine_.changed() && !gui_light_.changed() &&
      !gui_blend_res_.changed() && !gui_multi_level_.changed() &&
      !gui_use_stencil_.changed() && !gui_invert_opacity_.changed() &&
      !gui_level_flag_.changed())
  {
    if (texture_.get_rep())
    {
      for (unsigned int i = 0; i < texture_->bricks().size(); i++)
      {
	if (texture_->bricks()[i]->dirty())
	{
	  geom_oport_->flushViews();
	  break;
	}
      }
    }
    cerr << "Early exit\n";
    return;
  }


  string s;
  gui->eval(id + " hasUI", s);
  if( s == "0" )
    gui->execute(id + " buildTopLevel");

  if( texture_->nlevels() > 1 && gui_multi_level_.get() == 1){
    gui_multi_level_.set(texture_->nlevels());
    gui->execute(id + " build_multi_level");
  } else if(texture_->nlevels() == 1 && gui_multi_level_.get() > 1){
    gui_multi_level_.set(1);
    gui->execute(id + " destroy_multi_level");
  }

  cmap2_ = cmap2;
  if(!volren_) {
    volren_ = new VolumeRenderer(texture_, cmap1, cmap2_, clipping_planes_, int(card_mem_*1024*1024*0.8));
    oldmin = texture_->bbox().min();
    oldmax = texture_->bbox().max();
    oldni = texture_->nx();
    oldnj = texture_->ny();
    oldnk = texture_->nz();
    //    geom_oport_->delAll();
    geomID = geom_oport_->addObj(volren_, "VolumeRenderer Transparent");
  } else {
    volren_->set_texture(texture_);
    if(c1 && cmap1_dirty)
      volren_->set_colormap1(cmap1);
    else if (!c1)
      volren_->set_colormap1(0);
    if(c2 && cmap2_dirty)
      volren_->set_colormap2(cmap2_);
    else if (!c2)
      volren_->set_colormap2(cmap2_);
    int ni = texture_->nx();
    int nj = texture_->ny();
    int nk = texture_->nz();
    if(oldmin != texture_->bbox().min() || oldmax != texture_->bbox().max() ||
       ni != oldni || nj != oldnj || nk != oldnk) {
      geom_oport_->delObj(geomID);
      geomID = geom_oport_->addObj(volren_, "VolumeRenderer Transparent");
      oldni = ni; oldnj = nj; oldnk = nk;
      oldmin = texture_->bbox().min();
      oldmax = texture_->bbox().max();
    }
  }
 
  volren_->set_interp(bool(gui_interp_mode_.get()));

  switch(gui_render_style_.get()) {
  case 0:
    volren_->set_mode(VolumeRenderer::MODE_OVER);
    break;
  case 1:
    volren_->set_mode(VolumeRenderer::MODE_MIP);
    break;
  default:
    warning("Unsupported blend mode.  Using over.");
    volren_->set_mode(VolumeRenderer::MODE_OVER);
    gui_render_style_.set(0);
    break;
  }

  if (gui_num_slices_.get() > 0)
  {
    const double rate = volren_->num_slices_to_rate(gui_num_slices_.get());
    gui_sampling_rate_hi_.set(rate);
    gui_num_slices_.set(-1);
  }
  
  volren_->set_gradient_range(gui_gradient_min_.get(), 
			      gui_gradient_max_.get());
  volren_->set_sampling_rate(gui_sampling_rate_hi_.get());
  volren_->set_interactive_rate(gui_sampling_rate_lo_.get());
  volren_->set_adaptive(gui_adaptive_.get());
  volren_->set_colormap_size(1 << gui_cmap_size_.get());
  volren_->set_slice_alpha(gui_alpha_scale_.get());
  volren_->set_stencil( bool(gui_use_stencil_.get()) );
  volren_->invert_opacity( bool(gui_invert_opacity_.get()));
  if( texture_->nlevels() > 1 ){
    for(int i = 0; i < texture_->nlevels(); i++){
      string result;
      gui->eval(id + " isOn l" + to_string(i), result);
      if ( result == "0")
	volren_->set_draw_level(texture_->nlevels()-1 -i, false);
      else 
	volren_->set_draw_level(texture_->nlevels()-1 -i, true);


      gui->eval(id + " alphaVal s" + to_string(i), result);
      double val;
      if( string_to_double( result, val )){
	volren_->set_level_alpha(texture_->nlevels()-1 -i, val);
      }
    }
  }

  if(!volren_->use_pbuffer()) {
    gui_sw_raster_.set(1);
  } else {
    volren_->set_sw_raster(gui_sw_raster_.get());
  }
  if(!volren_->use_blend_buffer()) {
    gui_blend_res_.set(8);
  } else {
    volren_->set_blend_num_bits(gui_blend_res_.get());
  }

  volren_->set_shading(gui_shading_.get());
  volren_->set_material(gui_ambient_.get(), gui_diffuse_.get(),
                        gui_specular_.get(), gui_shine_.get());
  volren_->set_light(gui_light_.get());



  BBox bbox;
  texture_->get_bounds(bbox);;
  gui_num_clipping_planes_.set(cmap2_.size()-1);
  while (gui_num_clipping_planes_.get() >= 0 &&
         gui_num_clipping_planes_.get() != (int)widget_.size())
  {
    double scale = (bbox.diagonal()).length()/30.0;
    Point p = (bbox.min()+bbox.diagonal()/2.0);

    ArrowWidget *widget = scinew ArrowWidget(this, &widget_lock_, scale, true);
    widget_.push_back(widget);
    widget->Connect(geom_oport_);
    widget->SetCurrentMode(0);
    GeomSwitch *swtch = (GeomSwitch *)widget->GetWidget().get_rep();
    swtch->set_state(1);
    //    GeomHandle obj = 
    //      scinew GeomStickyRotate(widget->GetWidget(), p, -bbox.diagonal());
    //    obj = scinew GeomSwitch(obj);
    widget_switch_.push_back(swtch);
    ((GeomSwitch *)(widget_switch_.back().get_rep()))->set_state(1);
    widget_id_.push_back(geom_oport_->addObj(widget_switch_.back(),
					     "Clipping plane " +
					     to_string(widget_.size()),
					     &widget_lock_));

    widget->SetDirection(bbox.diagonal());
    widget->SetScale(scale);
    widget->SetLength(scale*2);
    widget->Move(p);
    widget->redraw();
    widget_plane_index_[widget] = clipping_planes_.size();
    clipping_planes_.push_back(Plane(1,1,1,1));
    Point up = texture_->transform().unproject(widget->GetPosition());
    clipping_planes_.back().ChangePlane(up,-bbox.diagonal());
  }


  for (unsigned int w = 0; w < widget_switch_.size(); ++w) 
    ((GeomSwitch*)widget_switch_[w].get_rep())->set_state(gui_show_clipping_widgets_.get());

  
  geom_oport_->flushViews();				  

  if(c1)
  {
    ColorMapHandle outcmap;
    outcmap = new ColorMap(*cmap1.get_rep()); 
    outcmap->Scale(texture_->vmin(), texture_->vmax());
    cmap_oport_->send_and_dereference(outcmap);
  }
}

} // End namespace SCIRun
