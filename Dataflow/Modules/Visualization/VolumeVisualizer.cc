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

private:
  TextureHandle tex;

  TextureIPort* intexture;
  ColorMapIPort* icmap1;
  ColorMap2IPort* icmap2;
  GeometryOPort* ogeom;
  ColorMapOPort* ocmap;
  int cmap1_prevgen_;
  int cmap2_prevgen_;
  int tex_prevgen_;
  int card_mem_;
   
  CrowdMonitor control_lock; 
  PointWidget* control_widget;
  GeomID control_id;

  GuiDouble gui_sampling_rate_hi_;
  GuiDouble gui_sampling_rate_lo_;
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
  GuiInt gui_multi_level_;
  GuiInt gui_use_stencil_;
  GuiInt gui_invert_opacity_;
  GuiInt gui_num_slices_;  // unused except for backwards compatability

  VolumeRenderer* volren_;
};


DECLARE_MAKER(VolumeVisualizer)
VolumeVisualizer::VolumeVisualizer(GuiContext* ctx)
  : Module("VolumeVisualizer", ctx, Source, "Visualization", "SCIRun"),
    tex(0),
    cmap1_prevgen_(0),
    cmap2_prevgen_(0),
    tex_prevgen_(0),
    card_mem_(video_card_memory_size()),
    control_lock("VolumeVisualizer resolution lock"),
    control_widget(0),
    control_id(-1),
    gui_sampling_rate_hi_(ctx->subVar("sampling_rate_hi")),
    gui_sampling_rate_lo_(ctx->subVar("sampling_rate_lo")),
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
    gui_multi_level_(ctx->subVar("multi_level")),
    gui_use_stencil_(ctx->subVar("use_stencil")),
    gui_invert_opacity_(ctx->subVar("invert_opacity")),
    gui_num_slices_(ctx->subVar("num_slices", false)), // don't save
    volren_(0)
{}

VolumeVisualizer::~VolumeVisualizer()
{}

void
VolumeVisualizer::execute()
{
  static Point oldmin(0,0,0), oldmax(0,0,0);
  static int oldni = 0, oldnj = 0, oldnk = 0;
  static GeomID geomID  = 0;
  
  intexture = (TextureIPort*)get_iport("Texture");
  icmap1 = (ColorMapIPort*)get_iport("ColorMap");
  icmap2 = (ColorMap2IPort*)get_iport("ColorMap2");
  ogeom = (GeometryOPort*)get_oport("Geometry");
  ocmap = (ColorMapOPort*)get_oport("ColorMap");

  if (!intexture->get(tex)) {
    warning("No texture, nothing done.");
    return;
  }
  else if (!tex.get_rep()) {
    warning("No texture, nothing done.");
    return;
  }

  bool shading_state = false;
  if (ShaderProgramARB::shaders_supported())
  {
    shading_state = (tex->nb(0) == 1);
  }

  gui->execute(id + " change_shading_state " + (shading_state?"0":"1"));
  
  ColorMapHandle cmap1;
  ColorMap2Handle cmap2;
  bool c1 = (icmap1->get(cmap1) && cmap1.get_rep());
  bool c2 = (icmap2->get(cmap2) && cmap2.get_rep());

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
      if (tex->nc() == 1)
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

  bool cmap1_dirty = false;
  if(c1 && (cmap1->generation != cmap1_prevgen_)) {
    cmap1_prevgen_ = cmap1->generation;
    cmap1_dirty = true;
  }

  bool cmap2_dirty = false;
  if(c2 && (cmap2->generation != cmap2_prevgen_)) {
    cmap2_prevgen_ = cmap2->generation;
    cmap2_dirty = true;
  }

  bool tex_dirty = false;
  if (tex.get_rep() && tex->generation != tex_prevgen_) {
    tex_prevgen_ = tex->generation;
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
      !gui_multi_level_.changed() && !gui_use_stencil_.changed() &&
      !gui_invert_opacity_.changed() && !gui_num_slices_.changed())
  {
    if (tex.get_rep())
    {
      for (unsigned int i = 0; i < tex->bricks().size(); i++)
      {
	if (tex->bricks()[i]->dirty())
	{
	  ogeom->flushViews();
	  break;
	}
      }
    }
    return;
  }

  string s;
  gui->eval(id + " hasUI", s);
  if( s == "0" )
    gui->execute(id + " buildTopLevel");

  if( tex->nlevels() > 1 && gui_multi_level_.get() == 1){
    gui_multi_level_.set(tex->nlevels());
    gui->execute(id + " build_multi_level");
  } else if(tex->nlevels() == 1 && gui_multi_level_.get() > 1){
    gui_multi_level_.set(1);
    gui->execute(id + " destroy_multi_level");
  }

  if(!volren_) {
    volren_ = new VolumeRenderer(tex, cmap1, cmap2, int(card_mem_*1024*1024*0.8));
    oldmin = tex->bbox().min();
    oldmax = tex->bbox().max();
    oldni = tex->nx();
    oldnj = tex->ny();
    oldnk = tex->nz();
    //    ogeom->delAll();
    geomID = ogeom->addObj(volren_, "VolumeRenderer Transparent");
  } else {
    volren_->set_texture(tex);
    if(c1 && cmap1_dirty)
      volren_->set_colormap1(cmap1);
    else if (!c1)
      volren_->set_colormap1(0);
    if(c2 && cmap2_dirty)
      volren_->set_colormap2(cmap2);
    else if (!c2)
      volren_->set_colormap2(0);
    int ni = tex->nx();
    int nj = tex->ny();
    int nk = tex->nz();
    if(oldmin != tex->bbox().min() || oldmax != tex->bbox().max() ||
       ni != oldni || nj != oldnj || nk != oldnk) {
      ogeom->delObj(geomID);
      geomID = ogeom->addObj(volren_, "VolumeRenderer Transparent");
      oldni = ni; oldnj = nj; oldnk = nk;
      oldmin = tex->bbox().min();
      oldmax = tex->bbox().max();
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
  
  volren_->set_sampling_rate(gui_sampling_rate_hi_.get());
  volren_->set_interactive_rate(gui_sampling_rate_lo_.get());
  volren_->set_adaptive(gui_adaptive_.get());
  volren_->set_colormap_size(1 << gui_cmap_size_.get());
  volren_->set_slice_alpha(gui_alpha_scale_.get());
  volren_->set_stencil( bool(gui_use_stencil_.get()) );
  volren_->invert_opacity( bool(gui_invert_opacity_.get()));
  if( tex->nlevels() > 1 ){
    for(int i = 0; i < tex->nlevels(); i++){
      string result;
      gui->eval(id + " isOn l" + to_string(i), result);
      if ( result == "0")
	volren_->set_draw_level(tex->nlevels()-1 -i, false);
      else 
	volren_->set_draw_level(tex->nlevels()-1 -i, true);


      gui->eval(id + " alphaVal s" + to_string(i), result);
      double val;
      if( string_to_double( result, val )){
	volren_->set_level_alpha(tex->nlevels()-1 -i, val);
      }
    }
  }

  if(!volren_->use_pbuffer()) {
    gui_sw_raster_.set(1);
  } else {
    volren_->set_sw_raster(gui_sw_raster_.get());
  }

  volren_->set_shading(gui_shading_.get());
  volren_->set_material(gui_ambient_.get(), gui_diffuse_.get(),
                        gui_specular_.get(), gui_shine_.get());
  volren_->set_light(gui_light_.get());
  
  ogeom->flushViews();				  

  if(c1)
  {
    ColorMapHandle outcmap;
    outcmap = new ColorMap(*cmap1.get_rep()); 
    outcmap->Scale(tex->vmin(), tex->vmax());
    ocmap->send(outcmap);
  }
}

} // End namespace SCIRun
