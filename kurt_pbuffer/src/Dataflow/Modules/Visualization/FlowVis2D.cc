//
//  For more information, please see: http://software.sci.utah.edu
//
//  The MIT License
//
//  Copyright (c) 2005 Scientific Computing and Imaging Institute,
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
//    File   : TwoDFlowVis.cc
//    Author : Kurt Zimmerman
//    Date   : Tues April 19 10:36:18 2005


#include <Core/Geom/ShaderProgramARB.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/ImageMesh.h>

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/ColorMapPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Ports/GeometryPort.h>

#include <Core/Util/NotFinished.h>
#include <Core/Volume/FlowRenderer2D.h>
#include <Core/Volume/VideoCardInfo.h>


namespace SCIRun {

class FlowVis2D : public Module
{
public:
  FlowVis2D(GuiContext*);
  virtual ~FlowVis2D();

  virtual void execute();

private:
  int vfield_prev_generation_;
  int cmap_prev_generation_;
  int card_mem_;
  GuiInt gui_vis_type_;
  GuiInt gui_clear_;
  GuiInt gui_adv_accums_;
  GuiInt gui_conv_accums_;
  
  FlowRenderer2D* flowren_;

  void build2DVectorTexture();
};

DECLARE_MAKER(FlowVis2D)

FlowVis2D::FlowVis2D(GuiContext* ctx)
  : Module("FlowVis2D", ctx, Source, "Visualization", "SCIRun"),
    vfield_prev_generation_(-1),
    cmap_prev_generation_(-1),
    card_mem_(video_card_memory_size()),
    gui_vis_type_(ctx->subVar("vis_type")),
    gui_clear_(ctx->subVar("clear")),
    gui_adv_accums_(ctx->subVar("adv_accums")),
    gui_conv_accums_(ctx->subVar("conv_accums")),
    flowren_(0)
{}

FlowVis2D::~FlowVis2D()
{}


void
FlowVis2D::execute()
{
  static GeomID geomID  = 0;
//   static unsigned old_i = 0, old_j = 0;

//   if (card_mem_ != 0 && gui_card_mem_auto_.get())
//   {
//     gui_card_mem_.set(card_mem_);
//   }
//   else if (card_mem_ == 0)
//   {
//     gui_card_mem_auto_.set(0);
//   }

  FieldIPort* ivfield = (FieldIPort *)get_iport("Vector Slice");
  ColorMapIPort* icmap = (ColorMapIPort*)get_iport("ColorMap");
  GeometryOPort* ogeom = (GeometryOPort *)get_oport("Geometry");
  ColorMapOPort* ocmap = (ColorMapOPort*)get_oport("ColorMap");

  FieldHandle vfield;
  ivfield->get(vfield);
  if (!vfield.get_rep())
  {
    error("Field has no representation.");
    return;
  }

  bool field_dirty = false;
  ImageMeshHandle imh(0);

  if (vfield->generation != vfield_prev_generation_)
  {
    // new field or range change
    VectorFieldInterfaceHandle vfi = vfield->query_vector_interface(this);
    if (!vfi.get_rep())
    {
      error("Input field does not contain vector data.");
      return;
    }

    if( ImageMesh* im = dynamic_cast<ImageMesh *> ((vfield->mesh()).get_rep())) 
    {
      imh = im;
    } else {
      error("Input mesh is not an ImageMesh.");
      return;
    }
    vfield_prev_generation_ = vfield->generation;
    field_dirty = true;
  }

  // check for shaders
  if( !ShaderProgramARB::shaders_supported())
  {
    error("Shaders are not supported on this platform. Nothing drawn.");
    return;
  }


  ColorMapHandle cmap;
  bool have_cmap = (icmap->get(cmap) && cmap.get_rep());

  bool cmap_dirty = false;
  if( have_cmap && (cmap->generation != cmap_prev_generation_))
  {
    cmap_prev_generation_ = cmap->generation;
    cmap_dirty = true;
  }

  if( !flowren_ ){
    build2DVectorTexture();
    flowren_ = new FlowRenderer2D(vfield, cmap, int(card_mem_*1024*1024*0.8));
    geomID = ogeom->addObj(flowren_, "FlowRenderer");
  }

  if( gui_clear_.get() == 1 ){
    flowren_->reset();
    gui_clear_.set(0);
  }

  if( field_dirty )
  {
    flowren_->set_field( vfield );
  }

  if( cmap_dirty )
  {
    flowren_->set_colormap( cmap );
  }

  flowren_->set_adv_accums( gui_adv_accums_.get() );
  flowren_->set_conv_accums( gui_conv_accums_.get() );


  ogeom->flushViews();
}

void
FlowVis2D::build2DVectorTexture()
{
  NOT_FINISHED("FlowVis2D::build2DVectorTexture()");
}

}// end namespace SCIRun
