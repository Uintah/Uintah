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
//    File   : TextureBuilder.cc
//    Author : Milan Ikits
//    Date   : Fri Jul 16 00:11:18 2004

#include <sci_defs/ogl_defs.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Datatypes/LatVolField.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Malloc/Allocator.h>

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>

#include <Core/Volume/VideoCardInfo.h>
#include <Dataflow/Ports/TexturePort.h>
#include <Core/Geom/ShaderProgramARB.h>
#include <Core/Algorithms/Visualization/TextureBuilderAlgo.h>
#include <Core/Util/DebugStream.h>

#include <Core/Datatypes/MRLatVolField.h>

namespace SCIRun {

static SCIRun::DebugStream dbg("TextureBuilder", false);

class TextureBuilder : public Module
{
public:
  TextureBuilder(GuiContext*);
  virtual ~TextureBuilder();

  virtual void execute();

private:
  TextureHandle texture_;

  GuiDouble gui_vminval_;
  GuiDouble gui_vmaxval_;
  GuiDouble gui_gminval_;
  GuiDouble gui_gmaxval_;

  GuiInt gui_fixed_;
  GuiInt gui_card_mem_;
  GuiInt gui_card_mem_auto_;
  int card_mem_;

  int vfield_last_generation_;
  int gfield_last_generation_;
  double vminval_, vmaxval_;
  double gminval_, gmaxval_;

  bool build_texture(FieldHandle, FieldHandle);

  bool new_vfield(ScalarFieldInterfaceHandle sfi);
  bool new_gfield(FieldHandle field);
};


DECLARE_MAKER(TextureBuilder)

TextureBuilder::TextureBuilder(GuiContext* ctx)
  : Module("TextureBuilder", ctx, Source, "Visualization", "SCIRun"),
    texture_(new Texture),
    gui_vminval_(ctx->subVar("vmin")),
    gui_vmaxval_(ctx->subVar("vmax")),
    gui_gminval_(ctx->subVar("gmin")),
    gui_gmaxval_(ctx->subVar("gmax")),
    gui_fixed_(ctx->subVar("is_fixed")),
    gui_card_mem_(ctx->subVar("card_mem")),
    gui_card_mem_auto_(ctx->subVar("card_mem_auto")),
    card_mem_(video_card_memory_size()),
    vfield_last_generation_(-1), gfield_last_generation_(-1)
{}

TextureBuilder::~TextureBuilder()
{}


void
TextureBuilder::execute()
{
  if (card_mem_ != 0 && gui_card_mem_auto_.get())
  {
    gui_card_mem_.set(card_mem_);
  }
  else if (card_mem_ == 0)
  {
    gui_card_mem_auto_.set(0);
  }

  FieldIPort* ivfield = (FieldIPort *)get_iport("Scalar Field");
  FieldIPort* igfield = (FieldIPort*)get_iport("Gradient Field");
  TextureOPort* otexture = (TextureOPort *)get_oport("Texture");

  FieldHandle vfield;
  ivfield->get(vfield);
  if (!vfield.get_rep())
  {
    error("Field has no representation.");
    return;
  }

  if ((vfield->generation != vfield_last_generation_) ||
      (gui_vminval_.get() != vminval_) || 
      (gui_vmaxval_.get() != vmaxval_))
  {
    // new field or range change
    ScalarFieldInterfaceHandle sfi = vfield->query_scalar_interface(this);
    if (!sfi.get_rep())
    {
      error("Input scalar field does not contain scalar data.");
      return;
    }

    // Warning::Temporary Hack!
    // In order to get the colors mapped correctly we need the min and max
    // values for every level of a Multi-level field.  This
    // temporary solution needs to be flushed out either in the sfi->minmax
    // algorithm or someplace else. We don't want to have to handle every
    // posible scalar type here.
    if( MRLatVolField<double>* vmrfield =
        dynamic_cast< MRLatVolField< double >* > (vfield.get_rep()) ) {
      double minv = DBL_MAX, maxv = -DBL_MAX;
      for(int i = 0 ; i < vmrfield->nlevels(); i++ ){
        const MultiResLevel<double>* lev = vmrfield->level( i );
        for(unsigned int j = 0; j < lev->patches.size(); j++ ){
          // Each patch in a level corresponds to a LatVolField.
          // Grab the field.
          LatVolField<double>* vmr = lev->patches[j].get_rep(); 
          // Now, get the min_max for the scalar field.
          ScalarFieldInterfaceHandle sub_sfi =
            vmr->query_scalar_interface(this);
          new_vfield(sub_sfi);
          minv = Min(vminval_, minv);
          maxv = Max(vmaxval_, maxv);
        }
        // Set the values
        gui_vminval_.set(minv);  vminval_ = minv;
        gui_vmaxval_.set(maxv);  vmaxval_ = maxv;
      }
    } else {
      new_vfield( sfi );
    }
    vfield_last_generation_ = vfield->generation;

  }


  FieldHandle gfield = 0;
  if (igfield)
  {
    igfield->get(gfield);
    if (gfield.get_rep())
    {
      if (!ShaderProgramARB::shaders_supported())
      {
        // TODO: Runtime check, change message to reflect that.
        warning("This machine does not support advanced volume rendering.  The gradient field will be ignored.");
        gfield = 0;
      }
      else
      {
        if (gfield->generation != gfield_last_generation_)
        {
          // new field
          if (!new_gfield(gfield)) return;
          gfield_last_generation_ = gfield->generation;
        }
        // this field must share a mesh and must have the same basis_order.
        if (vfield->basis_order() != gfield->basis_order())
        {
          error("both input fields must have the same basis order.");
          return;
        }
        if (vfield->mesh().get_rep() != gfield->mesh().get_rep())
        {
          error("both input fields must share a mesh.");
          return;
        }
      }
    }
  }
  
  if (build_texture(vfield, gfield))
  {
    otexture->send(texture_);
  }
}


bool
TextureBuilder::build_texture(FieldHandle vfield, FieldHandle gfield)
{
  // start new algorithm based code
  const TypeDescription* td = vfield->get_type_description();
  LockingHandle<TextureBuilderAlgoBase> builder;
  CompileInfoHandle ci = TextureBuilderAlgoBase::get_compile_info(td);
  if (!DynamicCompilation::compile(ci, builder, this))
  {
    error("Texture Builder can not work with this type of field.");
    return false;
  }
  builder->build(texture_, vfield, vminval_, vmaxval_,
                 gfield, gminval_, gmaxval_, gui_card_mem_.get());
  return true;
}


bool
TextureBuilder::new_vfield(ScalarFieldInterfaceHandle sfi)
{
  if( gui_fixed_.get() ){
    vminval_ = gui_vminval_.get();
    vmaxval_ = gui_vmaxval_.get();
  } else {
    // set vmin/vmax
    pair<double, double> vminmax;
    sfi->compute_min_max(vminmax.first, vminmax.second);
    if(vminmax.first != vminval_ || vminmax.second != vmaxval_) {
      gui_vminval_.set(vminmax.first);
      gui_vmaxval_.set(vminmax.second);
      vminval_ = vminmax.first;
      vmaxval_ = vminmax.second;
    }
  }
  return true;
}


bool
TextureBuilder::new_gfield(FieldHandle gfield)
{
  // set gmin/gmax
  LatVolField<Vector>* gfld =
    dynamic_cast<LatVolField<Vector>*>(gfield.get_rep());

  if (!gfld)
  {
    error("Input gradient field does not contain vector data.");
    return false;
  }

  FData3d<Vector>::const_iterator bi, ei;
  bi = gfld->fdata().begin();
  ei = gfld->fdata().end();
  double gminval = DBL_MAX;
  double gmaxval = DBL_MIN;
  // Moved sqrt out of loop, do after.
  while (bi != ei)
  {
    const double g = (*bi).length2();
    if (g < gminval) gminval = g;
    if (g > gmaxval) gmaxval = g;
    ++bi;
  }
  if (gmaxval >= 0)
  {
    gminval = sqrt(gminval);
    gmaxval = sqrt(gmaxval);
  }
  if (!gui_fixed_.get())
  {
    gui_gminval_.set(gminval);
    gui_gmaxval_.set(gmaxval);
  }
  gminval_ = gminval;
  gmaxval_ = gmaxval;
  return true;
}


} // end namespace SCIRun
