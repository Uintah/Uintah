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

#include <Core/Datatypes/MRLatVolField.h>

namespace SCIRun {



class TextureBuilder : public Module
{
public:
  TextureBuilder(GuiContext*);
  virtual ~TextureBuilder();

  virtual void execute();

private:
  TextureHandle tHandle_;

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
};


DECLARE_MAKER(TextureBuilder)

TextureBuilder::TextureBuilder(GuiContext* ctx)
  : Module("TextureBuilder", ctx, Source, "Visualization", "SCIRun"),
    tHandle_(new Texture),
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
  bool update = false;

  if (card_mem_ != 0 && gui_card_mem_auto_.get())
    gui_card_mem_.set(card_mem_);
  else if (card_mem_ == 0)
    gui_card_mem_auto_.set(0);

  FieldHandle vHandle;
  FieldHandle gHandle;

  // Get a handle to the input field port.
  FieldIPort* vfield_port = (FieldIPort *) get_iport("Scalar Field");

  // The scalar field input is required.
  if (!vfield_port->get(vHandle) || !(vHandle.get_rep()) ||
      !(vHandle->mesh().get_rep())) {
    error( "No scalar field handle or representation" );
    return;
  }

  if( vHandle->get_type_description(0)->get_name() != "LatVolField" &&
      vHandle->get_type_description(0)->get_name() != "MRLatVolField" &&
      vHandle->get_type_description(0)->get_name() != "ITKLatVolField" ) {
      
    error( "Only availible for regular topology with uniformly gridded data." );
    return;
  }

  // The input field must contain scalar data.
  ScalarFieldInterfaceHandle sfi = vHandle->query_scalar_interface(this);
  if (!sfi.get_rep()) {
    error("Input scalar field does not contain scalar data.");
    return;
  }

  if( !gui_fixed_.get() ){

    double vmin = DBL_MAX, vmax = -DBL_MAX;

    if( vHandle->get_type_description(0)->get_name() == "MRLatVolField" ) {
      // Warning::Temporary Hack!
      // In order to get the colors mapped correctly we need the min and max
      // values for every level of a Multi-level field.  This
      // temporary solution needs to be flushed out either in the sfi->minmax
      // algorithm or someplace else. We don't want to have to handle every
      // posible scalar type here.
      if( vHandle->get_type_description(1)->get_name() == "double" ) {
	MRLatVolField<double>* vmrfield = 
	  (MRLatVolField< double > *) vHandle.get_rep();

	for(int i = 0 ; i < vmrfield->nlevels(); i++ ){
	  const MultiResLevel<double>* lev = vmrfield->level( i );
	  for(unsigned int j = 0; j < lev->patches.size(); j++ ){
	    // Each patch in a level corresponds to a LatVolField.
	    // Grab the field.
	    LatVolField<double>* vmr = lev->patches[j].get_rep(); 
	    // Now, get the min_max for the scalar field.
	    ScalarFieldInterfaceHandle sub_sfi =
	      vmr->query_scalar_interface(this);

	    // set vmin/vmax
	    double vmintmp, vmaxtmp;
	    sfi->compute_min_max(vmintmp, vmaxtmp);

	    if( vmin > vmintmp ) vmin = vmintmp;
	    if( vmax < vmaxtmp ) vmax = vmintmp;
	  }
	}
      } else {
	error("Input scalar field, MRLatVolField does not contain double data.");
	return;
      }
    } else {
      sfi->compute_min_max(vmin, vmax);
    }

    gui_vminval_.set(vmin);
    gui_vmaxval_.set(vmax);
  }

  // Check to see if the input scalar field or the range has changed.
  if( vfield_last_generation_ != vHandle->generation  ||
      (gui_vminval_.get() != vminval_) || 
      (gui_vmaxval_.get() != vmaxval_)) {

    vfield_last_generation_ = vHandle->generation;
    
    vminval_ = gui_vminval_.get();
    vmaxval_ = gui_vmaxval_.get();
    
    update = true;
  }

  // Get a handle to the input gradient port.
  FieldIPort* gfield_port = (FieldIPort *) get_iport("Gradient Field");

  // The gradient field input is optional.
  if (gfield_port->get(gHandle) && gHandle.get_rep()) {
    
    if (!ShaderProgramARB::shaders_supported()) {
      // TODO: Runtime check, change message to reflect that.
      warning("This machine does not support advanced volume rendering. The gradient field will be ignored.");
      if( gfield_last_generation_ != -1 )
	update = true;
      gHandle = 0;
      gfield_last_generation_ = -1;

    } else {

      if( gHandle->get_type_description(0)->get_name() != "LatVolField" &&
	  gHandle->get_type_description(0)->get_name() != "MRLatVolField" &&
	  gHandle->get_type_description(0)->get_name() != "ITKLatVolField" ) {
    
	error( "Only availible for regular topology with uniformly gridded data." );
	return;
      }
 
      VectorFieldInterfaceHandle vfi = gHandle->query_vector_interface(this);
      if (!vfi.get_rep()) {
	error("Input gradient field does not contain vector data.");
	return;
      }

      // The gradient and the scalar fields must share a mesh.
      if (gHandle->mesh().get_rep() != vHandle->mesh().get_rep()) {
	error("both input fields must share a mesh.");
	return;
      }
 
      // The scalar and the gradient fields must have the same basis_order.
      if (gHandle->basis_order() != vHandle->basis_order()) {
	error("both input fields must have the same basis order.");
	return;
      }

      if (!gui_fixed_.get()) {

	// set gmin/gmax
	double gmin, gmax;
	vfi->compute_min_max(gmin, gmax);

	gui_gminval_.set(gmin);
	gui_gmaxval_.set(gmax);
      }

      // Check to see if the input gradient field has changed.
      if( gfield_last_generation_ != gHandle->generation  ||
	  (gui_gminval_.get() != gminval_) || 
	  (gui_gmaxval_.get() != gmaxval_) ) {
	gfield_last_generation_ = gHandle->generation;
	
	gminval_ = gui_gminval_.get();
 	gmaxval_ = gui_gmaxval_.get();
	
	update = true;
      }
    }
  } else {
    if( gfield_last_generation_ != -1 )
      update = true;
    gfield_last_generation_ = -1;
  }

  if( update ) {
    const TypeDescription* vftd = vHandle->get_type_description();
    const TypeDescription* gftd = (gHandle.get_rep() ?
				   gHandle->get_type_description(0) :
				   vHandle->get_type_description(0));

    CompileInfoHandle ci = TextureBuilderAlgo::get_compile_info(vftd, gftd);

    Handle<TextureBuilderAlgo> algo;
    if (!module_dynamic_compile(ci, algo)) return;

    algo->build(tHandle_,
		vHandle, vminval_, vmaxval_,
		gHandle, gminval_, gmaxval_,
		gui_card_mem_.get());
  }

  // Get a handle to the output texture port.
  if( tHandle_.get_rep() ) {
    TextureOPort* otexture_port = (TextureOPort *)get_oport("Texture");

    if (!otexture_port) {
      error("Unable to initialize oport 'Texture'.");
      return;
    }

    otexture_port->send(tHandle_);
  }
}

} // end namespace SCIRun
