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
//    File   : NrrdTextureBuilder.cc
//    Author : Milan Ikits
//    Date   : Fri Jul 16 03:28:21 2004

#include <sci_defs/ogl_defs.h>
#include <Core/Containers/StringUtil.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Malloc/Allocator.h>

#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/Ports/NrrdPort.h>

#include <Core/Volume/VideoCardInfo.h>
#include <Dataflow/Network/Ports/TexturePort.h>
#include <Core/Geom/ShaderProgramARB.h>
#include <Core/Algorithms/Visualization/NrrdTextureBuilderAlgo.h>



namespace SCIRun {

class NrrdTextureBuilder : public Module
{
public:
  NrrdTextureBuilder(GuiContext*);
  virtual ~NrrdTextureBuilder();

  virtual void execute();

private:
  TextureHandle tHandle_;

  GuiDouble gui_vminval_;
  GuiDouble gui_vmaxval_;
  GuiDouble gui_gminval_;
  GuiDouble gui_gmaxval_;
  GuiDouble gui_mminval_;
  GuiDouble gui_mmaxval_;

  GuiInt gui_fixed_;
  GuiInt gui_card_mem_;
  GuiInt gui_card_mem_auto_;
  GuiInt gui_uchar_;

  int card_mem_;
  int is_uchar_;
  int vnrrd_last_generation_;
  int gnrrd_last_generation_;
  int mnrrd_last_generation_;
  double vminval_, vmaxval_;
  double gminval_, gmaxval_;
  double mminval_, mmaxval_;
};

DECLARE_MAKER(NrrdTextureBuilder)
  
NrrdTextureBuilder::NrrdTextureBuilder(GuiContext* ctx)
  : Module("NrrdTextureBuilder", ctx, Source, "Visualization", "SCIRun"),
    tHandle_(new Texture),
    gui_vminval_(get_ctx()->subVar("vmin"), 0),
    gui_vmaxval_(get_ctx()->subVar("vmax"), 1),
    gui_gminval_(get_ctx()->subVar("gmin"), 0),
    gui_gmaxval_(get_ctx()->subVar("gmax"), 1),
    gui_mminval_(get_ctx()->subVar("mmin"), 0),
    gui_mmaxval_(get_ctx()->subVar("mmax"), 1),
    gui_fixed_(get_ctx()->subVar("is_fixed"), 0),
    gui_card_mem_(get_ctx()->subVar("card_mem"), 16),
    gui_card_mem_auto_(get_ctx()->subVar("card_mem_auto"), 1),
    gui_uchar_(get_ctx()->subVar("is_uchar"), 1),
    card_mem_(video_card_memory_size()),
    is_uchar_(1),
    vnrrd_last_generation_(-1),
    gnrrd_last_generation_(-1),
    mnrrd_last_generation_(-1)
{}


NrrdTextureBuilder::~NrrdTextureBuilder()
{}


void
NrrdTextureBuilder::execute()
{
  bool update = false;

  if (card_mem_ != 0 && gui_card_mem_auto_.get())
    gui_card_mem_.set(card_mem_);
  else if (card_mem_ == 0)
    gui_card_mem_auto_.set(0);

  NrrdDataHandle vHandle;
  NrrdDataHandle gHandle;
  NrrdDataHandle mHandle;

  // Get a handle to the input nrrd port.
  NrrdIPort* vnrrd_port = (NrrdIPort *) get_iport("Scalar Nrrd");

  // The scalar nrrd input is required.
  if (!vnrrd_port->get(vHandle) || !(vHandle.get_rep())) {
    error( "No scalar nrrd handle or representation" );
    return;
  }

  Nrrd* nv_nrrd = vHandle->nrrd_;

  if (nv_nrrd->dim != 3 && nv_nrrd->dim != 4) {
    error("Invalid dimension for input value nrrd.");
    return;
  }

  size_t axis_size[4];
  nrrdAxisInfoGet_nva(nv_nrrd, nrrdAxisInfoSize, axis_size);
  if (nv_nrrd->dim == 4 && axis_size[0] != 1 && axis_size[0] != 4) {
    error("Invalid axis size for Normal/Value nrrd.");
    return;
  }

  // The input nrrd type must be unsigned char.
  if (gui_uchar_.get() && vHandle->nrrd_->type != nrrdTypeUChar) {
    error("Normal/Value input nrrd type must be unsigned char.");
    return;
  }

  if( !gui_fixed_.get() ){
    // set vmin/vmax
    NrrdRange *range = nrrdRangeNewSet(vHandle->nrrd_, nrrdBlind8BitRangeFalse);

    gui_vminval_.set(range->min);
    gui_vmaxval_.set(range->max);
    nrrdRangeNix(range);
  }

  // Check to see if the input nrrd has changed.
  if( vnrrd_last_generation_ != vHandle->generation  ||
      (gui_vminval_.get() != vminval_) || 
      (gui_vmaxval_.get() != vmaxval_))
  {
    vnrrd_last_generation_ = vHandle->generation;

    vminval_ = gui_vminval_.get();
    vmaxval_ = gui_vmaxval_.get();

    update = true;
  }

  // Get a handle to the input gradient port.
  NrrdIPort* gnrrd_port = (NrrdIPort *) get_iport("Gradient Nrrd");
  
  // The gradient nrrd input is optional.
  if (gnrrd_port->get(gHandle) && gHandle.get_rep()) {
    
    if (!ShaderProgramARB::shaders_supported()) {
      // TODO: Runtime check, change message to reflect that.
      warning("This machine does not support advanced volume rendering. The gradient nrrd will be ignored.");
      if( gnrrd_last_generation_ != -1 )
	update = true;
      gHandle = 0;
      gnrrd_last_generation_ = -1;

    } else {
      Nrrd* gm_nrrd = gHandle->nrrd_;

      if (gm_nrrd->dim != 3 && gm_nrrd->dim != 4) {
	error("Invalid dimension for input gradient magnitude nrrd.");
	return;
      }
      
      if( gm_nrrd->dim == 4 ) {
	nrrdAxisInfoGet_nva(gm_nrrd, nrrdAxisInfoSize, axis_size);
	if (axis_size[0] != 1) {
	  error("Invalid axis size for gradient magnitude nrrd.");
	  return;
	}
      }

      // The input nrrd type must be unsigned char.
      if (gui_uchar_.get() && gHandle->nrrd_->type != nrrdTypeUChar) {
	error("Gradient magnitude input nrrd type must be unsigned char.");
	return;
      }

      if( !gui_fixed_.get() ){
	// set gmin/gmax
	NrrdRange *range =
	  nrrdRangeNewSet(gHandle->nrrd_, nrrdBlind8BitRangeFalse);

	gui_gminval_.set(range->min);
	gui_gmaxval_.set(range->max);
      }
	
      // Check to see if the input gradient nrrd has changed.
      if( gnrrd_last_generation_ != gHandle->generation  ||
	  (gui_gminval_.get() != gminval_) || 
	  (gui_gmaxval_.get() != gmaxval_) ) {
	gnrrd_last_generation_ = gHandle->generation;
	  
	gminval_ = gui_gminval_.get();
	gmaxval_ = gui_gmaxval_.get();
	  
	update = true;
      }
    }
  } else {
    if( gnrrd_last_generation_ != -1 )
      update = true;

    gnrrd_last_generation_ = -1;
  }

  // Get a handle to the input mask port.
  NrrdIPort* mnrrd_port = (NrrdIPort *) get_iport("Mask Nrrd");

  // The mask nrrd input is optional.
  if (mnrrd_port->get(mHandle) && mHandle.get_rep()) {
    
    if (!ShaderProgramARB::shaders_supported()) {
      // TODO: Runtime check, change message to reflect that.
      warning("This machine does not support advanced volume rendering. The mask nrrd will be ignored.");
      if( mnrrd_last_generation_ != -1 )
	update = true;
      mHandle = 0;
      mnrrd_last_generation_ = -1;

    } else {
      Nrrd* mask_nrrd = mHandle->nrrd_;

      if (mask_nrrd->dim != 3 && mask_nrrd->dim != 4) {
	error("Invalid dimension for input mask nrrd.");
	return;
      }
      
      if( mask_nrrd->dim == 4 ) {
	nrrdAxisInfoGet_nva(mask_nrrd, nrrdAxisInfoSize, axis_size);
	if (axis_size[0] != 1) {
	  error("Invalid axis size for mask nrrd.");
	  return;
	}
      }

      // The input nrrd type must be unsigned char.
      if (gui_uchar_.get() && mHandle->nrrd_->type != nrrdTypeUChar) {
	error("Mask input nrrd type must be unsigned char.");
	return;
      }

      if( !gui_fixed_.get() ){
	// set mmin/mmax
	NrrdRange *range =
	  nrrdRangeNewSet(mHandle->nrrd_, nrrdBlind8BitRangeFalse);

	gui_mminval_.set(range->min);
	gui_mmaxval_.set(range->max);
      }
	
      // Check to see if the input gradient nrrd has changed.
      if( mnrrd_last_generation_ != mHandle->generation  ||
	  (gui_mminval_.get() != mminval_) || 
	  (gui_mmaxval_.get() != mmaxval_) ) {
	mnrrd_last_generation_ = mHandle->generation;
	  
	mminval_ = gui_mminval_.get();
	mmaxval_ = gui_mmaxval_.get();
	  
	update = true;
      }
    }
  } else {
    if( mnrrd_last_generation_ != -1 )
      update = true;

    mnrrd_last_generation_ = -1;
  }

  if( gui_uchar_.get() != is_uchar_ ) {
    is_uchar_ = gui_uchar_.get();

    update = true;
  }

  if( update ) {

    CompileInfoHandle ci =
      NrrdTextureBuilderAlgo::get_compile_info(vHandle->nrrd_->type,
					       gHandle.get_rep() ? 
					       gHandle->nrrd_->type :
					       vHandle->nrrd_->type);
    
    Handle<NrrdTextureBuilderAlgo> algo;
    if (!module_dynamic_compile(ci, algo)) return;
    
    algo->build(tHandle_,
		vHandle, vminval_, vmaxval_,
		gHandle, gminval_, gmaxval_,
		//mHandle, mminval_, mmaxval_,
		gui_card_mem_.get(),
		gui_uchar_.get());
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

} // namespace SCIRun
