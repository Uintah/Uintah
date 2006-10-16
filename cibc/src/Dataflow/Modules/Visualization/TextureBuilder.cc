//
//  For more information, please see: http://software.sci.utah.edu
//
//  The MIT License
//
//  Copyright (c) 2004 Scientific Computing and Imaging Institute,
//  University of Utah.
//
//  
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


#include <Core/Basis/Constant.h>
#include <Core/Basis/HexTrilinearLgn.h>
#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Datatypes/MultiLevelField.h>
#include <Core/Containers/FData.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Algorithms/Visualization/TextureBuilderAlgo.h>
#include <Core/Geom/ShaderProgramARB.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Volume/VideoCardInfo.h>

#include <Dataflow/Network/Ports/TexturePort.h>
#include <Dataflow/Network/Ports/FieldPort.h>

#include <sci_defs/ogl_defs.h>


// Make sure to include this last
#include <Dataflow/Modules/Visualization/TextureBuilder.h>

namespace SCIRun {

typedef LatVolMesh<HexTrilinearLgn<Point> > LVMesh;



DECLARE_MAKER(TextureBuilder)

TextureBuilder::TextureBuilder(GuiContext* ctx, const std::string& name,
                 SchedClass sc,  const string& cat, const string& pack)
  : Module(name, ctx, sc, cat, pack),
    tHandle_(new Texture),
    gui_vminval_(get_ctx()->subVar("vmin"), 0),
    gui_vmaxval_(get_ctx()->subVar("vmax"), 1),
    gui_gminval_(get_ctx()->subVar("gmin"), 0),
    gui_gmaxval_(get_ctx()->subVar("gmax"), 1),
    gui_fixed_(get_ctx()->subVar("is_fixed"), 0),
    gui_card_mem_(get_ctx()->subVar("card_mem"), 16),
    gui_card_mem_auto_(get_ctx()->subVar("card_mem_auto"), 1),
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

  if (!get_input_handle("Scalar Field", vHandle)) return;

  string mesh_name = vHandle->get_type_description(Field::MESH_TD_E)->get_name();
  if( mesh_name.find("LatVolMesh", 0) == string::npos &&
      vHandle->get_type_description(Field::FIELD_NAME_ONLY_E)->get_name() != "MultiLevelField" &&
      vHandle->get_type_description(Field::FIELD_NAME_ONLY_E)->get_name() != "ITKLatVolField" ) {
      
    error( "Only availible for regular topology with uniformly gridded data." );
    return;
  } else {
    // We are using something that has regular topology, 
    // but it must currently be 3 dimensional
    LVMesh *mesh = (LVMesh * )vHandle->mesh().get_rep();
    int nx = mesh->get_ni();
    int ny = mesh->get_nj();
    int nz = mesh->get_nk();
    if(vHandle->basis_order() == 0) {
      --nx; --ny; --nz;
    }
    if (nx <= 1 || ny <=1 || nz <=1){
      error( "TextureBuilder requires 3D data. The received data is not 3D.");
      return;
    }
  }


  // The input field must contain scalar data.
  ScalarFieldInterfaceHandle sfi = vHandle->query_scalar_interface(this);
  if (!sfi.get_rep()) {
    error("Input scalar field does not contain scalar data.");
    return;
  }

  if( !gui_fixed_.get() ){

    double vmin = DBL_MAX, vmax = -DBL_MAX;

    if( vHandle->get_type_description(Field::FIELD_NAME_ONLY_E)->
        get_name().find("MultiLevelField") != string::npos ){
      // Warning::Temporary Hack!
      // In order to get the colors mapped correctly we need the min and max
      // values for every level of a Multi-level field.  This
      // temporary solution needs to be flushed out either in the sfi->minmax
      // algorithm or someplace else. We don't want to have to handle every
      // posible scalar type here.
      if( vHandle->get_type_description(Field::FDATA_TD_E)->
          get_name().find("double") != string::npos ) {
        typedef HexTrilinearLgn<double>          linearBasis;
        typedef ConstantBasis<double>            constantBasis;
        if(vHandle->get_type_description(Field::BASIS_TD_E)->
           get_name().find("Constant") != string::npos ) {
          typedef MultiLevelField<LVMesh, 
            constantBasis, 
            FData3d<double, LVMesh> > MRLVF_CB;
          typedef GenericField<LVMesh, 
            constantBasis, 
            FData3d<double, LVMesh> > LVF_CB;
          
          MRLVF_CB* vmrfield = (MRLVF_CB*) vHandle.get_rep();

          for(int i = 0 ; i < vmrfield->nlevels(); i++ ){
            const MultiLevelFieldLevel<LVF_CB>* lev = vmrfield->level( i );
            for(unsigned int j = 0; j < lev->patches.size(); j++ ){
              // Each patch in a level corresponds to a LatVolField.
              // Grab the field.
              MRLVF_CB::field_type* vmr = lev->patches[j].get_rep(); 
              // Now, get the min_max for the scalar field.
              ScalarFieldInterfaceHandle sub_sfi =
                vmr->query_scalar_interface(this);

              // set vmin/vmax
              double vmintmp, vmaxtmp;
              sub_sfi->compute_min_max(vmintmp, vmaxtmp);

              if( vmin > vmintmp ) vmin = vmintmp;
              if( vmax < vmaxtmp ) vmax = vmaxtmp;

            }
          }
        } else if(vHandle->get_type_description(Field::BASIS_TD_E)->
                  get_name().find("Linear") != string::npos ) {
          typedef MultiLevelField<LVMesh, 
                                linearBasis, 
                                FData3d<double, LVMesh> > MRLVF_LB;
          typedef GenericField<LVMesh, 
                               linearBasis, 
                               FData3d<double, LVMesh> > LVF_LB;
          
          MRLVF_LB* vmrfield =   (MRLVF_LB*) vHandle.get_rep();

          for(int i = 0 ; i < vmrfield->nlevels(); i++ ){
            const MultiLevelFieldLevel<LVF_LB>* lev = vmrfield->level( i );
            for(unsigned int j = 0; j < lev->patches.size(); j++ ){
              // Each patch in a level corresponds to a LatVolField.
              // Grab the field.
              MRLVF_LB::field_type* vmr = lev->patches[j].get_rep(); 
              // Now, get the min_max for the scalar field.
              ScalarFieldInterfaceHandle sub_sfi =
                vmr->query_scalar_interface(this);

              // set vmin/vmax
              double vmintmp, vmaxtmp;
              sub_sfi->compute_min_max(vmintmp, vmaxtmp);

              if( vmin > vmintmp ) vmin = vmintmp;
              if( vmax < vmaxtmp ) vmax = vmaxtmp;

            }
          }
        } else {
          error( "Input scalar field Basis function is not Constant or Linear.");
        }
      } else {
        error("Input scalar field, MultiLevelField does not contain double data.");
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

  // The gradient field input is optional.
  if (get_input_handle("Gradient Field", gHandle, false))
  {
    if (!ShaderProgramARB::shaders_supported()) {
      // TODO: Runtime check, change message to reflect that.
      warning("This machine does not support advanced volume rendering. The gradient field will be ignored.");
      if( gfield_last_generation_ != -1 )
	update = true;
      gHandle = 0;
      gfield_last_generation_ = -1;

    } else {
      string mesh_name = gHandle->get_type_description(Field::MESH_TD_E)->get_name();
      if( mesh_name.find("LatVolMesh", 0) == string::npos &&
 	  gHandle->get_type_description(Field::FIELD_NAME_ONLY_E)->
          get_name().find("MultiLevelField") != string::npos &&
	  gHandle->get_type_description(Field::FIELD_NAME_ONLY_E)->
          get_name().find("ITKLatVolField") != string::npos) {
    
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
	vfi->compute_length_min_max(gmin, gmax);

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
				   gHandle->get_type_description(Field::FIELD_NAME_ONLY_E) :
				   vHandle->get_type_description(Field::FIELD_NAME_ONLY_E));
    const TypeDescription* gmtd = (gHandle.get_rep() ?
				   gHandle->get_type_description(Field::MESH_TD_E) :
				   vHandle->get_type_description(Field::MESH_TD_E));
    const TypeDescription* gbtd = (gHandle.get_rep() ?
				   gHandle->get_type_description(Field::BASIS_TD_E) :
				   vHandle->get_type_description(Field::BASIS_TD_E));
    const TypeDescription* gdtd = (gHandle.get_rep() ?
				   gHandle->get_type_description(Field::FDATA_TD_E) :
				   vHandle->get_type_description(Field::FDATA_TD_E));

    CompileInfoHandle ci = TextureBuilderAlgo::get_compile_info(vftd, gftd,
								gmtd, gbtd,
								gdtd);

    Handle<TextureBuilderAlgo> algo;
    if (!module_dynamic_compile(ci, algo)) return;

    algo->build(tHandle_,
		vHandle, vminval_, vmaxval_,
		gHandle, gminval_, gmaxval_,
		gui_card_mem_.get());
  }

  send_output_handle("Texture", tHandle_, true);
}

} // end namespace SCIRun
