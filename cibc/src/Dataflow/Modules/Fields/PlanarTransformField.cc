/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   
   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and associated documentation files (the "Software"),
   to deal in the Software without restriction, including without limitation
   the rights to use, copy, modify, merge, publish, distribute, sublicense,
   and/or sell copies of the Software, and to permit persons to whom the
   Software is furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
   OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
   THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
   FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
   DEALINGS IN THE SOFTWARE.
*/



/*
 *  PlanarTransformField.cc
 *
 *  Rotate and flip field to get it into "standard" view
 *
 *  Written by:
 *   Allen Sanderson
 *   Scientific Computing and Imaging Institute
 *   University of Utah
 *   April 2006
 *
 *  Copyright (C) 2006 SCI Inst
 */
#include <Dataflow/Network/Module.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Core/Geometry/Transform.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MiscMath.h>
#include <iostream>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#include <Dataflow/Modules/Fields/PlanarTransformField.h>

namespace SCIRun {


class PlanarTransformField : public Module
{
public:
  PlanarTransformField(GuiContext* ctx);
  virtual ~PlanarTransformField();

  virtual void execute();

protected:
  GuiInt gui_axis_;
  GuiInt gui_invert_;
  GuiInt gui_trans_x_;
  GuiInt gui_trans_y_;

  FieldHandle field_output_handle_;
};


DECLARE_MAKER(PlanarTransformField)

PlanarTransformField::PlanarTransformField(GuiContext* context)
  : Module("PlanarTransformField", context, Filter, "FieldsGeometry", "SCIRun"),
    gui_axis_(context->subVar("axis"), 2),
    gui_invert_(context->subVar("invert"), 0),
    gui_trans_x_(context->subVar("trans_x"), 0),
    gui_trans_y_(context->subVar("trans_y"), 0)
{
}


PlanarTransformField::~PlanarTransformField()
{
}


void
PlanarTransformField::execute()
{
  // Get the input field.
  FieldHandle field_input_handle;
  if( !get_input_handle( "Input Field", field_input_handle, true ) ) return;

  // Get a handle to the optional index matrix port.
  MatrixHandle matrix_input_handle;
  get_input_handle( "Index Matrix", matrix_input_handle, false );

  if( matrix_input_handle.get_rep() ) {
    //! Check to see what index has been selected and if it matches
    //! the gui index.a
    if( gui_trans_x_.get() != matrix_input_handle->get(0, 0) ||
	gui_trans_y_.get() != matrix_input_handle->get(1, 0) ) {

      gui_trans_x_.set( (int) matrix_input_handle->get(0, 0) );
      gui_trans_y_.set( (int) matrix_input_handle->get(1, 0) );

      inputs_changed_ = true;
    }
  }

  //! If no data or an input change recreate the field. I.e Only
  //! execute when neeed.
  if (inputs_changed_  ||
      
      !field_output_handle_.get_rep() ||
      
      gui_axis_.changed( true ) ||
      gui_invert_.changed( true ) ||
      gui_trans_x_.changed( true ) ||
      gui_trans_y_.changed( true ) )
  {
    const TypeDescription *ftd = field_input_handle->get_type_description();
    CompileInfoHandle ci = PlanarTransformFieldAlgo::get_compile_info(ftd);

    Handle<PlanarTransformFieldAlgo> algo;
    if (!module_dynamic_compile(ci, algo)) return;

    field_output_handle_ = algo->execute(field_input_handle,
					 gui_axis_.get(),
					 gui_invert_.get(),
					 gui_trans_x_.get(),
					 gui_trans_y_.get());
  }
    
  // Send the data downstream.
  send_output_handle( "Transformed Field",  field_output_handle_, true );
}


CompileInfoHandle
PlanarTransformFieldAlgo::get_compile_info(const TypeDescription *ftd)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("PlanarTransformFieldAlgoT");
  static const string base_class_name("PlanarTransformFieldAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       ftd->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       ftd->get_name() );

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  ftd->fill_compile_info(rval);
  return rval;
}

} // End namespace SCIRun

