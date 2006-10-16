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
 *  ClipByFunction.cc:  Clip out parts of a field.
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   March 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Core/Util/DynamicCompilation.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Dataflow/Network/Ports/NrrdPort.h>
#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Core/Thread/CrowdMonitor.h>
#include <Dataflow/Widgets/BoxWidget.h>
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/FieldInterface.h>
#include <Core/Algorithms/Fields/ClipByFunction.h>
#include <Core/Containers/StringUtil.h>
#include <Core/Containers/HashTable.h>
#include <iostream>
#include <stack>

namespace SCIRun {

class ClipByFunction : public Module
{
public:
  ClipByFunction(GuiContext* ctx);
  virtual ~ClipByFunction();

  virtual void execute();

private:
  GuiString gui_mode_;
  GuiString gui_function_;
  GuiDouble gui_uservar0_;
  GuiDouble gui_uservar1_;
  GuiDouble gui_uservar2_;
  GuiDouble gui_uservar3_;
  GuiDouble gui_uservar4_;
  GuiDouble gui_uservar5_;

  FieldHandle  field_output_handle_;
  MatrixHandle matrix_output_handle_;
  NrrdDataHandle nrrd_output_handle_;

  bool error_;
};


DECLARE_MAKER(ClipByFunction)

ClipByFunction::ClipByFunction(GuiContext* ctx)
  : Module("ClipByFunction", ctx, Filter, "FieldsCreate", "SCIRun"),
    gui_mode_(get_ctx()->subVar("clipmode"), "cell"),
    gui_function_(get_ctx()->subVar("clipfunction"), "x < 0"),
    gui_uservar0_(get_ctx()->subVar("u0"), 0.0),
    gui_uservar1_(get_ctx()->subVar("u1"), 0.0),
    gui_uservar2_(get_ctx()->subVar("u2"), 0.0),
    gui_uservar3_(get_ctx()->subVar("u3"), 0.0),
    gui_uservar4_(get_ctx()->subVar("u4"), 0.0),
    gui_uservar5_(get_ctx()->subVar("u5"), 0.0)
{
}


ClipByFunction::~ClipByFunction()
{
}



void
ClipByFunction::execute()
{
  // Get input field.
  FieldHandle field_input_handle;
  if( !get_input_handle( "Input", field_input_handle, true ) ) return;

  if (!field_input_handle->mesh()->is_editable()) {
    error("Not an editable mesh type.");
    error("(Try passing Field through an Unstructure module first).");
    return;
  }

  // Check to see if the input field has changed.
  if( inputs_changed_ ||
      gui_function_.changed( true ) ||
      gui_mode_.changed( true ) ||
      !matrix_output_handle_.get_rep() ||
      !field_output_handle_.get_rep() ) {

    update_state(Executing);

    string function = gui_function_.get();

    // remove trailing white-space from the function string
    while (function.size() && isspace(function[function.size()-1]))
      function.resize(function.size()-1);

    const TypeDescription *ftd = field_input_handle->get_type_description();
    Handle<ClipByFunctionAlgo> algo;
    int hoffset = 0;

    while (1) {
      CompileInfoHandle ci =
	ClipByFunctionAlgo::get_compile_info(ftd, function, hoffset);
      if (!DynamicCompilation::compile(ci, algo, false, this)){
	  error("Your function would not compile.");
	  get_gui()->eval(get_id() + " compile_error "+ci->filename_);
	  DynamicLoader::scirun_loader().cleanup_failed_compile(ci);
	  return;
	}
      if (algo->identify() == function)
	  break;

      hoffset++;
    }

    int gMode = 0;
    if (gui_mode_.get() == "cell")
      gMode = 0;
    else if (gui_mode_.get() == "onenode")
      gMode = 1;
    else if (gui_mode_.get() == "majoritynodes")
      gMode = 2;
    else if (gui_mode_.get() == "allnodes")
      gMode = -1;

    // User Variables.
    algo->u0 = gui_uservar0_.get();
    algo->u1 = gui_uservar1_.get();
    algo->u2 = gui_uservar2_.get();
    algo->u3 = gui_uservar3_.get();
    algo->u4 = gui_uservar4_.get();
    algo->u5 = gui_uservar5_.get();

    if (!(field_input_handle->basis_order() == 0 && gMode == 0 ||
          field_input_handle->basis_order() == 1 && gMode != 0) &&
        field_input_handle->mesh()->dimensionality() > 0)
    {
      warning("Basis doesn't match clip location, value will always be zero.");
    }

    field_output_handle_ =
      algo->execute(this, field_input_handle, gMode, matrix_output_handle_);

    if( matrix_output_handle_.get_rep() ) {
      SparseRowMatrix *matrix =
	dynamic_cast<SparseRowMatrix *>(matrix_output_handle_.get_rep());
      size_t dim[NRRD_DIM_MAX];
      dim[0] = matrix->ncols();    
      nrrd_output_handle_ = scinew NrrdData;
      Nrrd *nrrd = nrrd_output_handle_->nrrd_;
      nrrdAlloc_nva(nrrd, nrrdTypeUChar, 1, dim);
      unsigned char *mask = (unsigned char *)nrrd->data;
      memset(mask, 0, dim[0]*sizeof(unsigned char));
      int *rr = matrix->rows;
      int *cc = matrix->columns;
      double *data = matrix->a;
      for (int i = 0; i < matrix->nrows(); ++i) {
	if (rr[i+1] == rr[i]) continue; // No entires on this row
	int col = cc[rr[i]];
	if (data[rr[i]] > 0.0) {
	  mask[col] = 1;
	}
      }
    }
  }

  send_output_handle( "Clipped", field_output_handle_, true );
  send_output_handle( "Mapping", matrix_output_handle_, true );
  send_output_handle( "MaskVector", nrrd_output_handle_, true );
}

} // End namespace SCIRun

