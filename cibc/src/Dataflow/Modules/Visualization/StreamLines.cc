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

//    File   : StreamLines.cc
//    Author : Allen R. Sanderson
//    Date   : July 2006


#include <Dataflow/Modules/Visualization/StreamLines.h>

#include <Dataflow/Network/Ports/FieldPort.h>

namespace SCIRun {

class StreamLines : public Module {
public:
  StreamLines(GuiContext* ctx);

  virtual ~StreamLines();

  virtual void execute();

private:
  FieldHandle                   field_output_handle_;

  GuiDouble                     gui_step_size_;
  GuiDouble                     gui_tolerance_;
  GuiInt                        gui_max_steps_;
  GuiInt                        gui_direction_;
  GuiInt                        gui_value_;
  GuiInt                        gui_remove_colinear_pts_;
  GuiInt                        gui_method_;
  GuiInt                        gui_nthreads_;

  bool execute_error_;
};

DECLARE_MAKER(StreamLines)

StreamLines::StreamLines(GuiContext* ctx) : 
  Module("StreamLines", ctx, Source, "Visualization", "SCIRun"),
  gui_step_size_(get_ctx()->subVar("stepsize"), 0.01),
  gui_tolerance_(get_ctx()->subVar("tolerance"), 0.0001),
  gui_max_steps_(get_ctx()->subVar("maxsteps"), 2000),
  gui_direction_(get_ctx()->subVar("direction"), 1),
  gui_value_(get_ctx()->subVar("value"), 1),
  gui_remove_colinear_pts_(get_ctx()->subVar("remove-colinear-pts"), 1),
  gui_method_(get_ctx()->subVar("method"), 4),
  gui_nthreads_(get_ctx()->subVar("nthreads"), 1),
  execute_error_(0)
{
}

StreamLines::~StreamLines()
{
}

void
StreamLines::execute()
{
  FieldHandle vfHandle;
  if( !get_input_handle( "Vector Field", vfHandle, true ) )
    return;

  //! Check that the input field input is a vector field.
  VectorFieldInterfaceHandle vfi =
    vfHandle.get_rep()->query_vector_interface(this);
  if (!vfi.get_rep()) {
    error("FlowField is not a Vector field.");
    return;
  }

  //! Works for surfaces and volume data only.
  if (vfHandle.get_rep()->mesh()->dimensionality() == 1) {
    error("The StreamLines module does not works on 1D fields.");
    return;
  }

  FieldHandle spHandle;
  if( !get_input_handle( "Seed Points", spHandle, true ) )
    return;

  if( !field_output_handle_.get_rep() ||
      
      inputs_changed_ ||

      gui_tolerance_.changed( true )           ||
      gui_step_size_.changed( true )           ||
      gui_max_steps_.changed( true )           ||
      gui_direction_.changed( true )           ||
      gui_value_.changed( true )               ||
      gui_remove_colinear_pts_.changed( true ) ||
      gui_method_.changed( true )              ||
      gui_nthreads_.changed( true )            ||

      execute_error_ ) {
  
    execute_error_ = false;

    update_state(Executing);

    Field *vField = vfHandle.get_rep();
    Field *sField = spHandle.get_rep();

    const TypeDescription *sftd = sField->get_type_description();
    
    const TypeDescription *sfdtd = 
      (*sField->get_type_description(Field::FDATA_TD_E)->get_sub_type())[0];
    const TypeDescription *sltd = sField->order_type_description();
    string dsttype = "double";
    if (gui_value_.get() == 0) dsttype = sfdtd->get_name();

    vField->mesh()->synchronize(Mesh::LOCATE_E);
    vField->mesh()->synchronize(Mesh::EDGES_E);

    if (gui_method_.get() == 5 ) {

      if( vfHandle->basis_order() != 0) {
	error("The Cell Walk method only works for cell centered FlowFields.");
	execute_error_ = true;
	return;
      }

      const string dftn =
        "GenericField<CurveMesh<CrvLinearLgn<Point> >, CrvLinearLgn<" +
        dsttype + ">, vector<" + dsttype + "> > ";

      const TypeDescription *vtd = vfHandle->get_type_description();
      CompileInfoHandle aci =
	StreamLinesAccAlgo::get_compile_info(sftd, sltd, vtd,
                                             dftn, gui_value_.get());
      Handle<StreamLinesAccAlgo> accalgo;
      if (!module_dynamic_compile(aci, accalgo)) return;
      
      field_output_handle_ =
	accalgo->execute(this, sField, vfHandle,
			 gui_max_steps_.get(),
			 gui_direction_.get(),
			 gui_value_.get(),
			 gui_remove_colinear_pts_.get());
    } else {
      CompileInfoHandle ci =
	StreamLinesAlgo::get_compile_info(sftd, dsttype, sltd,
					  gui_value_.get());
      Handle<StreamLinesAlgo> algo;
      if (!module_dynamic_compile(ci, algo)) return;
      
      field_output_handle_ =
	algo->execute(this, sField, vfi,
		      gui_tolerance_.get(),
		      gui_step_size_.get(),
		      gui_max_steps_.get(),
		      gui_direction_.get(),
		      gui_value_.get(),
		      gui_remove_colinear_pts_.get(),
		      gui_method_.get(),
		      gui_nthreads_.get());
    }
  }
   
  send_output_handle( "Streamlines", field_output_handle_, true );
}


vector<Point>::iterator
StreamLinesAlgo::CleanupPoints(vector<Point> &input, double e2)
{
  // Removes colinear points from the list of points.
  unsigned int i, j = 0;

  for (i=1; i < input.size()-1; i++)
  {
    const Vector v0 = input[i] - input[j];
    const Vector v1 = input[i] - input[i+1];

    if (Cross(v0, v1).length2() > e2 && Dot(v0, v1) < 0.0)
    {
      j++;
      if (i != j) { input[j] = input[i]; }
    }
  }

  if (input.size() > 1)
  {
    j++;
    input[j] = input[input.size()-1];
  }

  return input.begin() + j + 1;
}


CompileInfoHandle
StreamLinesAlgo::get_compile_info(const TypeDescription *fsrc,
				  const string &dsrc,
				  const TypeDescription *sloc,
				  int value)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("StreamLinesAlgoT");
  static const string base_class_name("StreamLinesAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + (value?"M":"F") + "." +
		       fsrc->get_filename() + "." +
		       sloc->get_filename() + ".",
                       base_class_name, 
                       template_class_name + (value?"M":"F"), 
		       fsrc->get_name() + ", " +
		       dsrc + ", " +
		       sloc->get_name());

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  rval->add_basis_include("../src/Core/Basis/CrvLinearLgn.h");
  rval->add_mesh_include("../src/Core/Datatypes/CurveMesh.h");
  fsrc->fill_compile_info(rval);
  return rval;
}


CompileInfoHandle
StreamLinesAccAlgo::get_compile_info(const TypeDescription *fsrc,
				     const TypeDescription *sloc,
				     const TypeDescription *vfld,
                                     const string &fdst,
				     int value)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("StreamLinesAccAlgoT");
  static const string base_class_name("StreamLinesAccAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + (value?"M":"F") + "." +
		       fsrc->get_filename() + "." +
		       sloc->get_filename() + "." +
		       vfld->get_filename() + ".",
                       base_class_name, 
                       template_class_name + (value?"M":"F"),
		       fsrc->get_name() + ", " +
		       sloc->get_name() + ", " +
		       vfld->get_name() + ", " +
                       fdst);

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  rval->add_basis_include("../src/Core/Basis/CrvLinearLgn.h");
  rval->add_mesh_include("../src/Core/Datatypes/CurveMesh.h");
  fsrc->fill_compile_info(rval);
  vfld->fill_compile_info(rval);
  return rval;
}


} // End namespace SCIRun
