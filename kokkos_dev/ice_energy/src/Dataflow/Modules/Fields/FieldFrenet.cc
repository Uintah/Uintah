/*
   For more information, please see: http://software.sci.utah.edu

   The MIT License

   Copyright (c) 2004 Scientific Computing and Imaging Institute,
   University of Utah.

   License for the specific language governing rights and limitations under
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
 *  FieldFrenet.cc:
 *
 *  Written by:
 *   Allen Sanderson
 *   School of Computing
 *   University of Utah
 *   April 2005
 *
 *  Copyright (C) 2005 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/GuiInterface/GuiVar.h>

#include <Dataflow/Modules/Fields/FieldFrenet.h>

namespace SCIRun {

class FieldFrenet : public Module {
public:
  FieldFrenet(GuiContext *context);

  virtual ~FieldFrenet();

  virtual void execute();

private:
  GuiInt Direction_;
  GuiInt Axis_;
  GuiInt Dims_;

  int direction_;
  int axis_;

  FieldHandle fHandle_;

  int fGeneration_;
};


DECLARE_MAKER(FieldFrenet)


FieldFrenet::FieldFrenet(GuiContext *context)
  : Module("FieldFrenet", context, Filter, "FieldsOther", "SCIRun"),
    
    Direction_(context->subVar("direction")),
    Axis_(context->subVar("axis")),
    Dims_(context->subVar("dims")),

    direction_(0),
    axis_(2),

    fGeneration_(-1)
{
}

FieldFrenet::~FieldFrenet(){
}

void FieldFrenet::execute(){
  update_state(NeedData);

  FieldHandle fHandle;

  // Get a handle to the input field port.
  FieldIPort* ifield_port = (FieldIPort *) get_iport("Input Field");

  // The field input is required.
  if (!ifield_port->get(fHandle) || !(fHandle.get_rep()) ||
      !(fHandle->mesh().get_rep()))
  {
    error( "No handle or representation" );
    return;
  }

  const TypeDescription *ftd = fHandle->get_type_description();

  // Get the dimensions of the mesh.
  if( ftd->get_name().find("StructHexVolField"  ) != 0 &&
      ftd->get_name().find("StructQuadSurfField") != 0 &&
      ftd->get_name().find("StructCurveField"   ) != 0 ) {

    error( fHandle->get_type_description(0)->get_name() );
    error( "Only availible for structured data." );
    return;
  }

  if( fHandle->basis_order() != 1 ) {
    error( fHandle->get_type_description(0)->get_name() );
    error( "Currently only availible for node data." );
    return;
  }

  vector<unsigned int> dim;
  fHandle.get_rep()->mesh()->get_dim( dim );
  unsigned int dims = dim.size();

  // Check to see if the dimensions have changed.
  if( dims != (unsigned int) Dims_.get() ) {
    ostringstream str;
    str << id << " set_size " << dims;
    gui->execute(str.str().c_str());
  }

  // If no data or a changed input field or axis recreate the mesh.
  if( !fHandle_.get_rep() ||
      fGeneration_ != fHandle->generation ||
      direction_ != Direction_.get() ||
      axis_ != Axis_.get() ) {

    fGeneration_ = fHandle->generation;
    direction_ = Direction_.get();
    axis_ = Axis_.get();

    const TypeDescription *btd = fHandle->get_type_description(0);

    CompileInfoHandle ci = FieldFrenetAlgo::get_compile_info(ftd,btd, dims);
    Handle<FieldFrenetAlgo> algo;

    if (!module_dynamic_compile(ci, algo)) return;
  
    fHandle_ = algo->execute(fHandle, direction_, axis_ );
  }

  // Send the data downstream
  if( fHandle_.get_rep() )
  {
    FieldOPort *ofield_port = (FieldOPort *) get_oport("Output Field");
    ofield_port->send( fHandle_ );
  }
}

CompileInfoHandle
FieldFrenetAlgo::get_compile_info(const TypeDescription *ftd,
				   const TypeDescription *btd,
				   const unsigned int dim)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("FieldFrenetAlgoT_");
  static const string base_class_name("FieldFrenetAlgo");

  char dimstr[6];

  sprintf( dimstr, "%d", dim );

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       ftd->get_filename() + ".",
                       base_class_name, 
                       template_class_name + dimstr + "D", 
                       ftd->get_name() + ", " +
		       btd->get_name() + "<Vector> " );

  // Add in the include path to compile this obj
  rval->add_include(include_path);

  // Structured meshs have a set_point method which is needed. However, it is not
  // defined for gridded meshes. As such, the include file defined below contains a
  // compiler flag so that when needed in FieldSlicer.h it is compiled.
  if( ftd->get_name().find("StructHexVolField"  ) == 0 ||
      ftd->get_name().find("StructQuadSurfField") == 0 ||
      ftd->get_name().find("StructCurveField"   ) == 0 ) {

    string header_path(include_path);  // Get the right path 

    // Insert the Dynamic header file name.
    header_path.insert( header_path.find_last_of("."), "Dynamic" );

    rval->add_include(header_path);
  }

  ftd->fill_compile_info(rval);
  return rval;
}

} // End namespace SCIRun
