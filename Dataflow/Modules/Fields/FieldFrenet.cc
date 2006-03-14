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
    
    Direction_(context->subVar("direction"), 0),
    Axis_(context->subVar("axis"), 2),
    Dims_(context->subVar("dims"), 3)
{
}


FieldFrenet::~FieldFrenet()
{
}


void
FieldFrenet::execute()
{
  update_state(NeedData);
  reset_vars();

  bool needToExecute = false;

  FieldHandle  fHandle;

  if( !getIHandle( "Input Field",  fHandle,  needToExecute, true  ) ) return;

  if( fHandle->mesh()->topology_geometry() ==
      (Mesh::STRUCTURED | Mesh::IRREGULAR) ) {

    error( fHandle->get_type_description(Field::FIELD_NAME_ONLY_E)->get_name() );
    error( "Only availible for topologically structured irregular data." );
    return;
  }

  if( fHandle->basis_order() != 1 ) {
    error( fHandle->get_type_description(Field::FIELD_NAME_ONLY_E)->get_name() );
    error( "Currently only availible for node data." );
    return;
  }

  // Get the dimensions of the mesh.
  vector<unsigned int> dims;
  fHandle.get_rep()->mesh()->get_dim( dims );

  Dims_.set( dims.size(), GuiVar::SET_GUI_ONLY );

  // Check to see if the dimensions have changed.
  if(  Dims_.changed( true ) ) {
    ostringstream str;
    str << id << " set_size ";
    gui->execute(str.str().c_str());
  }

  // If no data or a changed input field or axis recreate the mesh.
  if( !fHandle_.get_rep() ||
      Direction_.changed( true ) ||
      Axis_.changed( true ) || 
      needToExecute ) {

    const TypeDescription *ftd = fHandle->get_type_description();
    const TypeDescription *btd =
      fHandle->get_type_description(Field::FIELD_NAME_ONLY_E);

    CompileInfoHandle ci =
      FieldFrenetAlgo::get_compile_info(ftd, btd, dims.size());

    Handle<FieldFrenetAlgo> algo;

    if (!module_dynamic_compile(ci, algo)) return;
  
    fHandle_ = algo->execute(fHandle, direction_, axis_ );
  }

  // Send the data downstream
  if( fHandle_.get_rep() )
  {
    FieldOPort *ofield_port = (FieldOPort *) get_oport("Output Field");
    ofield_port->send_and_dereference( fHandle_, true );
  }
}


CompileInfoHandle
FieldFrenetAlgo::get_compile_info(const TypeDescription *ftd,
                                  const TypeDescription *btd,
                                  const unsigned int dims)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("FieldFrenetAlgoT_");
  static const string base_class_name("FieldFrenetAlgo");

  char dimstr[6];

  sprintf( dimstr, "%d", dims );

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       ftd->get_filename() + ".",
                       base_class_name, 
                       template_class_name + dimstr + "D", 
                       ftd->get_name() + ", " +
		       btd->get_name() + "<Vector> " );

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  ftd->fill_compile_info(rval);
  return rval;
}


} // End namespace SCIRun
