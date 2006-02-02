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
 *  InsertField.cc:  Insert a field into a TetVolMesh
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   January 2006
 *
 *  Copyright (C) 2006 SCI Group
 */

#include <Core/Util/DynamicCompilation.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/FieldInterface.h>
#include <Dataflow/Modules/Fields/InsertField.h>
#include <Core/Containers/StringUtil.h>
#include <iostream>

namespace SCIRun {


class InsertField : public Module
{
  int tet_generation_;
  int insert_generation_;
  FieldHandle output_field_;

public:
  InsertField(GuiContext* ctx);
  virtual ~InsertField();

  virtual void execute();

};


DECLARE_MAKER(InsertField)

InsertField::InsertField(GuiContext* ctx)
  : Module("InsertField", ctx, Filter, "FieldsGeometry", "SCIRun"),
    tet_generation_(-1),
    insert_generation_(-1)
{
}


InsertField::~InsertField()
{
}


void
InsertField::execute()
{
  // Get input field 0.
  FieldIPort *ifp = (FieldIPort *)get_iport("Input Tetrahedral Volume");
  FieldHandle tet_field;
  if (!(ifp->get(tet_field) && tet_field.get_rep())) {
    error("Required input Tetrahedral Volume is empty.");
    return;
  }

  // Get input field 1.
  ifp = (FieldIPort *)get_iport("Insert Field");
  FieldHandle insert_field;
  if (!(ifp->get(insert_field) && insert_field.get_rep())) {
    error("Required input Insert Field is empty.");
    return;
  }

  bool update = false;

  // Check to see if the source field has changed.
  if( tet_generation_ != tet_field->generation ) {
    tet_generation_ = tet_field->generation;
    update = true;
  }

  // Check to see if the source field has changed.
  if( insert_generation_ != insert_field->generation ) {
    insert_generation_ = insert_field->generation;
    update = true;
  }

  if( !output_field_.get_rep() || update)
  {
    const TypeDescription *ftd0 = tet_field->get_type_description();
    const TypeDescription *ftd1 = insert_field->get_type_description();

    CompileInfoHandle ci =
      InsertFieldAlgo::get_compile_info(ftd0, ftd1);
    Handle<InsertFieldAlgo> algo;
    if (!DynamicCompilation::compile(ci, algo, this)) {
      error("Dynamic compilation failed.");
      return;
    }

    output_field_ = tet_field;
    tet_field = 0;
    output_field_.detach();
    output_field_->mesh_detach();
    if (insert_field->mesh()->dimensionality() == 1)
    {
      algo->execute_1(output_field_, insert_field);
    }
    else
    {
      algo->execute_0(output_field_, insert_field);
    }
  }

  if( output_field_.get_rep() )
  {
    FieldOPort *ofield_port = (FieldOPort *)get_oport("Combined Fields");
    ofield_port->send_and_dereference(output_field_, true);
  }
}


CompileInfoHandle
InsertFieldAlgo::get_compile_info(const TypeDescription *ftet,
                                  const TypeDescription *finsert)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("InsertFieldAlgoT");
  static const string base_class_name("InsertFieldAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       ftet->get_filename() + "." +
                       finsert->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       ftet->get_name() + ", " +
                       finsert->get_name());

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  ftet->fill_compile_info(rval);
  finsert->fill_compile_info(rval);
  return rval;
}


} // End namespace SCIRun

