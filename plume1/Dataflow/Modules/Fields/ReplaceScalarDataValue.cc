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
 *  ReplaceScalarDataValue: Unary field data operations
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   June 2002
 *
 *  Copyright (C) 2002 SCI Group
 */

#include <Dataflow/Ports/FieldPort.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Dataflow/Modules/Fields/ReplaceScalarDataValue.h>
#include <Core/Containers/Handle.h>
#include <iostream>
#include <sstream>

namespace SCIRun {

class ReplaceScalarDataValue : public Module {
  GuiDouble oldvalue_;
  GuiDouble newvalue_;
public:
  ReplaceScalarDataValue(GuiContext* ctx);
  virtual ~ReplaceScalarDataValue();
  virtual void execute();
};


DECLARE_MAKER(ReplaceScalarDataValue)


  ReplaceScalarDataValue::ReplaceScalarDataValue(GuiContext* ctx)
    : Module("ReplaceScalarDataValue", ctx, Filter,"FieldsData", "SCIRun"),
      oldvalue_(ctx->subVar("oldvalue")), newvalue_(ctx->subVar("newvalue"))
{
}


ReplaceScalarDataValue::~ReplaceScalarDataValue()
{
}


void
ReplaceScalarDataValue::execute()
{
  warning("This module is deprecated.  Use TransformData instead.");

  // Get input field.
  FieldIPort *ifp = (FieldIPort *)get_iport("Input Field");
  FieldHandle ifieldhandle;
  if (!(ifp->get(ifieldhandle) && ifieldhandle.get_rep()))
  {
    error("Input field is empty.");
    return;
  }
  
  if (ifieldhandle->query_scalar_interface(this).get_rep() == 0)
  {
    error("This module only works on scalar fields.");
    return;
  }

  double oldvalue = oldvalue_.get();
  double newvalue = newvalue_.get();

  const TypeDescription *ftd = ifieldhandle->get_type_description();
  const TypeDescription *ltd = ifieldhandle->order_type_description();
  CompileInfoHandle ci =
    ReplaceScalarDataValueAlgo::get_compile_info(ftd, ltd);
  Handle<ReplaceScalarDataValueAlgo> algo;
  if (!module_dynamic_compile(ci, algo)) return;

  FieldHandle ofieldhandle(algo->execute(ifieldhandle, oldvalue, newvalue));

  FieldOPort *ofield_port = (FieldOPort *)get_oport("Output Field");
  ofield_port->send(ofieldhandle);
}


CompileInfoHandle
ReplaceScalarDataValueAlgo::get_compile_info(const TypeDescription *field_td,
					 const TypeDescription *loc_td)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("ReplaceScalarDataValueAlgoT");
  static const string base_class_name("ReplaceScalarDataValueAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       field_td->get_filename() + "." +
		       loc_td->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       field_td->get_name() + ", " + loc_td->get_name());

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  field_td->fill_compile_info(rval);
  return rval;
}


} // End namespace SCIRun
