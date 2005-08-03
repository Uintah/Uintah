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
 *  ScaleFieldData: Store/retrieve values from an input matrix to/from 
 *            the data of a field
 *
 *  Written by:
 *   Michael Callahan
 *   Department of Computer Science
 *   University of Utah
 *   February 2001
 *
 *  Copyright (C) 2001 SCI Institute
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Modules/Fields/ScaleFieldData.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Containers/Handle.h>

#include <iostream>
#include <stdio.h>

namespace SCIRun {

class ScaleFieldData : public Module
{
public:
  ScaleFieldData(GuiContext* ctx);
  virtual ~ScaleFieldData();
  virtual void execute();
};


DECLARE_MAKER(ScaleFieldData)
ScaleFieldData::ScaleFieldData(GuiContext* ctx)
  : Module("ScaleFieldData", ctx, Filter, "FieldsData", "SCIRun")
{
}



ScaleFieldData::~ScaleFieldData()
{
}



void
ScaleFieldData::execute()
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
  if (ifieldhandle->basis_order() == -1)
  {
    error("This module only supports fields containing data.");
    return;
  }
  MatrixIPort *imatrix_port = (MatrixIPort *)get_iport("Input Matrix");
  MatrixHandle imatrix;
  if (!imatrix_port->get(imatrix))
  {
    return;
  }

  const TypeDescription *ftd = ifieldhandle->get_type_description();
  const TypeDescription *ltd = ifieldhandle->order_type_description();
  CompileInfoHandle ci = ScaleFieldDataAlgo::get_compile_info(ftd, ltd);
  Handle<ScaleFieldDataAlgo> algo;
  if (!module_dynamic_compile(ci, algo)) return;

  FieldHandle ofieldhandle(algo->execute(ifieldhandle, imatrix));

  FieldOPort *ofield_port = (FieldOPort *)get_oport("Output Field");
  ofield_port->send(ofieldhandle);
}



CompileInfoHandle
ScaleFieldDataAlgo::get_compile_info(const TypeDescription *field_td,
				     const TypeDescription *loc_td)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("ScaleFieldDataAlgoT");
  static const string base_class_name("ScaleFieldDataAlgo");

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
