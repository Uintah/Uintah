/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


/*
 *  InterpFieldToMatrix.cc:  Unfinished modules
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   March 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Dataflow/Modules/Fields/InterpFieldToMatrix.h>
#include <iostream>

namespace SCIRun {

class InterpFieldToMatrix : public Module
{
public:
  InterpFieldToMatrix(GuiContext* ctx);
  virtual ~InterpFieldToMatrix();
  virtual void execute();
};


DECLARE_MAKER(InterpFieldToMatrix)

InterpFieldToMatrix::InterpFieldToMatrix(GuiContext* ctx)
  : Module("InterpFieldToMatrix", ctx, Filter, "Fields", "SCIRun")
{
}



InterpFieldToMatrix::~InterpFieldToMatrix()
{
}

void
InterpFieldToMatrix::execute()
{
  FieldIPort *ifp = (FieldIPort *)get_iport("Interp");
  FieldHandle fieldhandle;
  if (!ifp) {
    error("Unable to initialize iport 'Interp'.");
    return;
  }
  if (!(ifp->get(fieldhandle) && fieldhandle.get_rep()))
  {
    return;
  }

  // Compute output matrix.
  CompileInfoHandle ci_field =
    InterpFieldToMatrixAlgoBase::
    get_compile_info(fieldhandle->get_type_description(),
		     fieldhandle->data_at_type_description());
  Handle<InterpFieldToMatrixAlgoBase> algo_field;
  if (!module_dynamic_compile(ci_field, algo_field)) return;
  MatrixHandle denseMatrixH, columnMatrixH;
  algo_field->execute(fieldhandle, denseMatrixH, columnMatrixH);
  
  MatrixOPort *dmp = (MatrixOPort *)get_oport("DenseWeights");
  if (!dmp) {
    error("Unable to initialize oport 'DenseWeights'.");
    return;
  }
  dmp->send(denseMatrixH);

  MatrixOPort *cmp = (MatrixOPort *)get_oport("ColumnMap");
  if (!cmp) {
    error("Unable to initialize oport 'ColumnMap'.");
    return;
  }
  cmp->send(columnMatrixH);
}

CompileInfoHandle
InterpFieldToMatrixAlgoBase::get_compile_info(const TypeDescription *field_td,
					  const TypeDescription *loc_td)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("InterpFieldToMatrixAlgo");
  static const string base_class_name("InterpFieldToMatrixAlgoBase");

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


