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
 *  ManageFieldMesh: Store/retrieve values from an input matrix to/from 
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

#include <Core/Persistent/Pstreams.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Modules/Fields/ManageFieldMesh.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Containers/Handle.h>
#include <iostream>
#include <stdio.h>

namespace SCIRun {

class ManageFieldMesh : public Module
{
public:
  ManageFieldMesh(GuiContext* ctx);
  virtual ~ManageFieldMesh();

  virtual void execute();
};


DECLARE_MAKER(ManageFieldMesh)
ManageFieldMesh::ManageFieldMesh(GuiContext* ctx)
  : Module("ManageFieldMesh", ctx, Filter, "Fields", "SCIRun")
{
}



ManageFieldMesh::~ManageFieldMesh()
{
}



void
ManageFieldMesh::execute()
{
  // Get input field.
  FieldIPort *ifp = (FieldIPort *)get_iport("Input Field");
  FieldHandle ifieldhandle;
  if (!ifp) {
    error("Unable to initialize iport 'Input Field'.");
    return;
  }
  if (!(ifp->get(ifieldhandle) && (ifieldhandle.get_rep())))
  {
    error( "No handle or representation." );
    return;
  }
  
  // Extract the output matrix.
  const TypeDescription *mtd = ifieldhandle->mesh()->get_type_description();
  CompileInfoHandle ci_extract =
    ManageFieldMeshAlgoExtract::get_compile_info(mtd);
  Handle<ManageFieldMeshAlgoExtract> algo_extract;
  if (!module_dynamic_compile(ci_extract, algo_extract))
  {
    return;
  }
  MatrixOPort *omp = (MatrixOPort *)get_oport("Output Matrix");
  if (!omp)
  {
    error("Unable to initialize oport 'Output Matrix'.");
  }
  else
  {
    omp->send(algo_extract->execute(ifieldhandle->mesh()));
  }

  // Compute output field.
  FieldHandle result_field;
  MatrixIPort *imatrix_port = (MatrixIPort *)get_iport("Input Matrix");
  MatrixHandle imatrixhandle;
  if (!imatrix_port)
  {
    error("Unable to initialize iport 'Input Matrix'.");
    return;
  }
  if (!(imatrix_port->get(imatrixhandle) && imatrixhandle.get_rep()))
  {
    remark("No input matrix connected, sending field as is.");
    result_field = ifieldhandle;
  }
  else
  {
    const TypeDescription *ftd = ifieldhandle->get_type_description();
    CompileInfoHandle ci_insert =
      ManageFieldMeshAlgoInsert::get_compile_info(ftd);
    Handle<ManageFieldMeshAlgoInsert> algo_insert;
    if (!DynamicCompilation::compile(ci_insert, algo_insert, true, this))
    {
      error("Could not compile insertion algorithm.");
      error("Input field probably not of editable type.");
      return;
    }

    result_field = algo_insert->execute(this, ifieldhandle, imatrixhandle);

    if (!result_field.get_rep())
    {
      return;
    }

    // Copy the properties.
    *((PropertyManager *)(result_field.get_rep())) =
      *((PropertyManager *)(ifieldhandle.get_rep()));
  }

  FieldOPort *ofp = (FieldOPort *)get_oport("Output Field");
  if (!ofp) {
    error("Unable to initialize oport 'Output Field'.");
    return;
  }

  ofp->send(result_field);
}



CompileInfoHandle
ManageFieldMeshAlgoExtract::get_compile_info(const TypeDescription *msrc)
{
  // Use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("ManageFieldMeshAlgoExtractT");
  static const string base_class_name("ManageFieldMeshAlgoExtract");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       msrc->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       msrc->get_name());

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  msrc->fill_compile_info(rval);
  return rval;
}


CompileInfoHandle
ManageFieldMeshAlgoInsert::get_compile_info(const TypeDescription *fsrc)
{
  // Use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("ManageFieldMeshAlgoInsertT");
  static const string base_class_name("ManageFieldMeshAlgoInsert");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       fsrc->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       fsrc->get_name());

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  fsrc->fill_compile_info(rval);
  return rval;
}



} // End namespace SCIRun

