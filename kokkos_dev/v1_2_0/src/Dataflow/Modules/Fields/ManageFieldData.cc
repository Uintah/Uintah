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
 *  ManageFieldData: Store/retrieve values from an input matrix to/from 
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
#include <Dataflow/Modules/Fields/ManageFieldData.h>
#include <Core/GuiInterface/GuiVar.h>
#include <iostream>
#include <stdio.h>

namespace SCIRun {

class ManageFieldData : public Module
{
public:
  ManageFieldData(const string& id);
  virtual ~ManageFieldData();

  virtual void execute();
};


extern "C" Module* make_ManageFieldData(const string& id)
{
  return new ManageFieldData(id);
}

ManageFieldData::ManageFieldData(const string& id)
  : Module("ManageFieldData", id, Filter, "Fields", "SCIRun")
{
}



ManageFieldData::~ManageFieldData()
{
}



void
ManageFieldData::execute()
{
  // Get input field.
  FieldIPort *ifp = (FieldIPort *)get_iport("Input Field");
  FieldHandle ifieldhandle;
  Field *ifield;
  if (!ifp) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  if (!(ifp->get(ifieldhandle) && (ifield = ifieldhandle.get_rep())))
  {
    return;
  }

  int svt_flag = 0;
  if (ifieldhandle->query_scalar_interface())
  {
    svt_flag = 0;
  }
  else if (ifieldhandle->query_vector_interface())
  {
    svt_flag = 1;
  }
  else if (ifieldhandle->query_tensor_interface())
  {
    svt_flag = 2;
  }
  
  CompileInfo *ci_field =
    ManageFieldDataAlgoField::
    get_compile_info(ifieldhandle->get_type_description(),
		     ifieldhandle->data_at_type_description(),
		     svt_flag);
  DynamicAlgoHandle algo_handle_field;
  if (! DynamicLoader::scirun_loader().get(*ci_field, algo_handle_field))
  {
    error("Could not compile field algorithm.");
    return;
  }
  ManageFieldDataAlgoField *algo_field =
    dynamic_cast<ManageFieldDataAlgoField *>(algo_handle_field.get_rep());
  if (algo_field == 0)
  {
    error("Could not get field algorithm.");
    return;
  }

  MatrixOPort *omp = (MatrixOPort *)get_oport("Output Matrix");
  if (!omp) {
    postMessage("Unable to initialize "+name+"'s oport\n");
    return;
  }
  omp->send(algo_field->execute(ifieldhandle));


  MatrixIPort *imatrix_port = (MatrixIPort *)get_iport("Input Matrix");
  MatrixHandle imatrixhandle;
  if (!imatrix_port) {
    postMessage("Unable to initialize "+name+"'s iport\n");
    return;
  }
  if (!imatrix_port->get(imatrixhandle) && (!imatrixhandle.get_rep()))
  {
    remark("No input matrix connected.");
    return;
  }
  
  CompileInfo *ci_mesh =
    ManageFieldDataAlgoMesh::
    get_compile_info(ifieldhandle->mesh()->get_type_description(),
		     ifieldhandle->get_type_description(),
		     svt_flag);
  DynamicAlgoHandle algo_handle_mesh;
  if (! DynamicLoader::scirun_loader().get(*ci_mesh, algo_handle_mesh))
  {
    error("Could not compile mesh algorithm.");
    return;
  }
  ManageFieldDataAlgoMesh *algo_mesh =
    dynamic_cast<ManageFieldDataAlgoMesh *>(algo_handle_mesh.get_rep());
  if (algo_mesh == 0)
  {
    error("Could not get mesh algorithm.");
    return;
  }

  FieldOPort *ofp = (FieldOPort *)get_oport("Output Field");
  if (!ofp) {
    postMessage("Unable to initialize "+name+"'s oport\n");
    return;
  }
  ofp->send(algo_mesh->execute(ifieldhandle->mesh(), imatrixhandle));
}



CompileInfo *
ManageFieldDataAlgoField::get_compile_info(const TypeDescription *fsrc,
					   const TypeDescription *lsrc,
					   int svt_flag)
{
  // Use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string base_class_name("ManageFieldDataAlgoField");

  string extension;
  switch (svt_flag)
  {
  case 2:
    extension = "Tensor";
    break;

  case 1:
    extension = "Vector";
    break;

  default:
    extension = "Scalar";
    break;
  }

  CompileInfo *rval = 
    scinew CompileInfo(base_class_name + extension + "." +
		       to_filename(fsrc->get_name()) + "." +
		       to_filename(lsrc->get_name()) + ".",
                       base_class_name, 
                       base_class_name + extension, 
                       fsrc->get_name() + ", " + lsrc->get_name());

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  fsrc->fill_compile_info(rval);
  return rval;
}



CompileInfo *
ManageFieldDataAlgoMesh::get_compile_info(const TypeDescription *msrc,
					  const TypeDescription *fsrc,
					  int svt_flag)
{
  // Use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string base_class_name("ManageFieldDataAlgoMesh");

  string extension;
  string extension2;
  switch (svt_flag)
  {
  case 2:
    extension = "Tensor";
    extension2 = "Tensor";
    break;

  case 1:
    extension = "Vector";
    extension2 = "Vector";
    break;

  default:
    extension = "Scalar";
    extension2 = "double";
    break;
  }

  string::size_type loc = fsrc->get_name().find_first_of("<");
  string fout = fsrc->get_name().substr(0, loc) + "<" + extension2 + "> ";

  CompileInfo *rval = 
    scinew CompileInfo(base_class_name + extension + "." +
		       to_filename(msrc->get_name()) + "." +
		       to_filename(fout) + ".",
                       base_class_name, 
                       base_class_name + extension, 
                       msrc->get_name() + ", " + fout);

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  fsrc->fill_compile_info(rval);
  return rval;
}



} // End namespace SCIRun

