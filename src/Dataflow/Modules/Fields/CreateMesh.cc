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
 *  CreateMesh: Store/retrieve values from an input matrix to/from 
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
#include <Dataflow/Modules/Fields/CreateMesh.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Containers/Handle.h>
#include <iostream>
#include <stdio.h>

namespace SCIRun {

class CreateMesh : public Module
{
private:
  GuiString gui_fieldname_;
  GuiString gui_meshname_;
  GuiString gui_fieldbasetype_;
  GuiString gui_datatype_;

public:
  CreateMesh(GuiContext* ctx);
  virtual ~CreateMesh();

  virtual void execute();
};


DECLARE_MAKER(CreateMesh)
CreateMesh::CreateMesh(GuiContext* ctx)
  : Module("CreateMesh", ctx, Filter, "FieldsCreate", "SCIRun"),
    gui_fieldname_(ctx->subVar("fieldname")),
    gui_meshname_(ctx->subVar("meshname")),
    gui_fieldbasetype_(ctx->subVar("fieldbasetype")),
    gui_datatype_(ctx->subVar("datatype"))
{
}



CreateMesh::~CreateMesh()
{
}



void
CreateMesh::execute()
{
  MatrixIPort *elements_port = (MatrixIPort *)get_iport("Mesh Elements");
  MatrixHandle elementshandle;
  if (!elements_port)
  {
    error("Unable to initialize iport 'Mesh Elements'.");
    return;
  }
  if (!(elements_port->get(elementshandle) && elementshandle.get_rep()))
  {
    error("No input elements connected, unable to build mesh.");
    return;
  }

  MatrixIPort *positions_port = (MatrixIPort *)get_iport("Mesh Positions");
  MatrixHandle positionshandle;
  if (!positions_port)
  {
    error("Unable to initialize iport 'Mesh Positions'.");
    return;
  }
  if (!(positions_port->get(positionshandle) && positionshandle.get_rep()))
  {
    remark("No positions matrix connected, using zeros'.");
  }

  MatrixIPort *normals_port = (MatrixIPort *)get_iport("Mesh Normals");
  MatrixHandle normalshandle;
  if (!normals_port)
  {
    error("Unable to initialize iport 'Mesh Normals'.");
    return;
  }
  if (!(normals_port->get(normalshandle) && normalshandle.get_rep()))
  {
    remark("No input normals connected, not used.");
  }

  CompileInfoHandle ci =
      CreateMeshAlgo::get_compile_info(gui_fieldbasetype_.get(),
				       gui_datatype_.get());
  Handle<CreateMeshAlgo> algo;
  if (!DynamicCompilation::compile(ci, algo, false, this))
  {
    error("Could not compile algorithm.");
    return;
  }

  FieldHandle result_field =
    algo->execute(this, elementshandle, positionshandle,
		  normalshandle, Field::NODE);

  if (!result_field.get_rep())
  {
    return;
  }

  FieldOPort *ofp = (FieldOPort *)get_oport("Output Field");
  if (!ofp) {
    error("Unable to initialize oport 'Output Field'.");
    return;
  }

  ofp->send(result_field);
}



CompileInfoHandle
CreateMeshAlgo::get_compile_info(const string &basename,
				 const string &datatype)
{
  // Use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("CreateMeshAlgoT");
  static const string base_class_name("CreateMeshAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       to_filename(basename + datatype) + ".",
                       base_class_name, 
                       template_class_name, 
                       basename + "<" + datatype + "> ");

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  rval->add_namespace("SCIRun");

  return rval;
}



} // End namespace SCIRun

