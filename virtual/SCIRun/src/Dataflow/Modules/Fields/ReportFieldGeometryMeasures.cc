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



/*
 *  ReportFieldGeometryMeasures.cc: 
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   March 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Dataflow/Modules/Fields/ReportFieldGeometryMeasures.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Dataflow/GuiInterface/GuiVar.h>
#include <Core/Datatypes/Mesh.h>
#include <iostream>

namespace SCIRun {

class ReportFieldGeometryMeasures : public Module
{
private:
  GuiString simplexString_;
  GuiInt xFlag_;
  GuiInt yFlag_;
  GuiInt zFlag_;
  GuiInt idxFlag_;
  GuiInt sizeFlag_;
  GuiInt nNbrsFlag_;
  GuiInt normalsFlag_;
public:
  ReportFieldGeometryMeasures(GuiContext* ctx);
  virtual ~ReportFieldGeometryMeasures();

  virtual void execute();

};


DECLARE_MAKER(ReportFieldGeometryMeasures)

ReportFieldGeometryMeasures::ReportFieldGeometryMeasures(GuiContext* ctx)
  : Module("ReportFieldGeometryMeasures", ctx, Filter, "MiscField", "SCIRun"),
    simplexString_(get_ctx()->subVar("simplexString"), "Node"),
    xFlag_(get_ctx()->subVar("xFlag"), 1), 
    yFlag_(get_ctx()->subVar("yFlag"), 1),
    zFlag_(get_ctx()->subVar("zFlag"), 1), 
    idxFlag_(get_ctx()->subVar("idxFlag"), 0),
    sizeFlag_(get_ctx()->subVar("sizeFlag"), 0),
    nNbrsFlag_(get_ctx()->subVar("numNbrsFlag"), 0),
    normalsFlag_(get_ctx()->subVar("normalsFlag"), 0)
{
}



ReportFieldGeometryMeasures::~ReportFieldGeometryMeasures()
{
}


void
ReportFieldGeometryMeasures::execute()
{
  FieldHandle fieldhandle;
  if (!get_input_handle("Input Field", fieldhandle)) return;

  MeshHandle mesh = fieldhandle->mesh();

  //! This is a hack for now, it is definitely not an optimal way
  int syncflag = 0;
  if (simplexString_.get() == "Node")
    syncflag = Mesh::NODES_E | Mesh::NODE_NEIGHBORS_E;
  else if (simplexString_.get() == "Edge")
    syncflag = Mesh::EDGES_E | Mesh::EDGE_NEIGHBORS_E;
  else if (simplexString_.get() == "Face")
    syncflag = Mesh::FACES_E | Mesh::FACE_NEIGHBORS_E;
  else if (simplexString_.get() == "Cell")
    syncflag = Mesh::CELLS_E;
  // TOTAL HACK!
  else if (simplexString_.get() == "Elem")
    syncflag = (Mesh::ALL_ELEMENTS_E | 
		Mesh::NODE_NEIGHBORS_E | 
		Mesh::EDGE_NEIGHBORS_E | 
		Mesh::FACE_NEIGHBORS_E);

  mesh->synchronize(syncflag);

  bool nnormals =
    normalsFlag_.get() && simplexString_.get() == "Node";
  const bool fnormals =
    normalsFlag_.get() && simplexString_.get() == "Face";

  if (nnormals && !mesh->has_normals())
  {
    warning("This mesh type does not contain normals, skipping.");
    nnormals = false;
  }
  else if (normalsFlag_.get() && !(nnormals || fnormals))
  {
    warning("Cannot compute normals at that simplex location, skipping.");
  }
  if (nnormals)
  {
    mesh->synchronize(Mesh::NORMALS_E);
  }

  const TypeDescription *meshtd = mesh->get_type_description();
  const TypeDescription *simptd = 
    scinew TypeDescription(meshtd->get_name()+"::"+simplexString_.get(), 
			   meshtd->get_h_file_path(),
			   meshtd->get_namespace());
  CompileInfoHandle ci =
    ReportFieldGeometryMeasuresAlgo::get_compile_info(meshtd, simptd, nnormals, fnormals);
  Handle<ReportFieldGeometryMeasuresAlgo> algo;
  if (!module_dynamic_compile(ci, algo)) 
  {
    const string err = (string("Unable to compile ReportFieldGeometryMeasures for data at") +
			simplexString_.get() + 
			string(" in a ") +
			meshtd->get_name());
    error(err.c_str());
    return;
  }

  // Execute and Send (ha, no extraneous local variables here!)
  MatrixHandle mh(algo->execute(this, mesh,
                                xFlag_.get(),
                                yFlag_.get(),
                                zFlag_.get(), 
                                idxFlag_.get(), 
                                sizeFlag_.get(), 
                                nNbrsFlag_.get()));

  send_output_handle("Output Measures Matrix", mh);
}



CompileInfoHandle
ReportFieldGeometryMeasuresAlgo::get_compile_info(const TypeDescription *mesh_td,
				    const TypeDescription *simplex_td,
				    bool nnormals, bool fnormals)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(__FILE__);
  string template_class_name("ReportFieldGeometryMeasuresAlgoT");
  static const string base_class_name("ReportFieldGeometryMeasuresAlgo");

  if (nnormals) { template_class_name += "NN"; }
  if (fnormals) { template_class_name += "FN"; }

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       simplex_td->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       mesh_td->get_name() + ", " + simplex_td->get_name());

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  mesh_td->fill_compile_info(rval);
  simplex_td->fill_compile_info(rval);
  return rval;
}


} // End namespace SCIRun


