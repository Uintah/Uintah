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
 *  FieldMeasures.cc: 
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   March 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Dataflow/Modules/Fields/FieldMeasures.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Ports/MatrixPort.h>
#include <Core/Datatypes/ColumnMatrix.h>
#include <Core/Datatypes/DenseMatrix.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Datatypes/Mesh.h>
#include <iostream>

namespace SCIRun {

class FieldMeasures : public Module
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
  FieldMeasures(GuiContext* ctx);
  virtual ~FieldMeasures();

  virtual void execute();

};


DECLARE_MAKER(FieldMeasures)

FieldMeasures::FieldMeasures(GuiContext* ctx)
  : Module("FieldMeasures", ctx, Filter, "FieldsOther", "SCIRun"),
    simplexString_(ctx->subVar("simplexString")),
    xFlag_(ctx->subVar("xFlag")), 
    yFlag_(ctx->subVar("yFlag")),
    zFlag_(ctx->subVar("zFlag")), 
    idxFlag_(ctx->subVar("idxFlag")),
    sizeFlag_(ctx->subVar("sizeFlag")),
    nNbrsFlag_(ctx->subVar("numNbrsFlag")),
    normalsFlag_(ctx->subVar("normalsFlag"))
{
}



FieldMeasures::~FieldMeasures()
{
}


void
FieldMeasures::execute()
{
  FieldIPort *ifp = (FieldIPort *)get_iport("Input Field");
  FieldHandle fieldhandle;
  if (!(ifp->get(fieldhandle) && fieldhandle.get_rep()))
  {
    return;
  }

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
    FieldMeasuresAlgo::get_compile_info(meshtd, simptd, nnormals, fnormals);
  Handle<FieldMeasuresAlgo> algo;
  if (!module_dynamic_compile(ci, algo)) 
  {
    const string err = (string("Unable to compile FieldMeasures for data at") +
			simplexString_.get() + 
			string(" in a ") +
			meshtd->get_name());
    error(err.c_str());
    return;
  }

  // Execute and Send (ha, no extraneous local variables here!)
  MatrixOPort *omp = (MatrixOPort *)get_oport("Output Measures Matrix");
  omp->send(MatrixHandle(algo->execute(this, mesh,
				       xFlag_.get(),
				       yFlag_.get(),
				       zFlag_.get(), 
				       idxFlag_.get(), 
				       sizeFlag_.get(), 
				       nNbrsFlag_.get())));
}



CompileInfoHandle
FieldMeasuresAlgo::get_compile_info(const TypeDescription *mesh_td,
				    const TypeDescription *simplex_td,
				    bool nnormals, bool fnormals)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(__FILE__);
  string template_class_name("FieldMeasuresAlgoT");
  static const string base_class_name("FieldMeasuresAlgo");

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


