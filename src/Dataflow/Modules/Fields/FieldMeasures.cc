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
  FieldIPort *ifp;
  MatrixOPort *omp;
  GuiString simplexString_;
  GuiInt xFlag_;
  GuiInt yFlag_;
  GuiInt zFlag_;
  GuiInt idxFlag_;
  GuiInt sizeFlag_;
  GuiInt nNbrsFlag_;
  MeshHandle m_;
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
    nNbrsFlag_(ctx->subVar("numNbrsFlag"))
{
}



FieldMeasures::~FieldMeasures()
{
}


void
FieldMeasures::execute()
{
  ifp = (FieldIPort *)get_iport("Input Field");
  FieldHandle fieldhandle;
  if (!ifp) {
    error("Unable to initialize iport 'Input Field'.");
    return;
  }

  if (!(ifp->get(fieldhandle) && fieldhandle.get_rep()))
  {
    return;
  }

  omp = (MatrixOPort *)get_oport("Output Measures Matrix");
  if (!omp) {
    error("Unable to initialize oport 'Output Measures Matrix'.");
    return;
  }

  m_ = fieldhandle->mesh();

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

  m_->synchronize(syncflag);

  const TypeDescription *meshtd = m_->get_type_description();
  const TypeDescription *simptd = 
    scinew TypeDescription(meshtd->get_name()+"::"+simplexString_.get(), 
			   meshtd->get_h_file_path(),
			   meshtd->get_namespace());
  CompileInfoHandle ci = FieldMeasuresAlgo::get_compile_info(meshtd, simptd);
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
  omp->send(MatrixHandle(algo->execute(m_,
				       xFlag_.get(),
				       yFlag_.get(),
				       zFlag_.get(), 
				       idxFlag_.get(), 
				       sizeFlag_.get(), 
				       nNbrsFlag_.get(),
				       syncflag)));
}



} // End namespace SCIRun


