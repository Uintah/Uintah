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

#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/Matrix.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Packages/CardioWave/Core/Model/ModelAlgo.h>

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

namespace CardioWave {

using namespace SCIRun;

class BuildMembraneTable : public Module {
public:
  BuildMembraneTable(GuiContext*);

  virtual ~BuildMembraneTable();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);
};


DECLARE_MAKER(BuildMembraneTable)
BuildMembraneTable::BuildMembraneTable(GuiContext* ctx)
  : Module("BuildMembraneTable", ctx, Source, "Model", "CardioWave")
{
}

BuildMembraneTable::~BuildMembraneTable(){
}

void BuildMembraneTable::execute()
{
  FieldHandle ElementType;
  FieldHandle MembraneModel;
  MatrixHandle CompToGeom;
  MatrixHandle NodeLink;
  MatrixHandle ElemLink;
  MatrixHandle Table;
  MatrixHandle MappingMatrix;
  
  if (!(get_input_handle("ElementType",ElementType,true))) return;
  if (!(get_input_handle("MembraneModel",MembraneModel,true))) return;
  
  get_input_handle("CompToGeom",CompToGeom,false);
  get_input_handle("ElemLink",ElemLink,false);
  get_input_handle("NodeLink",NodeLink,false);
  
  MembraneTable MemTable;
  ModelAlgo algo(this);
  algo.DMDBuildMembraneTable(ElementType,MembraneModel,CompToGeom,NodeLink,ElemLink,MemTable,MappingMatrix);
  algo.DMDMembraneTableToMatrix(MemTable,Table);
  
  send_output_handle("MembraneTable",Table,true);
  send_output_handle("MappinfMatrix",MappingMatrix,true);

}

void
 BuildMembraneTable::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace CardioWave


