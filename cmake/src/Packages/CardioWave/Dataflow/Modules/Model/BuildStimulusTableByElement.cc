/*
 *  BuildStimulusTableByElement.cc:
 *
 *  Written by:
 *   jeroen
 *   TODAY'S DATE HERE
 *
 */

#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/Matrix.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Dataflow/Network/Ports/MatrixPort.h>
#include <Packages/CardioWave/Core/Model/ModelAlgo.h>
#include <Core/Algorithms/Converter/ConverterAlgo.h>

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

namespace CardioWave {

using namespace SCIRun;

class BuildStimulusTableByElement : public Module {
public:
  BuildStimulusTableByElement(GuiContext*);

  virtual ~BuildStimulusTableByElement();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);
};


DECLARE_MAKER(BuildStimulusTableByElement)
BuildStimulusTableByElement::BuildStimulusTableByElement(GuiContext* ctx)
  : Module("BuildStimulusTableByElement", ctx, Source, "Model", "CardioWave")
{
}

BuildStimulusTableByElement::~BuildStimulusTableByElement(){
}

void BuildStimulusTableByElement::execute()
{
  FieldHandle ElementType;
  FieldHandle StimulusModel;
  MatrixHandle CompToGeom;
  MatrixHandle DomainType;
  MatrixHandle Table;
  MatrixHandle MappingMatrix;
  
  if (!(get_input_handle("ElementType",ElementType,true))) return;
  if (!(get_input_handle("StimulusModel",StimulusModel,true))) return;
  
  get_input_handle("CompToGeom",CompToGeom,false);
  get_input_handle("DomainType",DomainType,false);
  
  double domaintype;
  SCIRunAlgo::ConverterAlgo convalgo(this);
  convalgo.MatrixToDouble(DomainType,domaintype);
  
  StimulusTable StimTable;
  ModelAlgo algo(this);
  algo.DMDBuildStimulusTableByElement(ElementType,StimulusModel,CompToGeom,domaintype,StimTable,MappingMatrix);
  algo.DMDStimulusTableToMatrix(StimTable,Table);
  
  send_output_handle("StimulusTable",Table,true);
  send_output_handle("MappingMatrix",MappingMatrix,true);
}

void
 BuildStimulusTableByElement::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace CardioWave


