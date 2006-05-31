
#include <Packages/Uintah/CCA/Ports/ModelInterface.h>

using namespace Uintah;

ModelInterface::ModelInterface(const ProcessorGroup* myworld)
  : d_myworld(myworld)
{
}

ModelInterface::~ModelInterface()
{
}

bool ModelInterface::computesThermoTransportProps() const
{
  return d_modelComputesThermoTransportProps;
}

void ModelInterface::activateModel(GridP& , SimulationStateP& , ModelSetup* )
{

}

void ModelInterface::scheduleCheckNeedAddMaterial(SchedulerP&,
                                                  const LevelP& ,
                                                  const ModelInfo*)
{

}

void ModelInterface::setMPMLabel(MPMLabel*)
{
  // Currently only need to do this in the SteadyBurn model
}

