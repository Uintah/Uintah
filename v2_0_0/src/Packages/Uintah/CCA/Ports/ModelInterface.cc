
#include <Packages/Uintah/CCA/Ports/ModelInterface.h>

using namespace Uintah;

ModelInterface::ModelInterface(const ProcessorGroup* myworld)
  : d_myworld(myworld)
{
}

ModelInterface::~ModelInterface()
{
}
