#include <Packages/Uintah/CCA/Components/SimulationController/SimulationController.h>
#include <Core/Exceptions/InternalError.h>

using namespace Uintah;
using namespace SCIRun;

// for calculating memory usage when sci-malloc is disabled.
char* SimulationController::start_addr = NULL;

SimulationController::SimulationController(const ProcessorGroup* myworld)
: UintahParallelComponent(myworld)
{
}

SimulationController::~SimulationController()
{
}

void SimulationController::doCombinePatches(std::string /*fromDir*/)
{
  throw InternalError("Patch combining not implement for this simulation controller");
}
