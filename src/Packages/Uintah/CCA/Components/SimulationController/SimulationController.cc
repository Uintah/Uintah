#include <Packages/Uintah/CCA/Components/SimulationController/SimulationController.h>

using namespace Uintah;

// for calculating memory usage when sci-malloc is disabled.
char* SimulationController::start_addr = NULL;

SimulationController::SimulationController(const ProcessorGroup* myworld)
: UintahParallelComponent(myworld)
{
}

SimulationController::~SimulationController()
{
}
