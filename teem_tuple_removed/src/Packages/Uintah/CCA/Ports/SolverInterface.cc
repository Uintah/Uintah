
#include <Packages/Uintah/CCA/Ports/SolverInterface.h>

using namespace Uintah;

SolverInterface::SolverInterface()
{
}

SolverInterface::~SolverInterface()
{
}

SolverParameters::~SolverParameters()
{
  solveOnExtraCells = true;
}
