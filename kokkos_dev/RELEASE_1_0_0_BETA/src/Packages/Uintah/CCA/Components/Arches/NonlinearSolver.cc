//----- NonlinearSolver.cc ----------------------------------------------

#include <Packages/Uintah/CCA/Components/Arches/NonlinearSolver.h>

using namespace Uintah;

//****************************************************************************
// Interface constructor for NonlinearSolver
//****************************************************************************
NonlinearSolver::NonlinearSolver(const ProcessorGroup* myworld)
   : d_myworld(myworld)
{
}

//****************************************************************************
// Destructor
//****************************************************************************
NonlinearSolver::~NonlinearSolver()
{
}

