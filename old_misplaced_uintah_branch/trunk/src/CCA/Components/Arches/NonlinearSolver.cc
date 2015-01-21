//----- NonlinearSolver.cc ----------------------------------------------

#include <CCA/Components/Arches/NonlinearSolver.h>
#include <iostream>
using namespace std;

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

