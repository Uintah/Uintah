//----- NonlinearSolver.cc ----------------------------------------------

/* REFERENCED */
static char *id="@(#) $Id$";

#include <Uintah/Components/Arches/NonlinearSolver.h>
#include <SCICore/Util/NotFinished.h>

using namespace Uintah::ArchesSpace;

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

//
// $Log$
// Revision 1.8  2000/09/20 18:05:33  sparker
// Adding support for Petsc and per-processor tasks
//
// Revision 1.7  2000/06/04 22:40:14  bbanerje
// Added Cocoon stuff, changed task, require, compute, get, put arguments
// to reflect new declarations. Changed sub.mk to include all the new files.
//
//
