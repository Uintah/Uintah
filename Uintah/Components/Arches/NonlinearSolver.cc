//----- NonlinearSolver.cc ----------------------------------------------

/* REFERENCED */
static char *id="@(#) $Id$";

#include <Uintah/Components/Arches/NonlinearSolver.h>
#include <SCICore/Util/NotFinished.h>

using namespace Uintah::ArchesSpace;

//****************************************************************************
// Interface constructor for NonlinearSolver
//****************************************************************************
NonlinearSolver::NonlinearSolver()
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
// Revision 1.7  2000/06/04 22:40:14  bbanerje
// Added Cocoon stuff, changed task, require, compute, get, put arguments
// to reflect new declarations. Changed sub.mk to include all the new files.
//
//
