//----- LinearSolver.cc ----------------------------------------------

/* REFERENCED */
static char *id="@(#) $Id$";

#include <Uintah/Components/Arches/LinearSolver.h>
#include <Uintah/Interface/ProblemSpec.h>
#include <Uintah/Grid/Level.h>
#include <Uintah/Interface/Scheduler.h>
#include <Uintah/Interface/DataWarehouse.h>
#include <SCICore/Util/NotFinished.h>

using namespace Uintah::ArchesSpace;
using namespace std;

//****************************************************************************
// Private constructor for LinearSolver
//****************************************************************************
LinearSolver::LinearSolver()
{
}

//****************************************************************************
// Destructor
//****************************************************************************
LinearSolver::~LinearSolver()
{
}

//
// $Log$
// Revision 1.1  2000/06/04 22:40:58  bbanerje
// Added the LinearSolver.cc file.
//
//
