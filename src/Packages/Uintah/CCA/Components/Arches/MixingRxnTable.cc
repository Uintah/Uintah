//----- MixingRxnTable.cc --------------------------------------------------
//
// See MixingRxnTable.h for additional information.
//

// includes for Arches
#include <Packages/Uintah/CCA/Components/Arches/MixingRxnTable.h>
#include <Packages/Uintah/CCA/Components/Arches/Properties.h>
#include <Packages/Uintah/CCA/Components/Arches/Arches.h>

// includes for Uintah
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>

// includes for C++
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <fcntl.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <time.h>
#include <unistd.h>
//======================================================

using namespace std;
using namespace Uintah;
using namespace SCIRun;

//****************************************************************************
// Default constructor for MixingRxnTable
//****************************************************************************
MixingRxnTable::MixingRxnTable()
{
}

//****************************************************************************
// Destructor
//****************************************************************************
MixingRxnTable::~MixingRxnTable()
{
}

//****************************************************************************
// MixingRxnTable problemSetup
//****************************************************************************
void
MixingRxnTable::problemSetup( const ProblemSpecP& params )
{

}

//****************************************************************************
// MixingRxnTable getState 
//****************************************************************************
const std::vector<double>
MixingRxnTable::getState( const double * indepVarValues )
{
    exit(1);
}

//****************************************************************************
// MixingRxnTable verifyTable
//****************************************************************************
void
MixingRxnTable::verifyTable( bool diagnosticMode,
                             bool strictMode )
{
}

//****************************************************************************
// MixingRxnTable getDepVars
//****************************************************************************
const std::vector<std::string> &
MixingRxnTable::getDepVars()
{
    exit(1);
}

//****************************************************************************
// MixingRxnTable getIndepVars
//****************************************************************************
const std::vector<std::string> &
MixingRxnTable::getIndepVars()
{
    exit(1);
}


