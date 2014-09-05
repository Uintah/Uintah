/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


//----- MixingRxnTable.cc --------------------------------------------------
//
// See MixingRxnTable.h for additional information.
//

// includes for Arches
#include <Uintah/CCA/Components/Arches/MixingRxnTable.h>
#include <Uintah/CCA/Components/Arches/Properties.h>
#include <Uintah/CCA/Components/Arches/Arches.h>

// includes for Uintah
#include <Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Uintah/Core/Exceptions/InvalidValue.h>
#include <Uintah/Core/Exceptions/ProblemSetupException.h>

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


