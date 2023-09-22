/*
 * The MIT License
 *
 * Copyright (c) 1997-2023 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <CCA/Components/PostProcessUda/Module.h>
#include <Core/DataArchive/DataArchive.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Material.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/MaterialManagerP.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <ostream>                         // for operator<<, basic_ostream
#include <vector>

using namespace Uintah;
using namespace std;

Module::Module()
{
}

Module::Module(ProblemSpecP     & prob_spec,
               MaterialManagerP & materialManager,
               Output           * dataArchiver,
               DataArchive      * dataArchive )
{
  d_materialManager  = materialManager;
  d_dataArchiver = dataArchiver;
  d_dataArchive  = dataArchive;

 if(!d_dataArchiver){
    throw InternalError("Module:couldn't get output port", __FILE__, __LINE__);
  }
}

Module::~Module()
{
}

//______________________________________________________________________
//  Parse the ups file and read timeStart and timeStop
void Module::readTimeStartStop(const ProblemSpecP & ps,
                               double & startTime,
                               double & stopTime)
{
  ProblemSpecP prob_spec = ps;
  prob_spec->require("timeStart",  startTime);
  prob_spec->require("timeStop",   stopTime);

  //__________________________________
  // bulletproofing
  // Start time < stop time
  if(startTime >= stopTime ){
    throw ProblemSetupException("\n ERROR:PostProcess: startTime >= stopTime. \n", __FILE__, __LINE__);
  }
  
  std::vector<int> udaTimesteps;
  d_dataArchive->queryTimesteps( udaTimesteps, d_udaTimes );

  if ( startTime < d_udaTimes[0] ){
    std::ostringstream warn;
    warn << "  Warning:PostProcess: The startTime (" << startTime
         << ") must be greater than the time at timestep 1 (" << d_udaTimes[0] << ")\n";
    proc0cout << warn.str();
  }
}




