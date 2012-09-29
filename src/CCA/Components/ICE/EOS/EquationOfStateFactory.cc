/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <CCA/Components/ICE/EOS/EquationOfStateFactory.h>
#include <CCA/Components/ICE/EOS/IdealGas.h>
#include <CCA/Components/ICE/EOS/HardSphereGas.h>
#include <CCA/Components/ICE/EOS/JWL.h>
#include <CCA/Components/ICE/EOS/TST.h>
#include <CCA/Components/ICE/EOS/JWLC.h>
#include <CCA/Components/ICE/EOS/Murnaghan.h>
#include <CCA/Components/ICE/EOS/BirchMurnaghan.h>
#include <CCA/Components/ICE/EOS/KnaussSeaWater.h>
#include <CCA/Components/ICE/EOS/Gruneisen.h>
#include <CCA/Components/ICE/EOS/Tillotson.h>
#include <CCA/Components/ICE/EOS/Thomsen_Hartka_water.h>
#include <Core/Parallel/Parallel.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Malloc/Allocator.h>
#include <fstream>
#include <iostream>
#include <string>

using std::cerr;
using std::ifstream;
using std::ofstream;

using namespace Uintah;

EquationOfState* EquationOfStateFactory::create(ProblemSpecP& ps)
{
  ProblemSpecP EOS_ps = ps->findBlock("EOS");
  if(!EOS_ps){
    throw ProblemSetupException("ERROR ICE: Cannot find EOS tag", __FILE__, __LINE__);
  }
  
  std::string EOS;
  if(!EOS_ps->getAttribute("type",EOS)){
    throw ProblemSetupException("ERROR ICE: Cannot find EOS 'type' tag", __FILE__, __LINE__); 
  }
  
  // warnings
  if (EOS == "Murnaghan"){
    proc0cout << "______________________________________________________\n"
          << "  ICE: EOS: WARNING: \n"
          << "  The Murnaghan equation of state has a discontinuity in dp_drho at the \n"
          << "  reference density (rho0)."
          << "  This will cause the Equilibration pressure calculation to fail\n"
          << "  under certain circumstances.\n"
          << "  Run the octave script: \n"
          << "    src/CCA/Components/ICE/Matlab/Murnahan.m \n"
          << "  to see the issue. \n"
          << "______________________________________________________\n" << endl;
  }
  
  
  if (EOS == "ideal_gas") 
    return(scinew IdealGas(EOS_ps));
  else if (EOS == "hard_sphere_gas") 
    return(scinew HardSphereGas(EOS_ps));
  else if (EOS == "TST") 
    return(scinew TST(EOS_ps));
  else if (EOS == "JWL") 
    return(scinew JWL(EOS_ps));
  else if (EOS == "JWLC") 
    return(scinew JWLC(EOS_ps));
  else if (EOS == "Murnaghan") 
    return(scinew Murnaghan(EOS_ps));
  else if (EOS == "BirchMurnaghan") 
    return(scinew BirchMurnaghan(EOS_ps));
  else if (EOS == "Gruneisen") 
    return(scinew Gruneisen(EOS_ps));
  else if (EOS == "Tillotson") 
    return(scinew Tillotson(EOS_ps));    
  else if (EOS == "Thomsen_Hartka_water") 
    return(scinew Thomsen_Hartka_water(EOS_ps));    
  else if (EOS == "KnaussSeaWater") 
    return(scinew KnaussSeaWater(EOS_ps));    
  else{
    ostringstream warn;
    warn << "ERROR ICE: Unknown Equation of State ("<< EOS << " )\n"
         << "Valid equations of State:\n" 
         << "ideal_gas\n"
         << "TST\n"
         << "JWL\n"
         << "JWLC\n"
         << "Murnaghan\n"
         << "Gruneisen\n"
         << "Tillotson\n"
         << "KnaussSeaWater\n"
         << "Thomsen_Hartka_water" << endl;
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
  }

}
