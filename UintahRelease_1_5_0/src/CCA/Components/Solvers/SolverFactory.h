/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
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


#ifndef Packages_Uintah_CCA_Components_Solvers_SolverFactory_h
#define Packages_Uintah_CCA_Components_Solvers_SolverFactory_h

/*--------------------------------------------------------------------------
CLASS
   SolverFactory
   
   Main class of the Solvers component.

GENERAL INFORMATION

   File: SolverFactory.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2005 SCI Group

KEYWORDS
   SolverFactory, SolverInterface, ProblemSpecP.

DESCRIPTION
   Class SolverFactory arbitrates between different solvers for an
   elliptic equation (normally, a pressure equation in implicit ICE;
   pressure defined at cell-centered). It is created in StandAlone/sus.cc.
   We support our own solvers (CGSolver) and several Hypre library solvers
   and preconditioners, among which: PFMG, SMG, FAC, AMG, CG. Solver
   arbitration is based on input file parameters.
  
WARNING
   Make sure to comment out any solver that is not completed yet, otherwise
   sus cannot pass linking.
   --------------------------------------------------------------------------*/

#include <Core/ProblemSpec/ProblemSpecP.h>
#include <CCA/Ports/SolverInterface.h>

namespace Uintah {

  class ProcessorGroup;

  class SolverFactory
    {
    public:
      // this function has a switch for all known solvers
    
      static SolverInterface* create(ProblemSpecP& ps,
                                     const ProcessorGroup* world,
                                     string cmdline);

    };
} // End namespace Uintah

#endif // Packages_Uintah_CCA_Components_Solvers_SolverFactory_h
