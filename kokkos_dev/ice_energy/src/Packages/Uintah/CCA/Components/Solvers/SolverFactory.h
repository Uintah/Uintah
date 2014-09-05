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
   Make sure to #if 0 any solver that is not completed yet, otherwise sus
   cannot pass linkage.
   --------------------------------------------------------------------------*/
#ifndef Packages_Uintah_CCA_Components_Solvers_SolverFactory_h
#define Packages_Uintah_CCA_Components_Solvers_SolverFactory_h

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/CCA/Ports/SolverInterface.h>

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
