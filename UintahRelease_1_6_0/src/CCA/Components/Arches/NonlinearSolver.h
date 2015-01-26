/*
 * The MIT License
 *
 * Copyright (c) 1997-2015 The University of Utah
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

//----- PicardNonlinearSolver.h -----------------------------------------------

#ifndef Uintah_Component_Arches_NonlinearSolver_h
#define Uintah_Component_Arches_NonlinearSolver_h

#include <CCA/Components/Arches/Arches.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>

/**************************************
CLASS
   NonlinearSolver
   
   Class NonlinearSolver is an abstract base class
   which defines the operations needed to implement
   a nonlinear solver for the ImplicitTimeIntegrator.

GENERAL INFORMATION
   NonlinearSolver.h - declaration of the class
   
   Author: Rajesh Rawat (rawat@crsim.utah.edu)

   All major modifications since 01.01.2004 done by:
   Stanislav Borodai(borodai@crsim.utah.edu)
      
   Creation Date:   Mar 1, 2000
   
   C-SAFE 
   
   
KEYWORDS


DESCRIPTION
   Class NonlinearSolver is an abstract type defining the interface
   for operations necessary for solving the nonlinear systems that
   arise during an implicit integration.

WARNING
   none
****************************************/

namespace Uintah {
class EnthalpySolver;
class TimeIntegratorLabel;
class PartVel; 
class DQMOM;
class MomentumSolver;
class NonlinearSolver {

public:

  // GROUP: Constructors:
  ////////////////////////////////////////////////////////////////////////
  // Blank constructor for NonlinearSolver.
  NonlinearSolver(const ProcessorGroup* myworld);

  // GROUP: Destructors:
  ////////////////////////////////////////////////////////////////////////
  // Virtual destructor for NonlinearSolver.
  virtual ~NonlinearSolver();


  // GROUP: Problem Setup :
  ///////////////////////////////////////////////////////////////////////
  // Interface for Set up of the problem specification database
  virtual void problemSetup(const ProblemSpecP& db,SimulationStateP&) = 0;


  virtual void sched_interpolateFromFCToCC(SchedulerP&, 
                                           const PatchSet* patches,
                                           const MaterialSet* matls,
                                           const TimeIntegratorLabel* timelabels) = 0;
  // GROUP: Schedule Action Computations :
  ///////////////////////////////////////////////////////////////////////
  // Interface for Solve the nonlinear system, return some error code.
  //    [in] 
  //        documentation here
  //    [out] 
  //        documentation here
  virtual int nonlinearSolve( const LevelP& level,
                              SchedulerP& sched
#                             ifdef WASATCH_IN_ARCHES
                              , Wasatch::Wasatch& wasatch,
                              ExplicitTimeInt* d_timeIntegrator,
                              SimulationStateP& state
#                             endif // WASATCH_IN_ARCHES
                              ) = 0;


  const std::string& getTimeIntegratorType() const
  {
    return d_timeIntegratorType;
  }
  virtual double recomputeTimestep(double current_dt) = 0;

  virtual double getAdiabaticAirEnthalpy() const = 0;

  virtual bool restartableTimesteps() = 0;

  virtual MomentumSolver* get_momentum_solver() = 0;

  virtual void setPartVel(PartVel* partVel) = 0; 

  virtual void setDQMOMSolver(DQMOM* dqmomSolver) = 0;
  
  virtual void setCQMOMSolver(CQMOM* cqmomSolver) = 0;

  virtual void setInitVelConditionInterface( const Patch          * patch, 
                                             SFCXVariable<double> & uvel, 
                                             SFCYVariable<double> & vvel, 
                                             SFCZVariable<double> & wvel ) = 0;
  
  virtual void checkMomBCs( SchedulerP& sched,
                            const LevelP& level,
                            const MaterialSet* matls) = 0;
  
  inline void set_use_wasatch_mom_rhs(const bool useWasatchMomRHS) { d_useWasatchMomRHS = useWasatchMomRHS; }
  inline bool get_use_wasatch_mom_rhs() { return d_useWasatchMomRHS; }

protected:
   const ProcessorGroup * d_myworld;
   std::string            d_timeIntegratorType;

   EnthalpySolver       * d_enthalpySolver;
   bool                   d_useWasatchMomRHS;

private:

}; // End class NonlinearSolver
} // End namespace Uintah

#endif


