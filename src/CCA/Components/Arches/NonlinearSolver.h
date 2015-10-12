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
class TimeIntegratorLabel;
class PartVel;
class DQMOM;
class CQMOM;
class CQMOM_Convection;
class CQMOMSourceWrapper;
class ArchesBCHelper;
class NonlinearSolver {

public:

  NonlinearSolver( const ProcessorGroup* myworld );

  virtual ~NonlinearSolver();

  void commonProblemSetup( ProblemSpecP db );

  virtual void problemSetup( const ProblemSpecP& db, SimulationStateP&, GridP& ) = 0;

  virtual int nonlinearSolve( const LevelP& level,
                              SchedulerP& sched ) = 0;

  virtual double recomputeTimestep(double current_dt) = 0;

  virtual bool restartableTimesteps() = 0;

  virtual void checkMomBCs( SchedulerP& sched,
                            const LevelP& level,
                            const MaterialSet* matls) = 0;

  virtual void initialize( const LevelP& lvl, SchedulerP& sched, const bool doing_restart ) = 0;

  virtual void sched_restartInitialize( const LevelP& level, SchedulerP& sched ) = 0;
  virtual void sched_restartInitializeTimeAdvance( const LevelP& level, SchedulerP& sched ) = 0; 

  class NLSolverBuilder {

    public:

      NLSolverBuilder(){}

      virtual ~NLSolverBuilder() {}

      virtual NonlinearSolver* build() = 0;

  };

  /** @brief specialized CFL condition **/
  inline bool get_underflow(){ return d_underflow; }

  /** @brief Return the initial dt **/
  inline double get_initial_dt(){ return d_initial_dt; }

  /** @brief Set the helper **/
  void set_bchelper( std::map< int, ArchesBCHelper* >* helper ){ _bcHelperMap = helper; }

protected:

   const ProcessorGroup * d_myworld;
   std::string            d_timeIntegratorType;

   double                 d_initial_dt;
   bool                   d_underflow;

   typedef std::map< int, ArchesBCHelper* >* BCHelperMapT;
   BCHelperMapT _bcHelperMap;

private:

}; // End class NonlinearSolver
} // End namespace Uintah

#endif
