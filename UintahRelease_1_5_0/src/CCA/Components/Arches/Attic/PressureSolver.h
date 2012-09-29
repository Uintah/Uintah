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

//----- PressureSolver.h -----------------------------------------------

#ifndef Uintah_Components_Arches_PressureSolver_h
#define Uintah_Components_Arches_PressureSolver_h

#include <CCA/Ports/SchedulerP.h>
#include <CCA/Components/Arches/ArchesConstVariables.h>
#include <CCA/Components/Arches/CellInformationP.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <CCA/Ports/DataWarehouseP.h>
#include <Core/Grid/LevelP.h>


namespace Uintah {

class MPMArchesLabel;
class ArchesLabel;
class ProcessorGroup;
class ArchesVariables;
class PhysicalConstants;
class Discretization;
class Source;
class BoundaryCondition;
class LinearSolver;
class TimeIntegratorLabel;

using namespace SCIRun;

/**************************************

CLASS
   PressureSolver
   
   Class PressureSolver linearizes and solves pressure
   equation on a grid hierarchy


GENERAL INFORMATION
   PressureSolver.h - declaration of the class
   
   Author: Rajesh Rawat (rawat@crsim.utah.edu)

   All major modifications since 01.01.2004 done by:
   Stanislav Borodai(borodai@crsim.utah.edu)
   
   Creation Date:   Mar 1, 2000
   
   C-SAFE 
   
   
KEYWORDS


DESCRIPTION
   Class PressureSolver linearizes and solves pressure
   equation on a grid hierarchy


WARNING
   none

************************************************************************/

class PressureSolver {

public:
  
  // GROUP: Constructors:
  ////////////////////////////////////////////////////////////////////////
  // Construct an instance of the Pressure solver.
  // PRECONDITIONS
  // POSTCONDITIONS
  //   A linear level solver is partially constructed.  
  PressureSolver(ArchesLabel* label,
                 const MPMArchesLabel* MAlb,
                 BoundaryCondition* bndry_cond,
                 PhysicalConstants* phys_const,
                 const ProcessorGroup* myworld);

  // GROUP: Destructors:
  ////////////////////////////////////////////////////////////////////////
  // Destructor
  ~PressureSolver();

  // GROUP: Problem Setup :
  ///////////////////////////////////////////////////////////////////////
  // Set up the problem specification database
  void problemSetup(const ProblemSpecP& params);

  // GROUP: Schedule Action :
  ///////////////////////////////////////////////////////////////////////
  // Schedule Solve of linearized pressure equation
  void sched_solve(const LevelP& level, SchedulerP&,
                   const TimeIntegratorLabel* timelabels,
                   bool extraProjection,
                   bool d_EKTCorrection,
                   bool doing_EKT_now);

  ///////////////////////////////////////////////////////////////////////
  // Schedule the build of the linearized eqn
  void sched_buildLinearMatrix(SchedulerP&, 
                               const PatchSet* patches,
                               const MaterialSet* matls,
                               const TimeIntegratorLabel* timelabels,
                               bool extraProjection,
                               bool d_EKTCorrection,
                               bool doing_EKT_now);

  void sched_pressureLinearSolve(const LevelP& level, SchedulerP& sched,
                                 const TimeIntegratorLabel* timelabels,
                                 bool extraProjection,
                                 bool d_EKTCorrection,
                                 bool doing_EKT_now);

  ///////////////////////////////////////////////////////////////////////
  // Schedule the addition of the hydrostatic term to the relative pressure

  void sched_addHydrostaticTermtoPressure(SchedulerP& sched,
                                          const PatchSet* patches,
                                          const MaterialSet* matls,
                                          const TimeIntegratorLabel* timelabels);

  inline void setPressureCorrectionFlag(bool pressure_correction) {
    d_pressure_correction = pressure_correction;
  }
  inline void setMMS(bool doMMS) {
    d_doMMS=doMMS;
  }
protected:

private:

  // GROUP: Constructors (Private):
  ////////////////////////////////////////////////////////////////////////
  // Default : Construct an empty instance of the Pressure solver.
  PressureSolver(const ProcessorGroup* myworld);

  // GROUP: Action Methods (private) :
  ///////////////////////////////////////////////////////////////////////
  // Actually Build the linear matrix
  //    [in] 
  //        add documentation here
  void buildLinearMatrix(const ProcessorGroup* pc,
                         const PatchSubset* patches,
                         const MaterialSubset* matls,
                         DataWarehouse* new_dw,
                         DataWarehouse* matrix_dw,
                         const TimeIntegratorLabel* timelabels,
                         bool extraProjection,
                         bool d_EKTCorrection,
                         bool doing_EKT_now);

  void pressureLinearSolve_all(const ProcessorGroup* pc,
                               const PatchSubset* patches,
                               const MaterialSubset* matls,
                               DataWarehouse* new_dw,
                               DataWarehouse* matrix_dw,
                               const TimeIntegratorLabel* timelabels,
                               bool extraProjection,
                               bool d_EKTCorrection,
                               bool doing_EKT_now);

  void pressureLinearSolve(const ProcessorGroup* pc,
                           const Patch* patch,
                           const int matlIndex,
                           DataWarehouse* new_dw,
                           DataWarehouse* matrix_dw,
                           ArchesVariables& pressureVars,
                           const TimeIntegratorLabel* timelabels,
                           bool extraProjection,
                           bool d_EKTCorrection,
                           bool doing_EKT_now);
                           
  ////////////////////////////////////////////////////////////////////////
  // Set stencil weights. (Pressure)
  // It uses second order hybrid differencing for computing
  // coefficients
  void calculatePressureCoeff(const Patch* patch,
                              CellInformation* cellinfo,
                              ArchesVariables* vars,
                              ArchesConstVariables* constvars); 

  ////////////////////////////////////////////////////////////////////////
  // Modify stencil weights (Pressure) to account for voidage due
  // to multiple materials
  void mmModifyPressureCoeffs(const Patch* patch,
                              ArchesVariables* vars,
                              ArchesConstVariables* constvars);

  ////////////////////////////////////////////////////////////////
  // addition of hydrostatic term to relative pressure
  void addHydrostaticTermtoPressure(const ProcessorGroup* pc,
                                    const PatchSubset* patches,
                                    const MaterialSubset* matls,
                                    DataWarehouse* old_dw,
                                    DataWarehouse* new_dw,
                                    const TimeIntegratorLabel* timelabels);

  ///////////////////////////////////////////////////////////////////////
  // Actually do normPressure
  //    [in] 
  //        add documentation here
  void normPressure(const Patch* patch,
                    ArchesVariables* vars);
                    
  void updatePressure(const Patch* patch,
                      ArchesVariables* vars);

 private:

  bool d_always_construct_A;
  bool d_construct_A; 
  bool d_construct_solver_obj; 

  // const VarLabel* (required)
  ArchesLabel* d_lab;
  // for multimaterial
  const MPMArchesLabel* d_MAlab;

  // computes coefficients
  Discretization* d_discretize;
  // computes sources
  Source* d_source;
  // linear solver
  LinearSolver* d_linearSolver;
  // boundary condition
  BoundaryCondition* d_boundaryCondition;
  // physical constants
  PhysicalConstants* d_physicalConsts;

  int d_iteration;

  // Maximum number of iterations to take before stopping/giving up.
  int d_maxIterations;
  //reference points for the solvers
  IntVector d_pressRef;
  const Patch* d_pressRefPatch;
  int d_pressRefProc;
  const PatchSet* d_perproc_patches;

  const ProcessorGroup* d_myworld;
#ifdef multimaterialform
  // set the values in problem setup
  MultiMaterialInterface* d_mmInterface;
  MultiMaterialSGSModel* d_mmSGSModel;
#endif
  bool d_pressure_correction;
  bool d_norm_pres;
  bool d_doMMS;
  bool d_do_only_last_projection;
}; // End class PressureSolver

} // End namespace Uintah

#endif

