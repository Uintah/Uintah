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

//----- EnthalpySolver.h -----------------------------------------------

#ifndef Uintah_Component_Arches_EnthalpySolver_h
#define Uintah_Component_Arches_EnthalpySolver_h

/**************************************
CLASS
   EnthalpySolver
   
   Class EnthalpySolver linearizes and solves momentum
   equation on a grid hierarchy


GENERAL INFORMATION
   EnthalpySolver.h - declaration of the class
   
   Author: Rajesh Rawat (rawat@crsim.utah.edu)

   All major modifications since 01.01.2004 done by:
   Stanislav Borodai(borodai@crsim.utah.edu)
   
   Creation Date:   Mar 1, 2000
   
   C-SAFE 
   

KEYWORDS


DESCRIPTION
   Class EnthalpySolver linearizes and solves scalar
   equation on a grid hierarchy


WARNING
   none

************************************************************************/

#include <CCA/Ports/SchedulerP.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <CCA/Ports/DataWarehouseP.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <CCA/Components/Arches/ArchesVariables.h>
#include <CCA/Components/Arches/ArchesConstVariables.h>

namespace Uintah {
  class ArchesLabel;
  class MPMArchesLabel;
  class ProcessorGroup;
  class TurbulenceModel;
  class PhysicalConstants;
  class Discretization;
  class Source;
  class BoundaryCondition;
  class RHSSolver;
  class DORadiationModel;
  class Ray; 
  class TimeIntegratorLabel;

class EnthalpySolver {

public:


  EnthalpySolver(const ArchesLabel* label, 
                 const MPMArchesLabel* MAlb, 
                 TurbulenceModel* turb_model, 
                 BoundaryCondition* bndry_cond,
                 PhysicalConstants* physConst,
                 const ProcessorGroup* myworld);


  ~EnthalpySolver();

  // GROUP: Problem Setup :
  ///////////////////////////////////////////////////////////////////////
  // Set up the problem specification database
  void problemSetup(const ProblemSpecP& params);

  // GROUP: Access Functions :
  ///////////////////////////////////////////////////////////////////////
    // Boolean to see whether or not DO Radiation model
    // is used
  inline bool checkDORadiation() const{
    return d_DORadiationCalc;
  }
  inline bool checkRadiation() const{
    return d_radiationCalc;
  }


  inline void setAdiabaticAirEnthalpy(double H_air) {
    d_H_air = H_air;
  }

  // GROUP: Schedule Action :
  ///////////////////////////////////////////////////////////////////////
  // Schedule Solve of linearized scalar equation
  void solve(const LevelP& level,
             SchedulerP& sched,
             const PatchSet* patches,
             const MaterialSet* matls,
             const TimeIntegratorLabel* timelabels);

  ///////////////////////////////////////////////////////////////////////
  // Schedule Build of linearized matrix
  void sched_buildLinearMatrix(const LevelP& level,
                               SchedulerP&, 
                               const PatchSet* patches,
                               const MaterialSet* matls,
                               const TimeIntegratorLabel* timelabels);

  ///////////////////////////////////////////////////////////////////////
  // Schedule Linear Solve for Enthalpy[index]
  void sched_enthalpyLinearSolve(SchedulerP&, 
                                 const PatchSet* patches,
                                 const MaterialSet* matls,
                                 const TimeIntegratorLabel* timelabels);

  inline void setMMS(bool doMMS) {
    d_doMMS=doMMS;
  }

protected:

private:

  // GROUP: Constructors (Private):
  ////////////////////////////////////////////////////////////////////////
  // Default : Construct an empty instance of the Pressure solver.
  EnthalpySolver();

  // GROUP: Action Methods (private) :
  ///////////////////////////////////////////////////////////////////////
  // Actually Build the linear matrix
  //    [in] 
  //        add documentation here
  void buildLinearMatrix(const ProcessorGroup* pc,
                         const PatchSubset* patches,
                         const MaterialSubset* /*matls*/,
                         DataWarehouse* old_dw,
                         DataWarehouse* new_dw,
                         const TimeIntegratorLabel* timelabels);

  ///////////////////////////////////////////////////////////////////////
  // Actually Solver the Linear System for Enthalpy[index]
  //    [in] 
  //        add documentation here
  void enthalpyLinearSolve(const ProcessorGroup* pc,
                           const PatchSubset* patches,
                           const MaterialSubset* /*matls*/,
                           DataWarehouse* old_dw,
                           DataWarehouse* new_dw,
                           const TimeIntegratorLabel* timelabels);

private:
  // const VarLabel* (required)
  const ArchesLabel* d_lab;
  const MPMArchesLabel* d_MAlab;

  // computes coefficients
  Discretization* d_discretize;
  // computes sources
  Source* d_source;
  // linear solver
  RHSSolver* d_rhsSolver;
  // turbulence model
  TurbulenceModel* d_turbModel;
  // boundary condition
  BoundaryCondition* d_boundaryCondition;
  // physical constants
  PhysicalConstants* d_physicalConsts;
  
  
  // radiation models
  DORadiationModel* d_DORadiation;
  bool d_radiationCalc;
  bool d_DORadiationCalc;
  
  const PatchSet* d_perproc_patches;

  const ProcessorGroup* d_myworld;
  int d_conv_scheme;
  int d_limiter_type;
  int d_boundary_limiter_type;
  bool d_central_limiter;
  bool d_dynScalarModel;
  double d_turbPrNo;
  double d_H_air;
  bool d_doMMS;
  
  vector<string> d_new_sources; 

}; // End class EnthalpySolver
} // End namespace Uintah


#endif

