/*

The MIT License

Copyright (c) 1997-2010 Center for the Simulation of Accidental Fires and 
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


//----- ExtraScalarSolver.h -----------------------------------------------

#ifndef Uintah_Component_Arches_ExtraScalarSolver_h
#define Uintah_Component_Arches_ExtraScalarSolver_h

/**************************************
CLASS
   ExtraScalarSolver
   
   Class ExtraScalarSolver linearizes and solves momentum
   equation on a grid hierarchy


GENERAL INFORMATION
   ExtraScalarSolver.h - declaration of the class
   
   Author: Rajesh Rawat (rawat@crsim.utah.edu)

   All major modifications since 01.01.2004 done by:
   Stanislav Borodai(borodai@crsim.utah.edu)
   
   Creation Date:   Mar 1, 2000
   
   C-SAFE 
   
   Copyright U of U 2000

KEYWORDS


DESCRIPTION
   Class ExtraScalarSolver linearizes and solves scalar
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
  class TimeIntegratorLabel;
  class ExtraScalarSrc;

class ExtraScalarSolver {

public:

  // GROUP: Constructors:
  ////////////////////////////////////////////////////////////////////////
  // Construct an instance of the Scalar solver.
  // PRECONDITIONS
  // POSTCONDITIONS
  //   A linear level solver is partially constructed.  
  ExtraScalarSolver(const ArchesLabel* label, 
                    const MPMArchesLabel* MAlb, 
                    PhysicalConstants* physConst);

  // GROUP: Destructors:
  ////////////////////////////////////////////////////////////////////////
  // Destructor
  ~ExtraScalarSolver();

  // GROUP: Problem Setup :
  ///////////////////////////////////////////////////////////////////////
  // Set up the problem specification database
  void problemSetup(const ProblemSpecP& params);

      // for setting the intial guess when performing a dummy solve        
      void sched_setInitialGuess(SchedulerP& sched,
                const PatchSet* patches,
                const MaterialSet* matls,
                const TimeIntegratorLabel* timelabels);


  // GROUP: Schedule Action :
  ///////////////////////////////////////////////////////////////////////
  // Schedule Solve of linearized scalar equation
  void solve(SchedulerP& sched,
             const PatchSet* patches,
             const MaterialSet* matls,
             const TimeIntegratorLabel* timelabels,
             bool d_EKTCorrection,
             bool doing_EKT_now);

  ///////////////////////////////////////////////////////////////////////
  // Schedule Build of linearized matrix
  void sched_buildLinearMatrix(SchedulerP&, 
                               const PatchSet* patches,
                               const MaterialSet* matls,
                               const TimeIntegratorLabel* timelabels,
                               bool d_EKTCorrection,
                               bool doing_EKT_now);

  ///////////////////////////////////////////////////////////////////////
  // Schedule Linear Solve for Scalar
  void sched_scalarLinearSolve(SchedulerP&, 
                               const PatchSet* patches,
                               const MaterialSet* matls,
                               const TimeIntegratorLabel* timelabels,
                               bool d_EKTCorrection,
                               bool doing_EKT_now);

  inline void setMMS(bool doMMS) {
    d_doMMS=doMMS;
  }
  inline const VarLabel* getScalarLabel() {
    return d_scalar_label;
  }
  inline double getScalarInitValue() {
    return d_scalar_init_value;
  }
  inline const string getScalarName() {
    return d_scalar_name;
  }
  inline void setTurbulenceModel(TurbulenceModel* turbModel) {
    d_turbModel = turbModel;
  }
  inline void setBoundaryCondition(BoundaryCondition* boundaryCondition) {
    d_boundaryCondition = boundaryCondition;
  }
  inline bool isDensityWeighted() {
    return d_scalar_density_weighted;
  }
  inline bool useforDen(){
    //if true, will use this scalar to compute rho
    return d_scalar_useforden;
    //warning!  This assumes (for now) that only one scalar
    //           can be used from all Extrascalars.  
  }
  inline const string getPrecompTabSrcName(){
    //if true, will use this scalar to compute rho
    return d_precompTabSrcName;
    //warning!  This assumes (for now) that only one scalar
    //           can be used from all Extrascalars.  
  }

  inline std::vector<ExtraScalarSrc*> getExtraScalarSources() {
    return d_extraScalarSources;
  }

  inline bool doCarbonBalance(){
    return d_carbon_balance;
  }

  inline bool doSulfurBalance(){
    return d_sulfur_balance;
  }

protected:


private:

      // GROUP: Constructors (Private):
      ////////////////////////////////////////////////////////////////////////
      // Default : Construct an empty instance of the solver.
      ExtraScalarSolver();

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
                             const TimeIntegratorLabel* timelabels,
                             bool d_EKTCorrection,
                             bool doing_EKT_now);

          // set the intial guess for nosolve
          void setInitialGuess(const ProcessorGroup* pc,
                                     const PatchSubset* patches,
                                     const MaterialSubset*,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw,
                                     const TimeIntegratorLabel* timelabels);


      ///////////////////////////////////////////////////////////////////////
      // Actually Solver the Linear System for Scalar
      //    [in] 
      //        add documentation here
      void scalarLinearSolve(const ProcessorGroup* pc,
                             const PatchSubset* patches,
                             const MaterialSubset* /*matls*/,
                             DataWarehouse* old_dw,
                             DataWarehouse* new_dw,
                             const TimeIntegratorLabel* timelabels,
                             bool d_EKTCorrection,
                             bool doing_EKT_now);


private:
      // const VarLabel* (required)
      const ArchesLabel* d_lab;
      const MPMArchesLabel* d_MAlab;

      ArchesVariables* d_scalarVars;
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
      int d_conv_scheme;
      int d_limiter_type;
      int d_boundary_limiter_type;
      bool d_central_limiter;
      bool d_dynScalarModel;
      double d_turbPrNo;
      bool d_doMMS;
#ifdef multimaterialform
      // set the values in problem setup
      MultiMaterialInterface* d_mmInterface;
      MultiMaterialSGSModel* d_mmSGSModel;
#endif
      string d_scalar_name;
      double d_scalar_init_value;
      bool d_scalar_diffusion;
      bool d_scalar_density_weighted;
      bool d_scalar_useforden;
      const VarLabel* d_scalar_label;
      const VarLabel* d_scalar_temp_label;
      const VarLabel* d_scalar_coef_label;
      const VarLabel* d_scalar_diff_coef_label;
      const VarLabel* d_scalar_nonlin_src_label;
      string d_precompTabSrcName;

      bool d_calcExtraScalarSrcs;
      ExtraScalarSrc* d_extraScalarSrc;
      vector<ExtraScalarSrc*> d_extraScalarSources;
      vector<string> d_tabSrcNames;
      bool d_carbon_balance;
      bool d_sulfur_balance; 
      double d_clipValue;
      bool d_noisyClipping;

}; // End class ExtraScalarSolver
} // End namespace Uintah


#endif

