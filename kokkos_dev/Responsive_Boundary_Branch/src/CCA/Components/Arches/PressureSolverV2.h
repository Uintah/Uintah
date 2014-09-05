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

class PressureSolver {

public:
  
  //______________________________________________________________________/
  // Construct an instance of the Pressure solver.  
  PressureSolver(ArchesLabel* label,
                 const MPMArchesLabel* MAlb,
                 BoundaryCondition* bndry_cond,
                 PhysicalConstants* phys_const,
                 const ProcessorGroup* myworld);

  //______________________________________________________________________/
  // Destructor
  ~PressureSolver();

  //______________________________________________________________________
  // Set up the problem specification database
  void problemSetup(const ProblemSpecP& params);


  //______________________________________________________________________
  //  Task that is called by Arches and constains scheduling of other tasks  
  void sched_solve(const LevelP& level, SchedulerP&,
                   const TimeIntegratorLabel* timelabels,
                   bool extraProjection,
                   bool d_EKTCorrection,
                   bool doing_EKT_now);



  void sched_pressureLinearSolve(const LevelP& level, SchedulerP& sched,
                                 const TimeIntegratorLabel* timelabels,
                                 bool extraProjection,
                                 bool d_EKTCorrection,
                                 bool doing_EKT_now);



  inline void setPressureCorrectionFlag(bool pressure_correction) {
    d_pressure_correction = pressure_correction;
  }
  inline void setMMS(bool doMMS) {
    d_doMMS=doMMS;
  }
protected:

private:
  //______________________________________________________________________/
  // Default : Construct an empty instance of the Pressure solver.
  PressureSolver(const ProcessorGroup* myworld);
  
  //______________________________________________________________________
  //  buildLinearMatrix:
  //  Compute the matrix coefficient and place them in either
  //  Hypre or Petsc data structures
  void sched_buildLinearMatrix(SchedulerP&, 
                               const PatchSet* patches,
                               const MaterialSet* matls,
                               const TimeIntegratorLabel* timelabels,
                               bool extraProjection,
                               bool d_EKTCorrection,
                               bool doing_EKT_now);

  void buildLinearMatrix(const ProcessorGroup* pc,
                         const PatchSubset* patches,
                         const MaterialSubset* matls,
                         DataWarehouse* new_dw,
                         DataWarehouse* matrix_dw,
                         const TimeIntegratorLabel* timelabels,
                         bool extraProjection,
                         bool d_EKTCorrection,
                         bool doing_EKT_now);

  //______________________________________________________________________
  //  setPressRHS:
  //  This is a wrapper task and passes the uintah data to either hypre or petsc
  //  which fills in the vector X and RHS
  void sched_setRHS_X_wrap(SchedulerP& sched,
                           const PatchSet* patches,
                           const MaterialSet* matls,
                           const TimeIntegratorLabel* timelabels,
                           bool extraProjection,
                           bool d_EKTCorrection,
                           bool doing_EKT_now);
                           
  void setRHS_X_wrap ( const ProcessorGroup*,
                       const PatchSubset* patches,
                       const MaterialSubset*,
                       DataWarehouse* ,
                       DataWarehouse* ,
                       const string& which_dw,
                       const TimeIntegratorLabel* timelabels );
                       
  //______________________________________________________________________
  // SolveSystem:
  // This task calls either Petsc or Hypre to solve the system                    
  void sched_SolveSystem(SchedulerP& sched,
                         const PatchSet* patches,
                         const MaterialSet* matls,
                         const TimeIntegratorLabel* timelabels,
                         bool extraProjection,
                         bool doing_EKT_now);
                         
  void solveSystem(const ProcessorGroup* pg,
                   const PatchSubset* patches,
                   const MaterialSubset*,
                   DataWarehouse* old_dw,
                   DataWarehouse* new_dw,
                   const TimeIntegratorLabel* timelabels);
                         
  //______________________________________________________________________
  //  Extract_X:
  //  This task places the solution to the into a uintah array and sets the value
  //  of the reference pressure.
  void schedExtract_X(SchedulerP& sched,
                      const PatchSet* patches,
                      const MaterialSet* matls,
                      const TimeIntegratorLabel* timelabels,
                      bool extraProjection,
                      bool doing_EKT_now);
                                 
  void  Extract_X ( const ProcessorGroup* pg,
                    const PatchSubset* patches,
                    const MaterialSubset* matls,
                    DataWarehouse* old_dw,
                    DataWarehouse* new_dw,
                    const string& compute_or_modify,
                    const VarLabel* varLabel );
                    
  //______________________________________________________________________
  //  normalizePress:
  //  Subtract off the reference pressure from pressure field                 
  void sched_normalizePress(SchedulerP& sched,
                            const PatchSet* patches,
                            const MaterialSet* matls);
                            
  void normalizePress ( const ProcessorGroup* pg,
                        const PatchSubset* patches,
                        const MaterialSubset* matls,
                        DataWarehouse*,
                        DataWarehouse* new_dw);
  //______________________________________________________________________
  // addHydrostaticTermtoPressure:
  // Add the hydrostatic term to the relative pressure
  void sched_addHydrostaticTermtoPressure(SchedulerP& sched,
                                          const PatchSet* patches,
                                          const MaterialSet* matls,
                                          const TimeIntegratorLabel* timelabels);

  void addHydrostaticTermtoPressure(const ProcessorGroup* pc,
                                    const PatchSubset* patches,
                                    const MaterialSubset* matls,
                                    DataWarehouse* old_dw,
                                    DataWarehouse* new_dw,
                                    const TimeIntegratorLabel* timelabels);
                                                                   
                           
  //__________________________________
  // Set stencil weights. (Pressure)
  // It uses second order hybrid differencing for computing
  // coefficients
  void calculatePressureCoeff(const Patch* patch,
                              CellInformation* cellinfo,
                              ArchesVariables* vars,
                              ArchesConstVariables* constvars); 

  //__________________________________
  // Modify stencil weights (Pressure) to account for voidage due
  // to multiple materials
  void mmModifyPressureCoeffs(const Patch* patch,
                              ArchesVariables* vars,
                              ArchesConstVariables* constvars);
                              
  //______________________________________________________________________
  //
 private:

  bool d_always_construct_A;
  bool d_construct_A; 
  bool d_construct_solver_obj; 

  ArchesLabel* d_lab;
  const MPMArchesLabel* d_MAlab;

  Discretization*     d_discretize;
  Source*             d_source;
  LinearSolver*       d_linearSolver;
  BoundaryCondition*  d_boundaryCondition;
  PhysicalConstants*  d_physicalConsts;

  // Maximum number of iterations to take before stopping/giving up.
  int d_maxIterations;
  
  int d_indx;         // Arches matl index.

  //reference point for the solvers
  IntVector d_pressRef;

  const ProcessorGroup* d_myworld;
  
#ifdef multimaterialform
  // set the values in problem setup
  MultiMaterialInterface* d_mmInterface;
  MultiMaterialSGSModel* d_mmSGSModel;
#endif
  
  bool d_pressure_correction;
  bool d_norm_press;
  bool d_doMMS;
  bool d_do_only_last_projection;
}; // End class PressureSolver

} // End namespace Uintah

#endif

