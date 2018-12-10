/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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

//----- PressureSolver.h -----------------------------------------------

#ifndef Uintah_Components_Arches_PressureSolverV2_h
#define Uintah_Components_Arches_PressureSolverV2_h
#include <CCA/Components/Arches/linSolver.h>
#include <CCA/Ports/SchedulerP.h>
#include <CCA/Ports/SolverInterface.h>
#include <Core/Grid/MaterialManagerP.h>
#include <CCA/Components/Arches/ArchesConstVariables.h>
#include <CCA/Components/Arches/ArchesVariables.h>
#include <CCA/Components/Arches/CellInformationP.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <CCA/Ports/DataWarehouseP.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/Task.h>
#include <CCA/Components/Arches/linSolver.h>
namespace Uintah {

class MPMArchesLabel;
class ArchesLabel;
class ProcessorGroup;
class ArchesVariables;
class PhysicalConstants;
class Discretization;
class Source;
class BoundaryCondition;
class TimeIntegratorLabel;


class PressureSolver {

public:

  //______________________________________________________________________/
  // Construct an instance of the Pressure solver.
  PressureSolver(ArchesLabel* label,
                 const MPMArchesLabel* MAlb,
                 BoundaryCondition* bndry_cond,
                 PhysicalConstants* phys_const,
                 const ProcessorGroup* myworld,
                 SolverInterface* hypreSolver);

  //______________________________________________________________________/
  // Destructor
  ~PressureSolver();

  //______________________________________________________________________
  // Set up the problem specification database
  void problemSetup(ProblemSpecP& params,MaterialManagerP& materialManager);


  //______________________________________________________________________
  // used to create hypre objects.
  void scheduleInitialize( const LevelP& level, 
                           SchedulerP& sched, 
                           const MaterialSet* matls);
                           
  void scheduleRestartInitialize( const LevelP& level, 
                                  SchedulerP& sched, 
                                  const MaterialSet* matls);

  //______________________________________________________________________
  //  Task that is called by Arches and constains scheduling of other tasks
  void sched_solve( const LevelP& level, SchedulerP&,
                    const TimeIntegratorLabel* timelabels,
                    bool extraProjection,
                    const int rk_stage );

  //______________________________________________________________________
  // addHydrostaticTermtoPressure:
  // Add the hydrostatic term to the relative pressure    THIS BE PRIVATE -Todd
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

  inline std::vector<std::string> get_pressure_source_ref(){
    return d_new_sources;
  }

  //__________________________________
  // Set stencil weights
  void calculatePressureCoeff(const Patch* patch,
                              ArchesVariables* vars,
                              ArchesConstVariables* constvars);

private:

  //______________________________________________________________________
  //  buildLinearMatrix:
  //  Compute the matrix coefficients & RHS and place them in either
  //  Hypre or Petsc data structures
  void sched_buildLinearMatrix(SchedulerP&,
                               const PatchSet* patches,
                               const MaterialSet* matls,
                               const TimeIntegratorLabel* timelabels,
                               bool extraProjection);

  void buildLinearMatrix(const ProcessorGroup* pc,
                         const PatchSubset* patches,
                         const MaterialSubset* matls,
                         DataWarehouse* new_dw,
                         DataWarehouse* matrix_dw,
                         const PatchSet* patchSet,
                         const TimeIntegratorLabel* timelabels,
                         bool extraProjection);

  //______________________________________________________________________
  //  setGuessForX:
  //  This either sets the initial guess for X to 0.0 or moves the
  //  timelabel->pressureGuess to the new_dw.

  void sched_setGuessForX(SchedulerP& sched,
                          const PatchSet* patches,
                          const MaterialSet* matls,
                          const TimeIntegratorLabel* timelabels,
                          bool extraProjection);

  void setGuessForX ( const ProcessorGroup* pg,
                      const PatchSubset* patches,
                      const MaterialSubset* matls,
                      DataWarehouse* old_dw,
                      DataWarehouse* new_dw,
                      const TimeIntegratorLabel* timelabels,
                      const bool extraProjection);

  //______________________________________________________________________
  // SolveSystem:
  // This task calls UCF:Hypre to solve the system
  void sched_SolveSystem(SchedulerP& sched,
                         const PatchSet* patches,
                         const MaterialSet* matls,
                         const TimeIntegratorLabel* timelabels,
                         bool extraProjection,
                         const int rk_stage);

  //______________________________________________________________________
  //  setRefPressure:
  //  This sets the value of the reference pressure.
  void sched_set_BC_RefPress(SchedulerP& sched,
                             const PatchSet* patches,
                             const MaterialSet* matls,
                             const TimeIntegratorLabel* timelabels,
                             bool extraProjection,
                             std::string& pressLabel);

  void  set_BC_RefPress ( const ProcessorGroup* pg,
                          const PatchSubset* patches,
                          const MaterialSubset* matls,
                          DataWarehouse* old_dw,
                          DataWarehouse* new_dw,
                          const VarLabel* pressLabel,
                          const VarLabel* refPressLabel,
                          const std::string integratorPhase );

  //______________________________________________________________________
  //  normalizePress:
  //  Subtract off the reference pressure from pressure field
  void sched_normalizePress(SchedulerP& sched,
                            const PatchSet* patches,
                            const MaterialSet* matls,
                            const std::string& pressLabel,
                            const TimeIntegratorLabel* timelabels);

  void normalizePress ( const ProcessorGroup* pg,
                        const PatchSubset* patches,
                        const MaterialSubset* matls,
                        DataWarehouse*,
                        DataWarehouse* new_dw,
                        const VarLabel* pressLabel,
                        const VarLabel* refPressLabel);

  //__________________________________
  // Modify stencil weights to account for voidage due
  // to multiple materials
  void mmModifyPressureCoeffs(const Patch* patch,
                              ArchesVariables* vars,
                              ArchesConstVariables* constvars);

  ArchesLabel* d_lab;
  const MPMArchesLabel* d_MAlab;
  IntVector d_periodic_vector;

  Source*             d_source;
  BoundaryCondition*  d_boundaryCondition;
  PhysicalConstants*  d_physicalConsts;

  int d_indx;             // Arches matl index.
  int d_iteration;
  int nExtraSources;
  IntVector d_pressRef;   // cell index for reference pressure

  const ProcessorGroup* d_myworld;
  double d_ref_value;

  bool d_norm_press;
  bool d_do_only_last_projection;
  bool d_enforceSolvability;
  bool do_custom_arches_linear_solve{false};
  SolverInterface* d_hypreSolver;

  linSolver* custom_solver;

  std::vector<std::string> d_new_sources;
  std::map<std::string, double> d_source_weights;
  std::vector<const VarLabel *> extraSourceLabels;

}; // End class PressureSolver

} // End namespace Uintah

#endif
