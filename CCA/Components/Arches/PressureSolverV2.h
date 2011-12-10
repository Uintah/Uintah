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

#ifndef Uintah_Components_Arches_PressureSolverV2_h
#define Uintah_Components_Arches_PressureSolverV2_h

#include <CCA/Ports/SchedulerP.h>
#include <CCA/Ports/SolverInterface.h>
#include <CCA/Components/Arches/ArchesConstVariables.h>
#include <CCA/Components/Arches/CellInformationP.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <CCA/Ports/DataWarehouseP.h>
#include <Core/Grid/LevelP.h>
#include <Core/Grid/Task.h>


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
  void problemSetup(ProblemSpecP& params);


  //______________________________________________________________________
  //  Task that is called by Arches and constains scheduling of other tasks  
  void sched_solve(const LevelP& level, SchedulerP&,
                   const TimeIntegratorLabel* timelabels,
                   bool extraProjection);

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
                         bool extraProjection);

  //______________________________________________________________________
  //  setRefPressure:
  //  This sets the value of the reference pressure.
  void sched_set_BC_RefPress(SchedulerP& sched,
                             const PatchSet* patches,
                             const MaterialSet* matls,
                             const TimeIntegratorLabel* timelabels,
                             bool extraProjection,
                             string& pressLabel);

  void  set_BC_RefPress ( const ProcessorGroup* pg,
                          const PatchSubset* patches,
                          const MaterialSubset* matls,
                          DataWarehouse* old_dw,
                          DataWarehouse* new_dw,
                          const VarLabel* pressLabel,
                          const VarLabel* refPressLabel,
                          const string integratorPhase );

  //______________________________________________________________________
  //  normalizePress:
  //  Subtract off the reference pressure from pressure field
  void sched_normalizePress(SchedulerP& sched,
                            const PatchSet* patches,
                            const MaterialSet* matls,
                            const string& pressLabel,
                            const TimeIntegratorLabel* timelabels);
                            
  void normalizePress ( const ProcessorGroup* pg,
                        const PatchSubset* patches,
                        const MaterialSubset* matls,
                        DataWarehouse*,
                        DataWarehouse* new_dw,
                        const VarLabel* pressLabel,
                        const VarLabel* refPressLabel);

  //__________________________________
  // Set stencil weights
  void calculatePressureCoeff(const Patch* patch,
                              CellInformation* cellinfo,
                              ArchesVariables* vars,
                              ArchesConstVariables* constvars); 

  //__________________________________
  // Modify stencil weights to account for voidage due
  // to multiple materials
  void mmModifyPressureCoeffs(const Patch* patch,
                              ArchesVariables* vars,
                              ArchesConstVariables* constvars);
  //______________________________________________________________________
  //
  ArchesLabel* d_lab;
  const MPMArchesLabel* d_MAlab;

  Source*             d_source;
  BoundaryCondition*  d_boundaryCondition;
  PhysicalConstants*  d_physicalConsts;
  
  int d_indx;             // Arches matl index.
  int d_iteration;
  IntVector d_pressRef;   // cell index for reference pressure

  const ProcessorGroup* d_myworld;

  bool d_norm_press;
  bool d_do_only_last_projection;
  
  SolverInterface* d_hypreSolver;
  SolverParameters* d_hypreSolver_parameters;

  vector<std::string> d_new_sources; 
  
  
}; // End class PressureSolver

} // End namespace Uintah

#endif

