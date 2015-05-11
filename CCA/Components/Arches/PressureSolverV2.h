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

//----- PressureSolver.h -----------------------------------------------

#ifndef Uintah_Components_Arches_PressureSolverV2_h
#define Uintah_Components_Arches_PressureSolverV2_h

#include <CCA/Ports/SchedulerP.h>
#include <CCA/Ports/SolverInterface.h>
#include <Core/Grid/SimulationStateP.h>
#include <CCA/Components/Arches/ArchesConstVariables.h>
#include <CCA/Components/Arches/ArchesVariables.h>
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
  void problemSetup(ProblemSpecP& params,SimulationStateP& state);


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
  // Set stencil weights
  void calculatePressureCoeff(const Patch* patch,
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
  /** @brief Adjusts the neighbors of the fix pressure point to account for the 
   * known pressure **/ 
  void adjustForRefPoint( const Patch* patch, ArchesVariables* vars, ArchesConstVariables* constvars );

  inline void fix_ref_coeffs(const Patch* patch, ArchesVariables* vars, ArchesConstVariables* constvars, IntVector c ){ 

    using std::cout; 
    using std::endl;

    if ( patch->containsCell( c ) ){ 

      IntVector E  = c + IntVector(1,0,0);   IntVector W  = c - IntVector(1,0,0); 
      IntVector N  = c + IntVector(0,1,0);   IntVector S  = c - IntVector(0,1,0);
      IntVector T  = c + IntVector(0,0,1);   IntVector B  = c - IntVector(0,0,1); 

      if ( constvars->cellType[c] != -1 ) {
        throw InvalidValue("Error: Your reference pressure point is not a flow cell.", __FILE__, __LINE__);
      }

      vars->pressCoeff[c].p = 1.0; 
      vars->pressCoeff[c].e = .0; 
      vars->pressCoeff[c].w = .0; 
      vars->pressCoeff[c].n = .0; 
      vars->pressCoeff[c].s = .0; 
      vars->pressCoeff[c].t = .0; 
      vars->pressCoeff[c].b = .0; 
      vars->pressNonlinearSrc[c] = d_ref_value; 

      if ( constvars->cellType[E] == -1 ){ 
        if ( patch->containsCell(E) ){ 
          vars->pressCoeff[E].p -= vars->pressCoeff[E].w;
          vars->pressCoeff[E].w = 0.0; 
          vars->pressNonlinearSrc[E] += d_ref_value;
        } else { 
          if ( d_periodic_vector[0] == 0 ){ 
            cout << " Reference neighbor = " << E << std::endl;
            throw InvalidValue("Error: (EAST DIRECTION) Reference point cannot be next to a patch boundary.", __FILE__, __LINE__);
          }
        } 
      } 
      if ( constvars->cellType[W] == -1 ){ 
        if ( patch->containsCell(W) ){  
          vars->pressCoeff[W].p -= vars->pressCoeff[W].e;
          vars->pressCoeff[W].e = 0.0; 
          vars->pressNonlinearSrc[W] += d_ref_value;
        } else { 
          if ( d_periodic_vector[0] == 0 ){ 
            cout << " Reference neighbor = " << W << endl;
            throw InvalidValue("Error: (WEST DIRECTION) Reference point cannot be next to a patch boundary.", __FILE__, __LINE__);
          }
        } 
      } 
      if ( constvars->cellType[N] == -1 ){ 
        if ( patch->containsCell(N) ){ 
          vars->pressCoeff[N].p -= vars->pressCoeff[N].s;
          vars->pressCoeff[N].s= 0.0; 
          vars->pressNonlinearSrc[N] += d_ref_value;
        } else { 
          if ( d_periodic_vector[1] == 0 ){ 
            cout << " Reference neighbor = " << N << endl;
            throw InvalidValue("Error: (NORTH DIRECTION) Reference point cannot be next to a patch boundary.", __FILE__, __LINE__);
          }
        } 
      } 
      if ( constvars->cellType[S] == -1 ){ 
        if ( patch->containsCell(S) ){ 
          vars->pressCoeff[S].p -= vars->pressCoeff[S].n;
          vars->pressCoeff[S].n = 0.0; 
          vars->pressNonlinearSrc[S] += d_ref_value;
        } else { 
          if ( d_periodic_vector[1] == 0 ){ 
            cout << " Reference neighbor = " << S << endl;
            throw InvalidValue("Error: (SOUTH DIRECTION) Reference point cannot be next to a patch boundary.", __FILE__, __LINE__);
          }
        } 
      } 
      if ( constvars->cellType[T] == -1 ){ 
        if ( patch->containsCell(T) ){
          vars->pressCoeff[T].p -= vars->pressCoeff[T].b;
          vars->pressCoeff[T].b= 0.0; 
          vars->pressNonlinearSrc[T] += d_ref_value;
        } else { 
          if ( d_periodic_vector[2] == 0 ){ 
            cout << " Reference neighbor = " << T << endl;
            throw InvalidValue("Error: (TOP DIRECTION) Reference point cannot be next to a patch boundary.", __FILE__, __LINE__);
          }
        } 
      } 
      if ( constvars->cellType[B] == -1 ){ 
        if ( patch->containsCell(B) ){ 
          vars->pressCoeff[B].p -= vars->pressCoeff[B].t;
          vars->pressCoeff[B].t = 0.0; 
          vars->pressNonlinearSrc[B] += d_ref_value;
        } else { 
          if ( d_periodic_vector[2] == 0 ){ 
            cout << " Reference neighbor = " << B << endl;
            throw InvalidValue("Error: (BOTTOM DIRECTION) Reference point cannot be next to a patch boundary.", __FILE__, __LINE__);
          }
        } 
      } 
    }
  }

  ArchesLabel* d_lab;
  const MPMArchesLabel* d_MAlab;
  IntVector d_periodic_vector; 

  Source*             d_source;
  BoundaryCondition*  d_boundaryCondition;
  PhysicalConstants*  d_physicalConsts;
  
  int d_indx;             // Arches matl index.
  int d_iteration;
  IntVector d_pressRef;   // cell index for reference pressure

  const ProcessorGroup* d_myworld;

  bool d_norm_press;
  bool d_do_only_last_projection;
  bool d_use_ref_point; 
  double d_ref_value; 
  std::vector<int> d_reduce_ref_dims; 
  int d_N_reduce_ref_dims; 
  
  SolverInterface* d_hypreSolver;
  SolverParameters* d_hypreSolver_parameters;

  std::vector<std::string> d_new_sources;
  
  
}; // End class PressureSolver

} // End namespace Uintah

#endif

