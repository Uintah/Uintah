#ifndef Uintah_Component_Arches_Ray_h
#define Uintah_Component_Arches_Ray_h

#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/BoundaryConditions/BCDataArray.h>
#include <Core/Grid/BoundaryConditions/BoundCond.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>

#include <iostream>
#include <cmath>
#include <string>
#include <vector>

//==========================================================================

/**
 * @class Ray
 * @author Isaac Hunsaker
 * @date July 8, 2011 
 *
 * @brief This file traces N (usually 1000+) rays per cell until the intensity reaches a predetermined threshold
 *
 *
 */

namespace Uintah{

  class Ray  {

    public: 

      Ray(); 
      ~Ray(); 

      //__________________________________
      //  TASKS
      /** @brief Interface to input file information */
      void  problemSetup( const ProblemSpecP& inputdb ); 

      /** @brief Algorithm for tracing rays through a single level*/ 
      void sched_rayTrace( const LevelP& level, 
                           SchedulerP& sched,
                           Task::WhichDW abskg_dw,
                           Task::WhichDW sigma_dw,
                           bool modifies_divQ );
                           
      /** @brief Algorithm for RMCRT using multilevel dataOnion approach*/ 
      void sched_rayTrace_dataOnion( const LevelP& level, 
                                     SchedulerP& sched,
                                     Task::WhichDW abskg_dw,
                                     Task::WhichDW sigma_dw,
                                     bool modifies_divQ  );

      /** @brief Schedule compute of blackbody intensity */ 
      void sched_sigmaT4( const LevelP& level, 
                          SchedulerP& sched,
                          Task::WhichDW temp_dw);

      /** @brief Initializes properties for the algorithm */ 
      void sched_initProperties( const LevelP&, 
                                 SchedulerP& sched, 
                                 const int time_sub_step );
                                 
      /** @brief Set boundary conditions and compute sigmaT4 */
      void  sched_setBoundaryConditions( const LevelP& level, 
                                         SchedulerP& sched );
      
      //__________________________________
      //  Helpers
      /** @brief map the component VarLabels to RMCRT VarLabels */
     void registerVarLabels(int   matl,
                            const VarLabel*  abskg,
                            const VarLabel* absorp,
                            const VarLabel* temperature,
                            const VarLabel* divQ);
                            
    void setBC(CCVariable<double>& Q_CC,
               const string& desc,
               const Patch* patch,          
               const int mat_id);

    private: 
      enum DIR {X, Y, Z, NONE};
      double _pi;
      double _alphaEW;//absorptivity of the East and West walls
      double _alphaNS;//absorptivity of the North and South walls
      double _alphaTB;//absorptivity of the top and bottom walls
      double _TEW;
      double _TNS;
      double _TTB;
      double _Threshold;
      double _sigma; 
      int    _NoOfRays;
      int    _slice;
      int    d_matl;
      MaterialSet* d_matlSet;
      
      double _sigma_over_pi; // Stefan Boltzmann divided by pi (W* m-2* K-4)

      int  _benchmark; 
      bool _isSeedRandom;
      bool _solveBoundaryFlux;
      bool _CCRays;
      bool _shouldSetBC;

      const VarLabel* d_sigmaT4_label; 
      const VarLabel* d_abskgLabel;
      const VarLabel* d_absorpLabel;
      const VarLabel* d_temperatureLabel;
      const VarLabel* d_divQLabel;

      //----------------------------------------
      void rayTrace( const ProcessorGroup* pc, 
                     const PatchSubset* patches, 
                     const MaterialSubset* matls, 
                     DataWarehouse* old_dw, 
                     DataWarehouse* new_dw,
                     bool modifies_divQ,
                     Task::WhichDW which_abskg_dw,
                     Task::WhichDW which_sigmaT4_dw ); 
      //__________________________________
      void rayTrace_dataOnion( const ProcessorGroup* pc, 
                               const PatchSubset* patches, 
                               const MaterialSubset* matls, 
                               DataWarehouse* old_dw, 
                               DataWarehouse* new_dw,
                     bool modifies_divQ,
                     Task::WhichDW which_abskg_dw,
                     Task::WhichDW which_sigmaT4_dw );
      
      //----------------------------------------
      void initProperties( const ProcessorGroup* pc, 
                           const PatchSubset* patches, 
                           const MaterialSubset* matls, 
                           DataWarehouse* old_dw, 
                           DataWarehouse* new_dw,
                           const int time_sub_step ); 

      //----------------------------------------
      void sigmaT4( const ProcessorGroup* pc,
                    const PatchSubset* patches,
                    const MaterialSubset* matls,
                    DataWarehouse* old_dw,
                    DataWarehouse* new_dw,
                    Task::WhichDW which_temp_dw );

      //__________________________________
      inline bool containsCell(const IntVector &low, 
                               const IntVector &high, 
                               const IntVector &cell,
                               const int &face);

    //______________________________________________________________________
    //   Boundary Conditions

      void setBoundaryConditions( const ProcessorGroup*,
                                  const PatchSubset* patches,           
                                  const MaterialSubset*,                
                                  DataWarehouse*,                
                                  DataWarehouse* new_dw );
                                  
    int numFaceCells(const Patch* patch,
                     const Patch::FaceIteratorType type,
                     const Patch::FaceType face);

  }; // class Ray
} // namespace Uintah

#endif
