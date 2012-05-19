/*

 The MIT License

 Copyright (c) 1997-2012 Center for the Simulation of Accidental Fires and
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


#ifndef Uintah_Component_Arches_Ray_h
#define Uintah_Component_Arches_Ray_h

#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/BoundaryConditions/BCDataArray.h>
#include <Core/Grid/BoundaryConditions/BoundCond.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>

#include <sci_defs/cuda_defs.h>
#ifdef HAVE_CUDA
#include <CCA/Components/Schedulers/GPUThreadedMPIScheduler.h>
#endif

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
#ifdef HAVE_CUDA
      Ray(GPUThreadedMPIScheduler* scheduler);
#endif
      ~Ray(); 

      //__________________________________
      //  TASKS
      /** @brief Interface to input file information */
      void  problemSetup( const ProblemSpecP& prob_spec,
                          const ProblemSpecP& rmcrt_ps ); 

      /** @brief Algorithm for tracing rays through a single level*/ 
      void sched_rayTrace( const LevelP& level, 
                           SchedulerP& sched,
                           Task::WhichDW abskg_dw,
                           Task::WhichDW sigma_dw,
                           Task::WhichDW celltype_dw,
                           bool modifies_divQ );

      /** @brief Algorithm for RMCRT using multilevel dataOnion approach*/ 
      void sched_rayTrace_dataOnion( const LevelP& level, 
                                     SchedulerP& sched,
                                     Task::WhichDW abskg_dw,
                                     Task::WhichDW sigma_dw,
                                     bool modifies_divQ );

      /** @brief Schedule compute of blackbody intensity */ 
      void sched_sigmaT4( const LevelP& level, 
                          SchedulerP& sched,
                          Task::WhichDW temp_dw,
                          const bool includeEC = true);

      /** @brief Initializes properties for the algorithm */ 
      void sched_initProperties( const LevelP&, 
                                 SchedulerP& sched, 
                                 const int time_sub_step );
                                 
      /** @brief Set boundary conditions and compute sigmaT4 */
      void  sched_setBoundaryConditions( const LevelP& level, 
                                         SchedulerP& sched,
                                         Task::WhichDW temp_dw,
                                         const bool backoutTemp = false);
                                         
      /** @brief Update the running total of the incident intensity */
      void  updateSumI ( const Vector& inv_direction_vector,
                         const Vector& ray_location,
                         const IntVector& origin,
                         const Vector& Dx,
                         const IntVector& domainLo,
                         const IntVector& domainHi,
                         constCCVariable<double>& sigmaT4Pi,
                         constCCVariable<double>& abskg,
                         unsigned long int& size,
                         double& sumI);

      /** @brief Adjust the location of a ray origin depending on the cell face */
      void adjustLocation(Vector &location,
                          const IntVector &indexOrder,         
                          const IntVector &shift,              
                          const double &DyDxRatio,             
                          const double &DzDxRatio);            

      /** @brief Adjust the direction of a ray depending on the cell face */
      void adjustDirection(Vector &directionVector,
                           const IntVector &indexOrder,         
                           const IntVector &signOrder);         




      //__________________________________
      //  Multilevel tasks
      void sched_Refine_Q(SchedulerP& sched,
                          const PatchSet* patches,
                          const MaterialSet* matls);
                      
      void sched_CoarsenAll( const LevelP& coarseLevel, 
                             SchedulerP& sched );
                             

      void sched_ROI_Extents ( const LevelP& level, 
                                 SchedulerP& scheduler );
                               
      //__________________________________
      //  Helpers
      /** @brief map the component VarLabels to RMCRT VarLabels */
     void registerVarLabels(int   matl,
                            const VarLabel*  abskg,
                            const VarLabel* absorp,
                            const VarLabel* temperature,
                            const VarLabel* celltype, 
                            const VarLabel* divQ);
                            
    void setBC(CCVariable<double>& Q_CC,
               const string& desc,
               const Patch* patch,          
               const int mat_id);

    private: 
      enum DIR {X, Y, Z, NONE};
      double _pi;
      double _Threshold;
      double _sigma;
      double _sigmaT4_thld;                  // threshold values for determining the extents of ROI
      double _abskg_thld;
      
       
      int    _NoOfRays;
      int    _NoRadRays;
      int    d_matl;
      int    d_orderOfInterpolation;         // Order of interpolation for interior fine patch
      
      MaterialSet* d_matlSet;
      IntVector _halo;                       // number of cells surrounding a coarse patch on coarser levels
      
      double _sigma_over_pi;                // Stefan Boltzmann divided by pi (W* m-2* K-4)

      int  _benchmark; 
      bool _isSeedRandom;
      bool _solveBoundaryFlux;
      bool _CCRays;
      bool _shouldSetBC;
      bool _isDbgOn;
      
      enum ROI_algo{fixed, dynamic, patch_based};
      ROI_algo  _whichROI_algo;
      Point _ROI_minPt;
      Point _ROI_maxPt;

      // Virtual Radiometer parameters
      bool _virtRad;
      double _viewAng;
      Vector _orient;
      IntVector _VRLocationsMin;        // These should be physical points in the domain   --Todd
      IntVector _VRLocationsMax;        // What happens if the resolution changes
      
      Ghost::GhostType d_gn;
      Ghost::GhostType d_gac;

      const VarLabel* d_sigmaT4_label; 
      const VarLabel* d_abskgLabel;
      const VarLabel* d_absorpLabel;
      const VarLabel* d_temperatureLabel;
      const VarLabel* d_cellTypeLabel; 
      const VarLabel* d_divQLabel;
      const VarLabel* d_VRFluxLabel;
      const VarLabel* d_boundFluxLabel;
      const VarLabel* d_mag_grad_abskgLabel;
      const VarLabel* d_mag_grad_sigmaT4Label;
      const VarLabel* d_flaggedCellsLabel;
      const VarLabel* d_ROI_LoCellLabel;
      const VarLabel* d_ROI_HiCellLabel;

      //__________________________________
      //  
      void constructor();
      
      //----------------------------------------
      void rayTrace( const ProcessorGroup* pc, 
                     const PatchSubset* patches, 
                     const MaterialSubset* matls, 
                     DataWarehouse* old_dw, 
                     DataWarehouse* new_dw,
                     bool modifies_divQ,
                     Task::WhichDW which_abskg_dw,
                     Task::WhichDW which_sigmaT4_dw,
                     Task::WhichDW which_celltype_dw);

#ifdef HAVE_CUDA

      GPUThreadedMPIScheduler* _gpuScheduler;

      //______________________________________________________________________
      //
      void rayTraceGPU( const ProcessorGroup* pg,
                        const PatchSubset* patches,
                        const MaterialSubset* matls,
                        DataWarehouse* old_dw,
                        DataWarehouse* new_dw,
                        int device,
                        bool modifies_divQ,
                        Task::WhichDW which_abskg_dw,
                        Task::WhichDW which_sigmaT4_dw );

#endif

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
                    Task::WhichDW which_temp_dw,
                    const bool includeEC );

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
                                  DataWarehouse* new_dw,
                                  Task::WhichDW temp_dw,
                                  const bool backoutTemp );
                                  
    int numFaceCells(const Patch* patch,
                     const Patch::FaceIteratorType type,
                     const Patch::FaceType face);

    //_____________________________________________________________________
    //    Multiple Level tasks
    void refine_Q(const ProcessorGroup*,
                  const PatchSubset* patches,
                  const MaterialSubset* matls,
                  DataWarehouse*,
                  DataWarehouse* new_dw);
                          
    // coarsen a single variable
    void sched_Coarsen_Q( const LevelP& coarseLevel,
                          SchedulerP& scheduler,
                          Task::WhichDW this_dw,
                          const bool modifies,
                          const VarLabel* variable);
                  
    void coarsen_Q ( const ProcessorGroup*,
                     const PatchSubset* patches,
                     const MaterialSubset* matls,
                     DataWarehouse*, 
                     DataWarehouse* new_dw,
                     const VarLabel* variable,
                     const bool modifies,
                     Task::WhichDW this_dw);
                     
    void ROI_Extents ( const ProcessorGroup*,
                       const PatchSubset* patches,
                       const MaterialSubset* matls,
                       DataWarehouse*,
                       DataWarehouse* new_dw);

  }; // class Ray
} // namespace Uintah

#endif
