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

#ifndef RAY_H
#define RAY_H

#include <CCA/Components/Models/Radiation/RMCRT/RMCRTCommon.h>
#include <CCA/Components/Models/Radiation/RMCRT/Radiometer.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Containers/StaticArray.h>
#include <Core/Disclosure/TypeDescription.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/BoundaryConditions/BCDataArray.h>
#include <Core/Grid/BoundaryConditions/BoundCond.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Variables/CCVariable.h>

#include <sci_defs/uintah_defs.h>
#include <sci_defs/cuda_defs.h>

#ifdef HAVE_CUDA
  #include <curand.h>
  #include <curand_kernel.h>
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
class MTRand; //forward declaration for use in updateSumI

namespace Uintah{

  class Ray : public RMCRTCommon  {

    public:

      Ray( TypeDescription::Type FLT_DBL );         // This class can  Float or Double
      ~Ray();

      //__________________________________
      //  TASKS
      /** @brief Interface to input file information */
      void  problemSetup( const ProblemSpecP& prob_spec,
                          const ProblemSpecP& rmcrt_ps,
                          const GridP& grid,
                          SimulationStateP& sharedState );

      /** @brief Algorithm for tracing rays through a single level*/
      void sched_rayTrace( const LevelP& level,
                           SchedulerP& sched,
                           Task::WhichDW abskg_dw,
                           Task::WhichDW sigma_dw,
                           Task::WhichDW celltype_dw,
                           bool modifies_divQ,
                           const int radCalc_freq );

      /** @brief Algorithm for RMCRT using multilevel dataOnion approach*/
      void sched_rayTrace_dataOnion( const LevelP& level,
                                     SchedulerP& sched,
                                     Task::WhichDW abskg_dw,
                                     Task::WhichDW sigma_dw,
                                     bool modifies_divQ,
                                     const int radCalc_freq );


      /** @brief Schedule filtering of q and divQ */
      void sched_filter( const LevelP& level,
                          SchedulerP& sched,
                          Task::WhichDW which_divQ_dw,
                          const bool includeEC = true,
                          bool modifies_divQFilt = false);

      //__________________________________
      //  Boundary condition related
      /** @brief Set boundary conditions */
      
      void setBC_onOff( const bool onOff){
        d_onOff_SetBCs = onOff;
      }
      
      void  sched_setBoundaryConditions( const LevelP& level,
                                         SchedulerP& sched,
                                         Task::WhichDW temp_dw,
                                         const int radCalc_freq,
                                         const bool backoutTemp = false);

      void BC_bulletproofing( const ProblemSpecP& rmcrtps );
                               

      template< class T, class V >
      void setBC(CCVariable<T>& Q_CC,
                 const std::string& desc,
                 const Patch* patch,
                 const int mat_id);

      //__________________________________
      //  Multilevel tasks
      void sched_Refine_Q(SchedulerP& sched,
                          const PatchSet* patches,
                          const MaterialSet* matls,
                          const int radCalc_freq);

      void sched_CoarsenAll( const LevelP& coarseLevel,
                             SchedulerP& sched,
                             const bool modifies_abskg,
                             const bool modifiesd_sigmaT4,
                             const int radCalc_freq );


      void sched_ROI_Extents ( const LevelP& level,
                               SchedulerP& scheduler );
                               
      Radiometer* getRadiometer(){
        return d_radiometer;
      }
                               


    //______________________________________________________________________
    private:
      double d_sigmaT4_thld;                 // threshold values for determining the extents of ROI
      double d_abskg_thld;
      int    d_nDivQRays;                    // number of rays per cell used to compute divQ
      int    d_nFluxRays;                    // number of rays per cell used to compute radiative flux
      int    d_orderOfInterpolation;         // Order of interpolation for interior fine patch
      IntVector d_halo;                      // number of cells surrounding a coarse patch on coarser levels

      bool d_solveBoundaryFlux;
      bool d_solveDivQ;
      bool d_CCRays;
      bool d_onOff_SetBCs;                // switch for setting boundary conditions
      bool d_isDbgOn;
      bool d_applyFilter;                 // Allow for filtering of boundFlux and divQ results

      enum ROI_algo{fixed, dynamic, patch_based};
      ROI_algo  d_whichROI_algo;
      Point d_ROI_minPt;
      Point d_ROI_maxPt;

      // Radiometer parameters
      Radiometer* d_radiometer;

      // Boundary flux constant variables  (consider using array container when C++ 11 is used)
      std::map <int,IntVector> d_dirIndexOrder;
      std::map <int,IntVector> d_dirSignSwap;
      std::map <int,IntVector> d_locationIndexOrder;
      std::map <int,IntVector> d_locationShift;

      const VarLabel* d_divQFiltLabel;
      const VarLabel* d_boundFluxLabel;
      const VarLabel* d_boundFluxFiltLabel;
      const VarLabel* d_radiationVolqLabel;
      const VarLabel* d_mag_grad_abskgLabel;
      const VarLabel* d_mag_grad_sigmaT4Label;
      const VarLabel* d_flaggedCellsLabel;
      const VarLabel* d_ROI_LoCellLabel;
      const VarLabel* d_ROI_HiCellLabel;

      //__________________________________
      template<class T>
      void rayTrace( const ProcessorGroup* pc,
                     const PatchSubset* patches,
                     const MaterialSubset* matls,
                     DataWarehouse* old_dw,
                     DataWarehouse* new_dw,
                     bool modifies_divQ,
                     Task::WhichDW which_abskg_dw,
                     Task::WhichDW whichd_sigmaT4_dw,
                     Task::WhichDW which_celltype_dw,
                     const int radCalc_freq );

      //__________________________________
      template<class T>
      void rayTraceGPU( Task::CallBackEvent event,
                        const ProcessorGroup* pg,
                        const PatchSubset* patches,
                        const MaterialSubset* matls,
                        DataWarehouse* old_dw,
                        DataWarehouse* new_dw,
                        void* stream,
                        int deviceID,
                        bool modifies_divQ,
                        Task::WhichDW which_abskg_dw,
                        Task::WhichDW whichd_sigmaT4_dw,
                        Task::WhichDW which_celltype_dw,
                        const int radCalc_freq);

      //__________________________________
      template<class T>
      void rayTrace_dataOnion( const ProcessorGroup* pc,
                               const PatchSubset* patches,
                               const MaterialSubset* matls,
                               DataWarehouse* old_dw,
                               DataWarehouse* new_dw,
                               bool modifies_divQ,
                               Task::WhichDW which_abskg_dw,
                               Task::WhichDW whichd_sigmaT4_dw,
                               const int radCalc_freq );
      //__________________________________
      template<class T>
      void  updateSumI_ML ( Vector& ray_direction,
                            Vector& ray_location,
                            const IntVector& origin,
                            const std::vector<Vector>& Dx,
                            const BBox& domain_BB,
                            const int maxLevels,
                            const Level* fineLevel,
                            double DyDx[],
                            double DzDx[],
                            const IntVector& fineLevel_ROI_Lo,
                            const IntVector& fineLevel_ROI_Hi,
                            std::vector<IntVector>& regionLo,
                            std::vector<IntVector>& regionHi,
                            StaticArray< constCCVariable< T > >& sigmaT4Pi,
                            StaticArray< constCCVariable< T > >& abskg,
                            unsigned long int& size,
                            double& sumI,
                            MTRand& mTwister);
     //__________________________________
     void computeExtents(LevelP level_0,
                        const Level* fineLevel,
                        const Patch* patch,
                        const int maxlevels,
                        DataWarehouse* new_dw,
                        IntVector& fineLevel_ROI_Lo,
                        IntVector& fineLevel_ROI_Hi,
                        std::vector<IntVector>& regionLo,
                        std::vector<IntVector>& regionHi);

      //__________________________________
      void filter( const ProcessorGroup* pc,
                    const PatchSubset* patches,
                    const MaterialSubset* matls,
                    DataWarehouse* old_dw,
                    DataWarehouse* new_dw,
                    Task::WhichDW which_divQ_dw,
                    const bool includeEC,
                    bool modifies_divQFilt);

      //__________________________________
      inline bool containsCell(const IntVector &low,
                               const IntVector &high,
                               const IntVector &cell,
                               const int &dir);

      //__________________________________
      /** @brief Adjust the location of a ray origin depending on the cell face */
      void rayLocation_cellFace( MTRand& mTwister,
                                 const IntVector& origin,
                                 const IntVector &indexOrder,
                                 const IntVector &shift,
                                 const double &DyDx,
                                 const double &DzDx,
                                 Vector& location );

      //__________________________________
      /** @brief Adjust the direction of a ray depending on the cell face */
      void rayDirection_cellFace( MTRand& mTwister,
                                  const IntVector& origin,
                                  const IntVector& indexOrder,
                                  const IntVector& signOrder,
                                  const int iRay,
                                  Vector& directionVector,
                                  double& cosTheta );

      /** @brief Determine if a flow cell is adjacent to a wall, and therefore has a boundary */
      bool has_a_boundary(const IntVector &c,
                          constCCVariable<int> &celltype,
                          std::vector<int> &boundaryFaces);
      //______________________________________________________________________
      //   Boundary Conditions
      template< class T >
      void setBoundaryConditions( const ProcessorGroup*,
                                  const PatchSubset* patches,
                                  const MaterialSubset*,
                                  DataWarehouse*,
                                  DataWarehouse* new_dw,
                                  Task::WhichDW temp_dw,
                                  const int radCalc_freq,
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
                  DataWarehouse* new_dw,
                  const int radCalc_freq);

    // coarsen a single variable
    void sched_Coarsen_Q( const LevelP& coarseLevel,
                          SchedulerP& scheduler,
                          Task::WhichDW this_dw,
                          const bool modifies,
                          const VarLabel* variable,
                          const int radCalc_freq);
    
    template< class T >
    void coarsen_Q ( const ProcessorGroup*,
                     const PatchSubset* patches,
                     const MaterialSubset* matls,
                     DataWarehouse*,
                     DataWarehouse* new_dw,
                     const VarLabel* variable,
                     const bool modifies,
                     Task::WhichDW this_dw,
                     const int radCalc_freq);
                     
    void coarsen_cellType( const ProcessorGroup*,
                           const PatchSubset* patches,       
                           const MaterialSubset*,      
                           DataWarehouse*,            
                           DataWarehouse* new_dw,            
                           const int radCalc_freq );
                     
    template< class T >
    void ROI_Extents ( const ProcessorGroup*,
                       const PatchSubset* patches,
                       const MaterialSubset* matls,
                       DataWarehouse*,
                       DataWarehouse* new_dw);

    //______________________________________________________________________
    //  Helpers
    bool less_Eq(    const IntVector& a, const IntVector& b ){
      return ( a.x() <= b.x() && a.y() <= b.y() && a.z() <= b.z() );
    }
    bool greater_Eq( const IntVector& a, const IntVector& b ){
      return ( a.x() >= b.x() && a.y() >= b.y() && a.z() >= b.z() );
    }

  }; // class Ray

} // namespace Uintah

#endif
