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

#ifndef CCA_COMPONENTS_MODELS_RADIATION_RMCRT_RAY_H
#define CCA_COMPONENTS_MODELS_RADIATION_RMCRT_RAY_H

#include <CCA/Ports/Scheduler.h>

#include <CCA/Components/Models/Radiation/RMCRT/RMCRTCommon.h>
#include <CCA/Components/Models/Radiation/RMCRT/Radiometer.h>

#include <Core/Disclosure/TypeDescription.h>
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
 * @brief This file traces N rays per cell until the intensity reaches a predetermined threshold
 *
 *
 */
class MTRand; // forward declaration for use in updateSumI

class ApplicationInterface;

namespace Uintah{

  class Ray : public RMCRTCommon  {

    public:

      Ray( TypeDescription::Type FLT_DBL );         // This class can  Float or Double
      ~Ray();


      //__________________________________
      //  public variables
      bool d_solveBoundaryFlux{false};

      enum modifiesComputes{ modifiesVar,
                             computesVar};

      //__________________________________
      //  TASKS
      /** @brief Interface to input file information */
      void  problemSetup( const ProblemSpecP& prob_spec,
                          const ProblemSpecP& rmcrt_ps,
                          const GridP& grid );

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
                                     Task::WhichDW celltype_dw,
                                     bool modifies_divQ );


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
                                         const bool backoutTemp = false);

      void BC_bulletproofing( const ProblemSpecP& rmcrtps );


      template< class T, class V >
      void setBC(CCVariable<T>& Q_CC,
                 const std::string& desc,
                 const Patch* patch,
                 const int mat_id);

      //__________________________________
      //  Multilevel tasks
      void sched_Refine_Q( SchedulerP& sched,
                           const PatchSet* patches,
                           const MaterialSet* matls );

      void sched_CoarsenAll( const LevelP& coarseLevel,
                             SchedulerP& sched,
                             const bool modifies_abskg,
                             const bool modifiesd_sigmaT4 );

      void sched_computeCellType ( const LevelP& coarseLevel,
                                   SchedulerP& sched,
                                   const Ray::modifiesComputes which );

      void sched_ROI_Extents ( const LevelP& level,
                               SchedulerP& scheduler );

      Radiometer* getRadiometer(){
        return d_radiometer;
      }

    //__________________________________
    //  public variables
    bool d_coarsenExtraCells{false};               // instead of setting BC on the coarse level, coarsen fine level extra cells

    //______________________________________________________________________
    private:

      double d_sigmaT4_thld{DBL_MAX};             // threshold values for determining the extents of ROI
      double d_abskg_thld{DBL_MAX};
      int    d_nDivQRays{10};                     // number of rays per cell used to compute divQ
      int    d_nFluxRays{1};                      // number of rays per cell used to compute radiative flux
      int    d_orderOfInterpolation{-9};          // Order of interpolation for interior fine patch
      IntVector d_haloCells{IntVector(-9,-9,-9)}; // Number of cells a ray will traverse after it exceeds a fine patch boundary before
                                                  // it moves to a coarser level
      double  d_haloLength{-9};                   // Physical length a ray will traverse after it exceeds a fine patch boundary before
                                                  // it moves to a coarser level. 

      std::vector <double>  _maxLengthFlux;
      std::vector <double>  _maxLength;
      std::vector< int >    _maxCells;

      bool d_solveDivQ{true};                     // switch for enabling computation of divQ
      bool d_CCRays{false};
      bool d_onOff_SetBCs{true};                  // switch for setting boundary conditions
      bool d_isDbgOn{false};
      bool d_applyFilter{false};                  // Allow for filtering of boundFlux and divQ results
      int  d_rayDirSampleAlgo{NAIVE};             // Ray sampling algorithm

      enum rayDirSampleAlgorithm{ NAIVE,          // random sampled ray direction
                                  LATIN_HYPER_CUBE
                                };

      enum Algorithm{ dataOnion,
                      coarseLevel,
                      singleLevel
                    };

      enum ROI_algo{  fixed,                // user specifies fixed low and high point for a bounding box
                      dynamic,              // user specifies thresholds that are used to dynamically determine ROI
                      patch_based,          // The patch extents + halo are the ROI
                      boundedRayLength,     // the patch extents + boundedRayLength/Dx are the ROI
                      entireDomain          // The ROI is the entire computatonal Domain
                    };

      int d_cellTypeCoarsenLogic{ROUNDUP};           // how to coarsen a cell type

      enum cellTypeCoarsenLogic{ ROUNDUP, ROUNDDOWN};

      ROI_algo  d_ROI_algo{entireDomain};
      Point d_ROI_minPt;
      Point d_ROI_maxPt;

      // Radiometer parameters
      Radiometer* d_radiometer{nullptr};

      // Boundary flux constant variables  (consider using array container when C++ 11 is used)
      std::map <int,IntVector> d_dirIndexOrder;
      std::map <int,IntVector> d_dirSignSwap;

      const VarLabel* m_timeStepLabel {nullptr};
      
      const VarLabel* d_mag_grad_abskgLabel;
      const VarLabel* d_mag_grad_sigmaT4Label;
      const VarLabel* d_flaggedCellsLabel;
      const VarLabel* d_ROI_LoCellLabel;
      const VarLabel* d_ROI_HiCellLabel;
      const VarLabel* d_PPTimerLabel;        // perPatch timer

      ApplicationInterface* m_application{nullptr};
    
      // const VarLabel* d_divQFiltLabel;
      // const VarLabel* d_boundFluxFiltLabel;

      //__________________________________
      template<class T>
      void rayTrace( const ProcessorGroup* pg,
                     const PatchSubset* patches,
                     const MaterialSubset* matls,
                     DataWarehouse* old_dw,
                     DataWarehouse* new_dw,
                     bool modifies_divQ,
                     Task::WhichDW which_abskg_dw,
                     Task::WhichDW whichd_sigmaT4_dw,
                     Task::WhichDW which_celltype_dw );

      //__________________________________
      template<class T>
      void rayTraceGPU( DetailedTask* dtask,
                        Task::CallBackEvent event,
                        const ProcessorGroup* pg,
                        const PatchSubset* patches,
                        const MaterialSubset* matls,
                        DataWarehouse* old_dw,
                        DataWarehouse* new_dw,
                        void* oldTaskGpuDW,
                        void* newTaskGpuDW,
                        void* stream,
                        int deviceID,
                        bool modifies_divQ,
                        int timeStep,
                        Task::WhichDW which_abskg_dw,
                        Task::WhichDW whichd_sigmaT4_dw,
                        Task::WhichDW which_celltype_dw);

      //__________________________________
      template<class T>
      void rayTrace_dataOnion( const ProcessorGroup* pg,
                               const PatchSubset* patches,
                               const MaterialSubset* matls,
                               DataWarehouse* old_dw,
                               DataWarehouse* new_dw,
                               bool modifies_divQ,
                               Task::WhichDW which_abskg_dw,
                               Task::WhichDW whichd_sigmaT4_dw,
                               Task::WhichDW which_celltype_dw );

      //__________________________________
      template<class T>
      void rayTraceDataOnionGPU( DetailedTask* dtask,
                                 Task::CallBackEvent event,
                                 const ProcessorGroup* pg,
                                 const PatchSubset* patches,
                                 const MaterialSubset* matls,
                                 DataWarehouse* old_dw,
                                 DataWarehouse* new_dw,
                                 void* oldTaskGpuDW,
                                 void* newTaskGpuDW,
                                 void* stream,
                                 int deviceID,
                                 bool modifies_divQ,
                                 int timeStep,
                                 Task::WhichDW which_abskg_dw,
                                 Task::WhichDW whichd_sigmaT4_dw,
                                 Task::WhichDW which_celltype_dw );
      //__________________________________
      template<class T>
      void updateSumI_ML ( Vector& ray_direction,
                           Vector& ray_origin,
                           const IntVector& origin,
                           const std::vector<Vector>& Dx,
                           const BBox& domain_BB,
                           const int maxLevels,
                           const Level* fineLevel,
                           const IntVector& fineLevel_ROI_Lo,
                           const IntVector& fineLevel_ROI_Hi,
                           std::vector<IntVector>& regionLo,
                           std::vector<IntVector>& regionHi,
                           std::vector< constCCVariable< T > >& sigmaT4Pi,
                           std::vector< constCCVariable< T > >& abskg,
                           std::vector< constCCVariable< int > >& cellType,
                           unsigned long int& size,
                           double& sumI,
                           MTRand& mTwister);

     //__________________________________
     void computeExtents( LevelP level_0,
                          const Level* fineLevel,
                          const Patch* patch,
                          const int maxlevels,
                          DataWarehouse* new_dw,
                          IntVector& fineLevel_ROI_Lo,
                          IntVector& fineLevel_ROI_Hi,
                          std::vector<IntVector>& regionLo,
                          std::vector<IntVector>& regionHi );

      //__________________________________
      void filter( const ProcessorGroup* pg,
                    const PatchSubset* patches,
                    const MaterialSubset* matls,
                    DataWarehouse* old_dw,
                    DataWarehouse* new_dw,
                    Task::WhichDW which_divQ_dw,
                    const bool includeEC,
                    bool modifies_divQFilt);

      //__________________________________
      inline bool containsCell( const IntVector &low,
                                const IntVector &high,
                                const IntVector &cell,
                                const int &dir );

      //__________________________________
      /** @brief Adjust the location of a ray origin depending on the cell face */
      void rayLocation_cellFace( MTRand& mTwister,
                                 const int face,
                                 const Vector Dx,
                                 const Point CC_pos,
                                 Vector& location);

      //__________________________________
      /** @brief Adjust the direction of a ray depending on the cell face */
      void rayDirection_cellFace( MTRand& mTwister,
                                  const IntVector& origin,
                                  const IntVector& indexOrder,
                                  const IntVector& signOrder,
                                  const int iRay,
                                  Vector& directionVector,
                                  double& cosTheta );

      //__________________________________
      /** @brief Sample Rays for directional flux using LHC sampling */
      void rayDirectionHyperCube_cellFace( MTRand& mTwister,
                                           const IntVector& origin,
                                           const IntVector& indexOrder,
                                           const IntVector& signOrder,
                                           const int iRay,
                                           Vector& directionVector,
                                           double& cosTheta,
                                           const int ibin,
                                           const int jbin);
      //__________________________________
      /** @brief Sample Rays for flux divergence using LHC sampling */
      Vector findRayDirectionHyperCube( MTRand& mTwister,
                                        const IntVector& = IntVector(-9,-9,-9),
                                        const int iRay = -9,
                                        const int bin_i = 0,
                                        const int bin_j = 0);


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
                                  const bool backoutTemp );

      //__________________________________
      int numFaceCells( const Patch* patch,
                        const Patch::FaceIteratorType type,
                        const Patch::FaceType face );

    //_____________________________________________________________________
    //    Multiple Level tasks
    void refine_Q( const ProcessorGroup*,
                   const PatchSubset* patches,
                   const MaterialSubset* matls,
                   DataWarehouse*,
                   DataWarehouse* new_dw );

    // coarsen a single variable
    void sched_Coarsen_Q( const LevelP& coarseLevel,
                          SchedulerP& scheduler,
                          Task::WhichDW this_dw,
                          const bool modifies,
                          const VarLabel* variable );

    template< class T >
    void coarsen_Q ( const ProcessorGroup*,
                     const PatchSubset* patches,
                     const MaterialSubset* matls,
                     DataWarehouse*,
                     DataWarehouse* new_dw,
                     const VarLabel* variable,
                     const bool modifies,
                     Task::WhichDW this_dw );

    void computeCellType( const ProcessorGroup*,
                          const PatchSubset* patches,
                          const MaterialSubset* matls,
                          DataWarehouse* old_dw,
                          DataWarehouse* new_dw,
                          const Ray::modifiesComputes which );

    template< class T >
    void ROI_Extents ( const ProcessorGroup*,
                       const PatchSubset* patches,
                       const MaterialSubset* matls,
                       DataWarehouse*,
                       DataWarehouse* new_dw );

    //______________________________________________________________________
    //  Helpers
    bool less_Eq(    const IntVector& a, const IntVector& b ){
      return ( a.x() <= b.x() && a.y() <= b.y() && a.z() <= b.z() );
    }
    bool greater_Eq( const IntVector& a, const IntVector& b ){
      return ( a.x() >= b.x() && a.y() >= b.y() && a.z() >= b.z() );
    }
    bool greater( const IntVector& a, const IntVector& b ){
      return ( a.x() > b.x() && a.y() > b.y() && a.z() > b.z() );
    }
  }; // class Ray

} // namespace Uintah

#endif
