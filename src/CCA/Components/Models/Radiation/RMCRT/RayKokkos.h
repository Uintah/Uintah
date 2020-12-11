/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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

#ifndef CCA_COMPONENTS_MODELS_RADIATION_RMCRT_RAYKOKKOS_H
#define CCA_COMPONENTS_MODELS_RADIATION_RMCRT_RAYKOKKOS_H

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
#include <sci_defs/kokkos_defs.h>

#ifdef HAVE_CUDA
  #include <curand.h>
  #include <curand_kernel.h>
#endif

#include <iostream>
#include <cmath>
#include <string>
#include <vector>

class ApplicationInterface;

namespace Uintah{

  struct RMCRT_flags{

    struct xyz {
      unsigned int x{0};
      unsigned int y{0};
      unsigned int z{0};
    };

    xyz finePatchLow;
    xyz finePatchSize;

    unsigned int startCell{0};
    unsigned int endCell{0};
    unsigned int cellsPerGroup{0};
  }; // struct RMCRT_flags

  struct LevelParams {

    double Dx[3];                // cell spacing
    double anchor[3];            // level anchor

    LevelParams() = default;

    //Allow only copying, which is needed for functors.
    LevelParams(const LevelParams& rhs) {
      this->Dx[0] = rhs.Dx[0];
      this->Dx[1] = rhs.Dx[1];
      this->Dx[2] = rhs.Dx[2];
      this->anchor[0] = rhs.anchor[0];
      this->anchor[1] = rhs.anchor[1];
      this->anchor[2] = rhs.anchor[2];
    }

    void operator=(const LevelParams& rhs){
      this->Dx[0] = rhs.Dx[0];
      this->Dx[1] = rhs.Dx[1];
      this->Dx[2] = rhs.Dx[2];
      this->anchor[0] = rhs.anchor[0];
      this->anchor[1] = rhs.anchor[1];
      this->anchor[2] = rhs.anchor[2];
    }

    LevelParams(LevelParams&& rhs) = delete;

    void operator=(LevelParams&& rhs) = delete;

    //__________________________________
    //  Portable version of level::getCellPosition()
    KOKKOS_INLINE_FUNCTION
    void getCellPosition(const int x, const int y, const int z, double cellPos[3]) const
    {
      cellPos[0] = anchor[0] + (Dx[0] * x) + (0.5 * Dx[0]);
      cellPos[1] = anchor[1] + (Dx[1] * y) + (0.5 * Dx[1]);
      cellPos[2] = anchor[2] + (Dx[2] * z) + (0.5 * Dx[2]);
    }

    KOKKOS_INLINE_FUNCTION
    void print() const {
      printf(" Dx: [%g,%g,%g]\n", Dx[0], Dx[1], Dx[2]);
    }
  }; // struct LevelParams

  struct LevelParamsML : public LevelParams {

    int    regionLo[3];          // never use these regionLo/Hi in the kernel
    int    regionHi[3];          // they vary on every patch and must be passed into the kernel
    int    refinementRatio[3];
    //int    index;                // level index
    //bool   hasFinerLevel;

    LevelParamsML() = default;

    //Allow only copying, which is needed for functors.
    LevelParamsML(const LevelParamsML& rhs) : LevelParams(rhs) {
      this->regionLo[0] = rhs.regionLo[0];
      this->regionLo[1] = rhs.regionLo[1];
      this->regionLo[2] = rhs.regionLo[2];
      this->regionHi[0] = rhs.regionHi[0];
      this->regionHi[1] = rhs.regionHi[1];
      this->regionHi[2] = rhs.regionHi[2];
      this->refinementRatio[0] = rhs.refinementRatio[0];
      this->refinementRatio[1] = rhs.refinementRatio[1];
      this->refinementRatio[2] = rhs.refinementRatio[2];
    }

    void operator=(const LevelParamsML& rhs) {
      LevelParams::operator=(rhs);
      this->regionLo[0] = rhs.regionLo[0];
      this->regionLo[1] = rhs.regionLo[1];
      this->regionLo[2] = rhs.regionLo[2];
      this->regionHi[0] = rhs.regionHi[0];
      this->regionHi[1] = rhs.regionHi[1];
      this->regionHi[2] = rhs.regionHi[2];
      this->refinementRatio[0] = rhs.refinementRatio[0];
      this->refinementRatio[1] = rhs.refinementRatio[1];
      this->refinementRatio[2] = rhs.refinementRatio[2];
    }

    LevelParamsML(LevelParamsML&& rhs) = delete;

    void operator=(LevelParamsML&& rhs) = delete;

    //__________________________________
    //  Portable version of level::mapCellToCoarser()
    KOKKOS_INLINE_FUNCTION
    void mapCellToCoarser(int idx[3]) const
    {

      //TODO, level::mapCellToCoarser has this code.  Do we need it here too?
      //IntVector refinementRatio = m_refinement_ratio;
      //while (--level_offset) {
      //  refinementRatio = refinementRatio * m_grid->getLevel(m_index - level_offset)->m_refinement_ratio;
      //}
      //IntVector ratio = idx / refinementRatio;

      int ratio[3];
      ratio[0] = idx[0] / refinementRatio[0];
      ratio[1] = idx[1] / refinementRatio[1];
      ratio[2] = idx[2] / refinementRatio[2];

      // If the fine cell index is negative
      // you must add an offset to get the right
      // coarse cell. -Todd
      int offset[3] = {0,0,0};

      if ( (idx[0] < 0) && (refinementRatio[0]  > 1 )){
        offset[0] = (int)fmod( (double)idx[0], (double)refinementRatio[0] ) ;
      }

      if ( (idx[1] < 0) && (refinementRatio[1] > 1 )){
        offset[1] = (int)fmod( (double)idx[1], (double)refinementRatio[1] ) ;
      }

      if ( (idx[2] < 0) && (refinementRatio[2] > 1)){
        offset[2] = (int)fmod( (double)idx[2], (double)refinementRatio[2] ) ;
      }

      idx[0] = ratio[0] + offset[0];
      idx[1] = ratio[1] + offset[1];
      idx[2] = ratio[2] + offset[2];
    }

    KOKKOS_INLINE_FUNCTION
    void print() {
      printf( " LevelParams: Dx: [%g,%g,%g] ", Dx[0], Dx[1], Dx[2]);
      printf( " regionLo: [%i,%i,%i], regionHi: [%i,%i,%i] ",regionLo[0], regionLo[1], regionLo[2], regionHi[0], regionHi[1], regionHi[2]);
      printf( " RefineRatio: [%i,%i,%i]\n",refinementRatio[0], refinementRatio[1], refinementRatio[2]);
    }
  }; // struct LevelParamsML

  class Ray : public RMCRTCommon  {

    public:

      Ray( TypeDescription::Type FLT_DBL );         // This class can be Float or Double
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

      void BC_bulletproofing( const ProblemSpecP& rmcrtps,
                              const bool chk_temp,
                              const bool chk_absk );

      template< class T, class V >
      void setBC( CCVariable<T>& Q_CC,
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
                      dataOnionSlim,       // Derek's experimental fast implementation
                      coarseLevel,
                      singleLevel
                    };

      enum ROI_algo{ fixed,                // user specifies fixed low and high point for a bounding box
                     dynamic,              // user specifies thresholds that are used to dynamically determine ROI
                     patch_based,          // The patch extents + halo are the ROI
                     boundedRayLength,     // the patch extents + boundedRayLength/Dx are the ROI
                     entireDomain          // The ROI is the entire computatonal Domain
                    };

      int d_cellTypeCoarsenLogic{ROUNDUP};           // how to coarsen a cell type

      enum cellTypeCoarsenLogic{ROUNDUP, ROUNDDOWN};

      Algorithm d_algorithm{dataOnion};

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

      bool      m_use_virtual_ROI {false};    //Use virtual ROI set in environment variable VIR_ROI
      IntVector m_virtual_ROI {IntVector(0,0,0)};

      //__________________________________
      template<class T, typename ExecSpace, typename MemSpace>
      void rayTrace( const PatchSubset* patches,
                     const MaterialSubset* matls,
                     OnDemandDataWarehouse* old_dw,
                     OnDemandDataWarehouse* new_dw,
                     UintahParams& uintahParams,
                     ExecutionObject<ExecSpace, MemSpace>& execObj,
                     bool modifies_divQ,
                     Task::WhichDW which_abskg_dw,
                     Task::WhichDW which_sigmaT4_dw,
                     Task::WhichDW which_celltype_dw );

      //__________________________________
      template<int numLevels, typename T, typename ExecSpace, typename MemSpace>
      void rayTrace_dataOnion( const PatchSubset* patches,
                               const MaterialSubset* matls,
                               OnDemandDataWarehouse* old_dw,
                               OnDemandDataWarehouse* new_dw,
                               UintahParams& uintahParams,
                               ExecutionObject<ExecSpace, MemSpace>& execObj,
                               bool modifies_divQ,
                               Task::WhichDW which_abskg_dw,
                               Task::WhichDW which_sigmaT4_dw,
                               Task::WhichDW which_celltype_dw );

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

      /** @brief Determine if a flow cell is adjacent to a wall, and therefore has a boundary */
      bool has_a_boundary( const IntVector &c,
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

#endif // CCA_COMPONENTS_MODELS_RADIATION_RMCRT_RAYKOKKOS_H
