/*
 * The MIT License
 *
 * Copyright (c) 1997-2014 The University of Utah
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

#ifndef Uintah_Component_Arches_Ray_h
#define Uintah_Component_Arches_Ray_h

#include <CCA/Ports/Scheduler.h>
#include <Core/Containers/StaticArray.h>
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

  class Ray  {

    public: 

      Ray();
      ~Ray(); 

      //__________________________________
      //  TASKS
      /** @brief Interface to input file information */
      void  problemSetup( const ProblemSpecP& prob_spec,
                          const ProblemSpecP& rmcrt_ps,
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

      /** @brief Schedule compute of blackbody intensity */ 
      void sched_sigmaT4( const LevelP& level, 
                          SchedulerP& sched,
                          Task::WhichDW temp_dw,
                          const int radCalc_freq,
                          const bool includeEC = true );


      /** @brief Schedule filtering of q and divQ */
      void sched_filter( const LevelP& level,
                          SchedulerP& sched,
                          Task::WhichDW which_divQ_dw,
                          const bool includeEC = true,
                          bool modifies_divQFilt = false);
                                 
      /** @brief Set boundary conditions and compute sigmaT4 */
      void  sched_setBoundaryConditions( const LevelP& level, 
                                         SchedulerP& sched,
                                         Task::WhichDW temp_dw,
                                         const int radCalc_freq,
                                         const bool backoutTemp = false);

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
 

      //__________________________________
      //  Carry Forward tasks     
      // transfer a variable from old_dw -> new_dw for convience */   
      void sched_CarryForward_Var ( const LevelP& level,
                                    SchedulerP& scheduler,
                                    const VarLabel* variable );

                               
      //__________________________________
      //  Helpers
      /** @brief map the component VarLabels to RMCRT VarLabels */
     void registerVarLabels(int   matl,
                            const VarLabel*  abskg,
                            const VarLabel* absorp,
                            const VarLabel* temperature,
                            const VarLabel* celltype, 
                            const VarLabel* divQ);
                            
    template< class T >                        
    void setBC(CCVariable<T>& Q_CC,
               const std::string& desc,
               const Patch* patch,          
               const int mat_id);

    private: 
      enum DIR {X=0, Y=1, Z=2, NONE=-9};
      //           -x      +x       -y       +y     -z     +z
      enum FACE {EAST=0, WEST=1, NORTH=2, SOUTH=3, TOP=4, BOT=5, nFACES=6};
      
      double d_threshold;
      double d_sigma;
      double d_sigmaScat;
      double d_sigmaT4_thld;                 // threshold values for determining the extents of ROI
      double d_abskg_thld;
      
       
      int    d_nDivQRays;                    // number of rays per cell used to compute divQ
      int    d_nRadRays;                     // number of rays per radiometer used to compute radiative flux
      int    d_nFluxRays;                    // number of rays per cell used to compute radiative flux
      int    d_matl;
      int    d_orderOfInterpolation;        // Order of interpolation for interior fine patch
      
      MaterialSet* d_matlSet;
      IntVector _halo;                      // number of cells surrounding a coarse patch on coarser levels
      
      double d_sigma_over_pi;                // Stefan Boltzmann divided by pi (W* m-2* K-4)
      bool d_isSeedRandom;
      bool d_solveBoundaryFlux;
      bool d_solveDivQ;          
      bool d_allowReflect;                // specify as false when doing DOM comparisons
      bool d_CCRays;
      bool d_onOff_SetBCs;                // switch for setting boundary conditions                    
      bool d_isDbgOn;
      bool d_applyFilter;                 // Allow for filtering of boundFlux and divQ results
      
      enum ROI_algo{fixed, dynamic, patch_based};
      ROI_algo  d_whichROI_algo;
      Point d_ROI_minPt;
      Point d_ROI_maxPt;

      // Virtual Radiometer parameters
      bool d_virtRad;
      double d_viewAng;
      Point d_VRLocationsMin;
      Point d_VRLocationsMax;
      
      struct VR_variables{
        double thetaRot;
        double phiRot; 
        double psiRot;
        double deltaTheta;
        double range;
        double sldAngl;
      };
      VR_variables d_VR;
      
      // Boundary flux constant variables  (consider using array container when C++ 11 is used)
      std::map <int,IntVector> d_dirIndexOrder;
      std::map <int,IntVector> d_dirSignSwap;
      std::map <int,IntVector> d_locationIndexOrder;
      std::map <int,IntVector> d_locationShift;

      Ghost::GhostType d_gn;
      Ghost::GhostType d_gac;

      SimulationStateP d_sharedState;
      const VarLabel* d_sigmaT4_label; 
      const VarLabel* d_abskgLabel;
      const VarLabel* d_absorpLabel;
      const VarLabel* d_temperatureLabel;
      const VarLabel* d_cellTypeLabel;
      const VarLabel* d_divQLabel;
      const VarLabel* d_VRFluxLabel;
      const VarLabel* d_divQFiltLabel;
      const VarLabel* d_boundFluxLabel;
      const VarLabel* d_boundFluxFiltLabel;
      const VarLabel* d_radiationVolqLabel;
      const VarLabel* d_mag_grad_abskgLabel;
      const VarLabel* d_mag_grad_sigmaT4Label;
      const VarLabel* d_flaggedCellsLabel;
      const VarLabel* d_ROI_LoCellLabel;
      const VarLabel* d_ROI_HiCellLabel;

      //----------------------------------------
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

      //______________________________________________________________________
      void rayTraceGPU( Task::CallBackEvent event,
                        const ProcessorGroup* pg,
                        const PatchSubset* patches,
                        const MaterialSubset* matls,
                        DataWarehouse* old_dw,
                        DataWarehouse* new_dw,
                        void* stream,
                        bool modifies_divQ,
                        Task::WhichDW which_abskg_dw,
                        Task::WhichDW whichd_sigmaT4_dw,
                        Task::WhichDW which_celltype_dw,
                        const int radCalc_freq);

      //__________________________________
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
      // @brief Update the running total of the incident intensity */
      void  updateSumI ( Vector& ray_direction, // can change if scattering occurs
                         Vector& ray_location,
                         const IntVector& origin,
                         const Vector& Dx,
                         constCCVariable<double>& sigmaT4Pi,
                         constCCVariable<double>& abskg,
                         constCCVariable<int>& celltype,
                         unsigned long int& size,
                         double& sumI,
                         MTRand& mTwister);
                         
      
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
                            StaticArray< constCCVariable<double> >& sigmaT4Pi,
                            StaticArray< constCCVariable<double> >& abskg,
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

      //----------------------------------------
      void sigmaT4( const ProcessorGroup* pc,
                    const PatchSubset* patches,
                    const MaterialSubset* matls,
                    DataWarehouse* old_dw,
                    DataWarehouse* new_dw,
                    Task::WhichDW which_temp_dw,
                    const int radCalc_freq,
                    const bool includeEC );

      //----------------------------------------
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
      //
      void reflect(double& fs,
                   IntVector& cur,
                   IntVector& prevCell,
                   const double abskg,
                   bool& in_domain,
                   int& step,
                   bool& sign,
                   double& ray_direction);

      //__________________________________
      //
      void findStepSize(int step[],
                        bool sign[],
                        const Vector& inv_direction_vector);
      
      //__________________________________
      //
      void rayLocation( MTRand& mTwister,
                       const IntVector origin,
                       const double DyDx, 
                       const double DzDx,
                       const bool useCCRays,
                       Vector& location);

      
                       
      /** @brief Adjust the location of a ray origin depending on the cell face */
      void rayLocation_cellFace( MTRand& mTwister,
                                 const IntVector& origin,
                                 const IntVector &indexOrder, 
                                 const IntVector &shift, 
                                 const double &DyDx, 
                                 const double &DzDx,
                                 Vector& location );
      //__________________________________
      //
      Vector findRayDirection( MTRand& mTwister,
                               const bool isSeedRandom,
                               const IntVector& = IntVector(-9,-9,-9),
                               const int iRay = -9);
      //__________________________________
      //  
      void rayDirection_VR( MTRand& mTwister,
                            const IntVector& origin,
                            const int iRay,
                            VR_variables& VR,
                            const double DyDx,
                            const double DzDx,
                            Vector& directionVector,
                            double& cosVRTheta );

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
                  
    void coarsen_Q ( const ProcessorGroup*,
                     const PatchSubset* patches,
                     const MaterialSubset* matls,
                     DataWarehouse*, 
                     DataWarehouse* new_dw,
                     const VarLabel* variable,
                     const bool modifies,
                     Task::WhichDW this_dw,
                     const int radCalc_freq);
                     
    void ROI_Extents ( const ProcessorGroup*,
                       const PatchSubset* patches,
                       const MaterialSubset* matls,
                       DataWarehouse*,
                       DataWarehouse* new_dw);
    //______________________________________________________________________
    //    Carry Foward tasks       
    bool doCarryForward( const int timestep,
                         const int radCalc_freq);
                        
    void carryForward_Var ( const ProcessorGroup*,
                            const PatchSubset* ,
                            const MaterialSubset*,
                            DataWarehouse*,
                            DataWarehouse*,
                            const VarLabel* variable);
                        
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
