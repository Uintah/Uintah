/*
 *
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
 *
 * ----------------------------------------------------------
 * SPME_Common.h
 *
 *  Created on: May 15, 2014
 *      Author: jbhooper
 */

#ifndef SPME_COMMON_H_
#define SPME_COMMON_H_

#include <vector>

#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SoleVariable.h>
#include <Core/Grid/Variables/PerPatch.h>

#include <Core/Thread/ConditionVariable.h>

#include <Core/Util/DebugStream.h>

#include <CCA/Components/MD/SimpleGrid.h>
#include <CCA/Components/MD/MDSubcomponent.h>

#include <CCA/Components/MD/CoordinateSystems/CoordinateSystem.h>
#include <CCA/Components/MD/Electrostatics/Electrostatics.h>
#include <CCA/Components/MD/Electrostatics/Ewald/Realspace.h>

#include <sci_defs/fftw_defs.h>

#include <vector>

//-------1_________2---------3_________4---------5________6---------7_________8

namespace Uintah {
  typedef std::complex<double> dblcomplex;

  static SCIRun::DebugStream ewaldDebug("EwaldInverseSpace_Dbg", false);


  /*
   * @class   EwaldDirect
   * @ingroup MD
   * @author  Justin Hooper
   * @date    July, 2015
   *
   * @brief   Fourier space solver for direct Ewald summation technique.
   */

  class Ewald: public Electrostatics, public MDSubcomponent {
    public:
      // Constructors and destructors
      Ewald(const double            ewaldBeta,
            const double            cutoffRadius,
            const int               ghostCells,
            const SCIRun::IntVector kLimits,
            const bool              polarizable                 =  false,
            const int               maxPolarizableIterations    = 0,
            const double            polarizationTolerance       =
                                      MDConstants::defaultPolarizationTolerance);
     ~Ewald();

// Inherited from Electrostatics public interface
     /*
      * @brief  Initializes constant data and arrays for the Fourier space
      *         calculations
      */
     virtual void initialize(const ProcessorGroup*      pg,
                             const PatchSubset*         perProcPatches,
                             const MaterialSubset*      atomTypes,
                                   DataWarehouse*     /* old_dw */,
                                   DataWarehouse*       newDw,
                             const SimulationStateP*    simState,
                                   MDSystem*            systemInfo,
                             const MDLabel*             label,
                                   CoordinateSystem*    coordSys);

     virtual void setup     (const ProcessorGroup*      pg,
                             const PatchSubset*         patches,
                             const MaterialSubset*      atomTypes,
                                   DataWarehouse*       oldDW,
                                   DataWarehouse*       newDW,
                             const SimulationStateP*    simState,
                                   MDSystem*            systemInfo,
                             const MDLabel*             label,
                                   CoordinateSystem*    coordSys);

     /*
      * @brief Performs the calculation of the entire charge loop.
      */
      virtual void calculate (const ProcessorGroup*     pg,
                              const PatchSubset*        patches,
                              const MaterialSubset*     materials,
                                    DataWarehouse*      old_dw,
                                    DataWarehouse*      new_dw,
                              const SimulationStateP*   simState,
                                    MDSystem*           systemInfo,
                              const MDLabel*            label,
                                    CoordinateSystem*   coordSys,
                                    SchedulerP&         subscheduler,
                              const LevelP&             level);

     /*
      * @brief Performs clean up after simulation has been run
      */
      virtual void finalize  (const ProcessorGroup*     pg,
                              const PatchSubset*        patches,
                              const MaterialSubset*     materials,
                              DataWarehouse*            old_dw,
                              DataWarehouse*            new_dw,
                              const SimulationStateP*   simState,
                              MDSystem*                 systemInfo,
                              const MDLabel*            label,
                              CoordinateSystem*         coordSys);

//  Inherited from MDSubcomponent
     /*
      * @brief Registers the necessary per-particle variables for this
      *        subcomponent which are to be tracked and updated as particles
      *        transition across patch boundaries
      */
      virtual void registerRequiredParticleStates(
                                         LabelArray& particleState,
                                         LabelArray& particleState_preReloc,
                                         MDLabel* labels) const;

     /*
      * @brief Registers the required and provided variable labels for the
      *        Initialize method
      */
      virtual void addInitializeRequirements(   Task*  task,
                                                MDLabel*  labels) const;
      virtual void addInitializeComputes(       Task*  task,
                                                MDLabel*  labels) const;

     /*
      * @brief Registers the required and provided variable labels for the
      *        Setup method
      */
      virtual void addSetupRequirements(        Task*  task,
                                                MDLabel*  labels) const;
      virtual void addSetupComputes(            Task*  task,
                                                MDLabel*  labels) const;

     /*
      * @brief Registers the required and provided variable labels for the
      *        Calculate method
      */
      virtual void addCalculateRequirements(    Task*  task,
                                                MDLabel*  labels) const;
      virtual void addCalculateComputes(        Task*  task,
                                                MDLabel*  labels) const;

     /*
      * @brief Registers the required and provided variable labels for the
      *        Finalize method
      */
      virtual void addFinalizeRequirements(     Task*  task,
                                                MDLabel*  labels) const;
      virtual void addFinalizeComputes(         Task*  task,
                                                MDLabel*  labels) const;

//  Local methods
      inline double getRealspaceCutoff() {
        return d_electrostaticRadius;
      }

      inline ElectrostaticsType getType() const
      {
        return d_electrostaticMethod;
      }

      inline int requiredGhostCells() const {
        return d_electrostaticGhostCells;
      }

      inline bool isPolarizable() const {
        return f_polarizable;
      }
    private:
      void calculateForceDipole(const ProcessorGroup*       pg,
                                const PatchSubset*             patches,
                                const MaterialSubset*          atomTypes,
                                      DataWarehouse*        subOldDW,
                                      DataWarehouse*        subNewDW,
                                const SimulationStateP*     simState,
                                const MDLabel*              label,
                                      CoordinateSystem*     coordSys);
      void calculateForceCharge(const ProcessorGroup*   pg,
                                const PatchSubset*      patches,
                                const MaterialSubset*   atomTypes,
                                      DataWarehouse*    parentOldDW,
                                      DataWarehouse*    parentNewDW,
                                const SimulationStateP* simState,
                                const MDLabel*          label,
                                      CoordinateSystem* coordSys);

      void initializePrefactors();
      void scheduleCalculateRealspace(const ProcessorGroup*     pg,
                                      const PatchSet*           patches,
                                      const MaterialSet*        atomTypes,
                                            DataWarehouse*      subOldDW,
                                            DataWarehouse*      subNewDW,
                                      const SimulationStateP*   simState,
                                      const MDLabel*            label,
                                            CoordinateSystem*   coordSys,
                                            SchedulerP&         subscheduler,
                                            DataWarehouse*      parentOldDW);
      void scheduleCalculateFourierspace(const ProcessorGroup*      pg,
                                         const PatchSet*            patches,
                                         const MaterialSet*         atomTypes,
                                               DataWarehouse*       subOldDW,
                                               DataWarehouse*       subNewDW,
                                         const SimulationStateP*    simState,
                                         const MDLabel*             label,
                                               CoordinateSystem*    coordSys,
                                               SchedulerP&          subscheduler,
                                               DataWarehouse*       parentOldDW);
      void scheduleUpdateFieldAndStress(const ProcessorGroup*   pg,
                                        const PatchSet*         patches,
                                        const MaterialSet*      atomTypes,
                                              DataWarehouse*    subOldDW,
                                              DataWarehouse*    subNewDW,
                                        const MDLabel*          label,
                                              CoordinateSystem* coordSys,
                                              SchedulerP&       subscheduler,
                                              DataWarehouse*    parentOldDW);
      void scheduleCalculateNewDipoles(const ProcessorGroup*    pg,
                                       const PatchSet*          patches,
                                       const MaterialSet*       atomTypes,
                                             DataWarehouse*     subOldDW,
                                             DataWarehouse*     subNewDW,
                                       const MDLabel*           label,
                                             SchedulerP&        subscheduler,
                                             DataWarehouse*     parentOldDW);
      void scheduleCheckConvergence(const ProcessorGroup*   pg,
                                    const PatchSet*         patches,
                                    const MaterialSet*      atomTypes,
                                          DataWarehouse*    subOldDW,
                                          DataWarehouse*    subNewDW,
                                    const MDLabel*          label,
                                          SchedulerP&       subscheduler,
                                          DataWarehouse*    parentOldDW);

      // Data members
      ElectrostaticsType            d_electrostaticMethod;

      // Charge related variables
      double                        d_ewaldBeta;
      double                        d_electrostaticRadius;
      int                           d_electrostaticGhostCells;
      SCIRun::IntVector             d_kLimits;

      // Dipole related variables
      bool                          f_polarizable;
      int                           d_maxPolarizableIterations;
      const double                  d_polarizationTolerance;
      static const double           d_dipoleMixRatio;

      // Specific to this implementation
      LinearArray3<double>          d_prefactor;
      LinearArray3<Uintah::Matrix3> d_stressPrefactor;

      std::vector<double>           d_kX, d_kY, d_kZ;

      Uintah::Matrix3               d_inverseUnitCell;
  };
}

//    private:
////---->>>> Subordinate functions for the primary interface
//      // Generate a vector of possible fractional frequencies up to
//      // the nyquist frequency, and negative frequencies thereafter
//      /**
//       * @brief Generates split grid vector.
//       *        Generates the vector of points from 0..K/2 in the first half of the array, followed by -K/2..-1
//       * @param mPrime - A reference to the vector to populate
//       * @param kMax - Maximum number of grid points for direction
//       * @param spline - CenteredCardinalBSpline that determines the number of wrapping points necessary
//       * @return std::vector<double> of (0..[m=K/2],[K/2-K]..-1);
//       */
//      inline void generateMPrimeVector(std::vector<double>& mPrime,
//                                       int                  kMax) const {
//        int halfMax = kMax/2;
//
//        for (int Index = 0; Index < kMax; ++Index) {
//          mPrime[Index]     = static_cast<double> (Index);
//        }
//
//        for (int Index = halfMax + 1; Index < kMax; ++Index) {
//          mPrime[Index]    -= static_cast<double> (kMax);
//        }
//
//      }
//
//      void generateBVectorChunk(std::vector<dblcomplex>&    bVector,
//                                const int                   m_initial,
//                                const int                   localGridExtent,
//                                const int                   K) const;
//
//      void generateMPrimeChunk(std::vector<double>& mPrimeChunk,
//                               const int            m_initial,
//                               const int            localGridExtent,
//                               const int            K) const;
//
//      void calculateBGrid(SimpleGrid<double>&   BGrid) const;
//
//      void calculateCGrid(SimpleGrid<double>&   CGrid,
//                          CoordinateSystem*     coordSys) const;
//
//      void calculateStressPrefactor(SimpleGrid<Matrix3>*    stressPrefactor,
//                                    CoordinateSystem*       coordSys);
//
////---->>>> Functions which serve as proxies to schedule the necessary
////         calculations for the electrostatics subscheduler.
//      /*
//       * @brief Schedules local data initialization for the patch Q Grids
//       */
//      void scheduleInitializeLocalStorage(const ProcessorGroup* pg,
//                                          const PatchSet*       patches,
//                                          const MaterialSet*    materials,
//                                                DataWarehouse*  subOldDW,
//                                                DataWarehouse*  subNewDW,
//                                          const MDLabel*        label,
//                                          const LevelP&         level,
//                                                SchedulerP&     sched);
//
//      /*
//       * @brief Schedules the realspace portion of the calculation
//       */
//       void scheduleCalculateRealspace(const ProcessorGroup*    pg,
//                                       const PatchSet*          patches,
//                                       const MaterialSet*       materials,
//                                             DataWarehouse*     subOldDW,
//                                             DataWarehouse*     subNewDW,
//                                       const SimulationStateP*  sharedState,
//                                       const MDLabel*           label,
//                                             CoordinateSystem*  coordSys,
//                                             SchedulerP&        sched,
//                                             DataWarehouse*     parentOldDW);
//
//       /*
//        * @brief    Places the calculation of the charge spreading into the
//        *           task graph
//        */
//       void scheduleCalculatePretransform(const ProcessorGroup*     pg,
//                                          const PatchSet*           patches,
//                                          const MaterialSet*        materials,
//                                                DataWarehouse*      subOldDW,
//                                                DataWarehouse*      subNewDW,
//                                          const SimulationStateP*   simState,
//                                          const MDLabel*            label,
//                                                CoordinateSystem*   coordSystem,
//                                                SchedulerP&         sched,
//                                                DataWarehouse*      parentOldDW);
//       /*
//        * @brief    Places the reduction of nodewide fourier space data into
//        *           the task graph
//        */
//       void scheduleReduceNodeLocalQ(const ProcessorGroup*  pg,
//                                     const PatchSet*        patches,
//                                     const MaterialSet*     materials,
//                                     DataWarehouse*         subOldDW,
//                                     DataWarehouse*         subNewDW,
//                                     const MDLabel*         label,
//                                     SchedulerP&            sched);
//
//       /*
//        * @brief    Places the real->fourier transform into the task graph
//        */
//       void scheduleTransformRealToFourier(const ProcessorGroup*    pg,
//                                           const PatchSet*          patches,
//                                           const MaterialSet*       materials,
//                                           DataWarehouse*           subOldDW,
//                                           DataWarehouse*           subNewDW,
//                                           const MDLabel*           label,
//                                           const LevelP&            level,
//                                           SchedulerP&              sched);
//
//       /*
//        * @brief    Places the fourier space calculations into the task graph
//        */
//       void scheduleCalculateInFourierSpace(const ProcessorGroup*   pg,
//                                            const PatchSet*         patches,
//                                            const MaterialSet*      materials,
//                                            DataWarehouse*          subOldDW,
//                                            DataWarehouse*          subNewDW,
//                                            const MDLabel*          label,
//                                            SchedulerP&             sched);
//
//       /*
//        * @brief    Places the fourier->real transform into the task graph
//        */
//       void scheduleTransformFourierToReal(const ProcessorGroup*    pg,
//                                           const PatchSet*          patches,
//                                           const MaterialSet*       materials,
//                                           DataWarehouse*           subOldDW,
//                                           DataWarehouse*           subNewDW,
//                                           const MDLabel*           label,
//                                           const LevelP&            level,
//                                           SchedulerP&              sched);
//
//       /*
//        * @brief    Places the distribution of aggregate data to individual
//        *           nodes into the task graph
//        */
//       void scheduleDistributeNodeLocalQ(const ProcessorGroup*  pg,
//                                         const PatchSet*        patches,
//                                         const MaterialSet*     materials,
//                                         DataWarehouse*         subOldDW,
//                                         DataWarehouse*         subNewDW,
//                                         const MDLabel*         label,
//                                         SchedulerP&            sched);
//
//       /*
//        * @brief    Places the reduction of the electrostatic field and update
//        *           of stress tensor into the task graph (dipole only)
//        */
//       void scheduleUpdateFieldAndStress(const ProcessorGroup*  pg,
//                                         const PatchSet*        patches,
//                                         const MaterialSet*     materials,
//                                         DataWarehouse*         subOldDW,
//                                         DataWarehouse*         subNewDW,
//                                         const MDLabel*         label,
//                                         CoordinateSystem*      coordSystem,
//                                         SchedulerP&            sched,
//                                               DataWarehouse*   parentOldDW);
//
//       /*
//        * @brief    Check convergence of the iterated dipoles in polarizable systems
//        */
//       void scheduleCheckConvergence(const ProcessorGroup*  pg,
//                                     const PatchSet*        patches,
//                                     const MaterialSet*     materials,
//                                           DataWarehouse*   subOldDW,
//                                           DataWarehouse*   subNewDW,
//                                     const MDLabel*         label,
//                                           SchedulerP&      sched,
//                                           DataWarehouse*   parentOldDW);
//
//       /*
//        * @brief    Place the calculation of new dipoles into the task graph
//        */
//       void scheduleCalculateNewDipoles(const ProcessorGroup*   pg,
//                                        const PatchSet*         patches,
//                                        const MaterialSet*      materials,
//                                              DataWarehouse*    subOldDW,
//                                              DataWarehouse*    subNewDW,
//                                        const SimulationStateP* simState,
//                                        const MDLabel*          label,
//                                              SchedulerP&       sched,
//                                              DataWarehouse*    parentOldDW);
//
//       void scheduleCalculatePostTransform(const ProcessorGroup*    pg,
//                                           const PatchSet*          patches,
//                                           const MaterialSet*       materials,
//                                                 DataWarehouse*     subOldDW,
//                                                 DataWarehouse*     subNewDW,
//                                           const SimulationStateP*  simState,
//                                           const MDLabel*           label,
//                                                 CoordinateSystem*  coordSystem,
//                                                 SchedulerP&        sched);
//
//// ---->>>> Actual calculation routines; non-framework logic resides in these
//       /*
//        * @brief Initializes the local Q grid in a non-race condition fashion
//        */
//       void initializeLocalStorage(const ProcessorGroup*    pg,
//                                   const PatchSubset*       patches,
//                                   const MaterialSubset*    materials,
//                                   DataWarehouse*           subOldDW,
//                                   DataWarehouse*           subNewDW,
//                                   const MDLabel*           label);
//
//       /*
//        * @brief Realspace portion of Ewald calculation for non-dipolar systems
//        */
//       void calculateRealspace(const ProcessorGroup*    pg,
//                               const PatchSubset*       patches,
//                               const MaterialSubset*    materials,
//                                     DataWarehouse*     subOldDW,
//                                     DataWarehouse*     subNewDW,
//                               const SimulationStateP*  sharedState,
//                               const MDLabel*           label,
//                                     CoordinateSystem*  coordSystem,
//                                     DataWarehouse*     parentOldDW);
//
//
//       /*
//        * @brief    Realspace portion of Ewald calculation or induced dipoles
//        *           with Thole screening
//        */
//       void calculateRealspaceTholeDipole(const ProcessorGroup*     pg,
//                                          const PatchSubset*        patches,
//                                          const MaterialSubset*     materials,
//                                                DataWarehouse*      subOldDW,
//                                                DataWarehouse*      subNewDW,
//                                          const SimulationStateP*   sharedState,
//                                          const MDLabel*            label,
//                                                CoordinateSystem*   coordSystem,
//                                                DataWarehouse*      parentOldDW);
//
//       /*
//        * @brief    Realspace portion of Ewald calculation for induced point
//        *           dipole systems
//        */
//       void calculateRealspacePointDipole(const ProcessorGroup*     pg,
//                                          const PatchSubset*        patches,
//                                          const MaterialSubset*     materials,
//                                                DataWarehouse*      subOldDW,
//                                                DataWarehouse*      subNewDW,
//                                          const SimulationStateP*   sharedState,
//                                          const MDLabel*            label,
//                                                CoordinateSystem*   coordSystem,
//                                                DataWarehouse*      parentOldDW);
//
//       /*
//        * @brief    Functions which generate the necessary quantities for
//        *           calculation of realspace contributions to energy, stress,
//        *           and forces
//        */
//       void generatePointScreeningMultipliers(const double& radius,
//                                                    double& B0,
//                                                    double& B1,
//                                                    double& B2,
//                                                    double& B3);
//
//       void generateTholeScreeningMultipliers(const double& a_thole,
//                                              const double& sqrt_alphai_alphaj,
//                                              const double& r,
//                                                    double& B1,
//                                                    double& B2,
//                                                    double& B3);
//
//       void generateDipoleFunctionalTerms(const double&         q_i,
//                                          const double&         q_j,
//                                          const SCIRun::Vector& mu_i,
//                                          const SCIRun::Vector& mu_j,
//                                          const SCIRun::Vector& r_ij,
//                                                double&         mu_jDOTr_ij,
//                                                double&         G0,
//                                                double&         G1_mu_q,
//                                                double&         G1_mu_mu,
//                                                double&         G2,
//                                                SCIRun::Vector& gradG0,
//                                                SCIRun::Vector& gradG1,
//                                                SCIRun::Vector& gradG2);
//
//       void generateVectorSubsets(const SCIRun::Vector& R);
//
//       /*
//        * @brief    Generates the basic coefficients for mapping charge to the
//        *           fourier space grid
//        */
//       void generateChargeMap(const ProcessorGroup*  pg,
//                              const PatchSubset*     patches,
//                              const MaterialSubset*  materials,
//                              DataWarehouse*         subOldDW,
//                              DataWarehouse*         subNewDW,
//                              const MDLabel*         label,
//                              CoordinateSystem*      coordSystem);
//
//       /*
//         * @brief    Generates the basic coefficients for mapping charge +
//         *           dipole contributions to the fourier space grid.
//         */
//        void generateChargeMapDipole(const ProcessorGroup*  pg,
//                                     const PatchSubset*     patches,
//                                     const MaterialSubset*  materials,
//                                     DataWarehouse*         old_dw,
//                                     DataWarehouse*         new_dw,
//                                     const MDLabel*         label,
//                                     CoordinateSystem*      coordSystem);
//        /*
//         * @brief   Performs calculations necessary to fill the fourier charge
//         *          grid.
//         */
//       void calculatePreTransform(const ProcessorGroup*     pg,
//                                  const PatchSubset*        patches,
//                                  const MaterialSubset*     materials,
//                                        DataWarehouse*      oldDW,
//                                        DataWarehouse*      newDW,
//                                  const SimulationStateP*   simState,
//                                  const MDLabel*            label,
//                                        CoordinateSystem*   coordSys,
//                                        DataWarehouse*      parentOldDW);
//
//       /*
//        * @brief    Performs calculations necessary to fill the fourier charge
//        *           + dipole grid.
//        */
//       void calculatePreTransformDipole(const ProcessorGroup*   pg,
//                                        const PatchSubset*      patches,
//                                        const MaterialSubset*   materials,
//                                              DataWarehouse*          oldDW,
//                                              DataWarehouse*          newDW,
//                                        const SimulationStateP*       simState,
//                                        const MDLabel*          label,
//                                              CoordinateSystem*       coordSys,
//                                              DataWarehouse*    parentOldDW);
//
//       /*
//        * @brief    Maps the particle charge value onto the K-space grid
//        */
//       void mapChargeToGrid(SPMEPatch*              spmePatch,
//                            const spmeMapVector*    gridMap,
//                            ParticleSubset*         atomSet,
//                            double                  charge,
//                            CoordinateSystem*       coordSys);
//
//       /*
//        * @brief    Maps the particle charge/dipole value onto the K-space grid
//        */
//       void mapChargeToGridDipole(SPMEPatch*                        spmePatch,
//                                  const spmeMapVector*              gridMap,
//                                  ParticleSubset*                   pset,
//                                  double                            charge,
//                                  constParticleVariable<Vector>&    p_Dipole,
//                                  CoordinateSystem*                 coordSys);
//
//       void calculateInFourierSpace(const ProcessorGroup*   pg,
//                                    const PatchSubset*      patches,
//                                    const MaterialSubset*   materials,
//                                    DataWarehouse*          oldDW,
//                                    DataWarehouse*          newDW,
//                                    const MDLabel*          label);
//
//       /*
//        * @brief    Calculate the new predicted dipole from the total field
//        */
//       void calculateNewDipoles(const ProcessorGroup*   pg,
//                                const PatchSubset*      patches,
//                                const MaterialSubset*   materials,
//                                      DataWarehouse*    oldDW,
//                                      DataWarehouse*    newDW,
//                                const SimulationStateP* sharedState,
//                                const MDLabel*          label,
//                                      DataWarehouse*    parentOldDW);
//
//       /*
//        * @brief    Determine the force of the charge-only system
//        */
//       void mapForceFromGrid(const SPMEPatch*           spmePatch,
//                             const spmeMapVector*       gridMap,
//                             ParticleSubset*            atomSet,
//                             const double               charge,
//                             ParticleVariable<Vector>&  pForceRecip,
//                             CoordinateSystem*          coordSys);
//
//       /*
//        * @brief    Determine the force of the induced dipole system
//        */
//       void mapForceFromGridDipole(const SPMEPatch*                 spmePatch,
//                                   const spmeMapVector*             gridMap,
//                                   ParticleSubset*                  particles,
//                                   const double                     charge,
//                                   const ParticleVariable<Vector>&  pDipole,
//                                   ParticleVariable<Vector>&        pForceRecip,
//                                   CoordinateSystem*                coordSystem);
//
//       /*
//        * @brief    Set up force calculations after real->fourier transform
//        */
//       void calculatePostTransformDipole(const ProcessorGroup*  pg,
//                                         const PatchSubset*     patches,
//                                         const MaterialSubset*  materials,
//                                         DataWarehouse*         oldDW,
//                                         DataWarehouse*         newDW,
//                                         const SimulationStateP*      simState,
//                                         const MDLabel*         label,
//                                         CoordinateSystem*      coordSystem);
//
//       /*
//        * @brief    Calculate forces for the non-dipolar system
//        */
//       void calculatePostTransform(const ProcessorGroup*    pg,
//                                   const PatchSubset*       patches,
//                                   const MaterialSubset*    materials,
//                                   DataWarehouse*           oldDW,
//                                   DataWarehouse*           newDW,
//                                   const SimulationStateP*        simState,
//                                   const MDLabel*           label,
//                                   CoordinateSystem*        coordSystem);
//
//       /*
//        * @brief    Calculates the updated field for the induced polarizable
//        *           system and also calculates the dipole correction to the
//        *           stress tensor.
//        */
//       void dipoleUpdateFieldAndStress(const ProcessorGroup*    pg,
//                                       const PatchSubset*       patches,
//                                       const MaterialSubset*    materials,
//                                             DataWarehouse*     oldDW,
//                                             DataWarehouse*     newDW,
//                                       const MDLabel*           label,
//                                             CoordinateSystem*  coordSystem,
//                                             DataWarehouse*     parentOldDW);
//
//       /*
//        * @brief    On-processor reduction of the local, per-thread instances
//        *           of the Q grid before the FFT is called.
//        */
//       void reduceNodeLocalQ(const ProcessorGroup*  pg,
//                             const PatchSubset*     patches,
//                             const MaterialSubset*  materials,
//                                   DataWarehouse*         oldDW,
//                                   DataWarehouse*         newDW,
//                             const MDLabel*         label);
//
//       /*
//        * @brief    Places the transformed Q data back into the local,
//        *           per-thread Q variable
//        */
//       void distributeNodeLocalQ(const ProcessorGroup*  pg,
//                                 const PatchSubset*     patches,
//                                 const MaterialSubset*  materials,
//                                       DataWarehouse*         oldDW,
//                                       DataWarehouse*         newDW,
//                                 const MDLabel*         label);
//
//       void transformRealToFourier(const ProcessorGroup*    pg,
//                                   const PatchSubset*       patches,
//                                   const MaterialSubset*    materials,
//                                         DataWarehouse*           oldDW,
//                                         DataWarehouse*           newDW,
//                                   const MDLabel*           label);
//
//       void transformFourierToReal(const ProcessorGroup*    pg,
//                                   const PatchSubset*       patches,
//                                   const MaterialSubset*    materials,
//                                   DataWarehouse*           oldDW,
//                                   DataWarehouse*           newDW,
//                                   const MDLabel*           label);
//
//
//       void checkConvergence(const ProcessorGroup*      pg,
//                             const PatchSubset*         patches,
//                             const MaterialSubset*      materials,
//                                   DataWarehouse*       oldDW,
//                                   DataWarehouse*       newDW,
//                             const MDLabel*             label,
//                                   DataWarehouse*       parentOldDW);
//
////  Data members
//   // Implementation type for electrostatic calculation
//       ElectrostaticsType           d_electrostaticMethod;
//
//  //--> Basic charge related variables
//   // B1: Ewald calculation damping coefficient
//   // B2: Cutoff radius for realspace portion of Ewald calculation
//   // B3: Number of ghost cells necessary for realspace portion of Ewald calc.
//   // B4: Number of fourier space grid divisions along each unit axis
//       double                       d_ewaldBeta;                // B1
//       double                       d_electrostaticRadius;      // B2
//       int                          d_electrostaticGhostCells;  // B3
//       SCIRun::IntVector            d_kLimits;                  // B4
//
//
//  //--> Dipole related variables
//   // D1) Whether or not calculation includes induced polarizability
//   // D2) Number of iterations to perform before giving up on convergence
//   // D3) Tolerance threshold for polarizability convergence
//   // D4)Amount of old dipole to mix with new in convergence loop
//       bool                         f_polarizable;              // D1
//       int                          d_maxPolarizableIterations; // D2
//       const double                 d_polarizationTolerance;    // D3
//       static const double          d_dipoleMixRatio;           // D4
//
//  //--> FFTW related variables and structures
//       struct LocalFFTData {
//         fftw_complex*    complexData;
//         ptrdiff_t        numElements;
//         ptrdiff_t        startAddress;
//       };
//   // F1) Forward FFTW MPI transformation plan:
//   // F2) Backward FFTW MPI transformation plan:
//   // F3) Local portion of the entire FFT data set:
//       fftw_plan                    d_forwardPlan;              // F1
//       fftw_plan                    d_backwardPlan;             // F2
//       LocalFFTData                 d_localFFTData;             // F3
//
//  //--> Processor local helper variables and objects necessary to perform SPME
//   // H1: Spline object used to spread the aggregate charges onto Fourier grid
//   // H2: Map which holds the pieces of the k-Space grid, indexed by the patch
//   //     to which they correspond
//   // H3: Data structure which holds the entire processor contribution to the
//   //     gridded charge
//   // H4: Data structure which acts as a scratch pad for calculations on the
//   //     local processor contribution to the gridded charge
//   // H5: Lock for threaded access to the local processor charge grids
//       ShiftedCardinalBSpline       d_interpolatingSpline;      // H1
//       std::map<int, SPMEPatch*>    d_spmePatchMap;             // H2
//       SimpleGrid<dblcomplex>*      d_Q_nodeLocal;              // H3
//       SimpleGrid<dblcomplex>*      d_Q_nodeLocalScratch;       // H4
//       Mutex                        d_Qlock;                    // H5
//
//       mutable CrowdMonitor         d_spmeLock;
//  };
//}

#endif /* SPME_COMMON_H_ */
