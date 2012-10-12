/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef UINTAH_HOMEBREW_SHELLMPM_H
#define UINTAH_HOMEBREW_SHELLMPM_H

#include <CCA/Components/MPM/SerialMPM.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>

namespace Uintah {

using namespace SCIRun;

  /////////////////////////////////////////////////////////////////////////////
  /*!
    \class  ShellMPM
    \brief  Extended MPM with extra stuff for shell formulation
    \author Biswajit Banerjee \n
            C-SAFE and Department of Mechanical Engineering \n
            University of Utah \n
            
  */
  /////////////////////////////////////////////////////////////////////////////

class ShellMPM : public SerialMPM {

public:

  ///////////////////////////////////////////////////////////////////////////
  //
  /*! Constructor:  Uses the constructor of SerialMPM.  Changes to
  //  the constructor of SerialMPM should be reflected here.*/
  //
  ///////////////////////////////////////////////////////////////////////////
  ShellMPM(const ProcessorGroup* myworld);
  virtual ~ShellMPM();

  ///////////////////////////////////////////////////////////////////////////
  //
  /*! Setup problem -- additional set-up parameters may be added here
  // for the shell problem */
  //
  ///////////////////////////////////////////////////////////////////////////
  virtual void problemSetup(const ProblemSpecP& params, 
                            const ProblemSpecP& restart_prob_spec, 
                            GridP& grid,SimulationStateP&);
         
protected:

  ///////////////////////////////////////////////////////////////////////////
  //
  /*! Setup problem -- material parameters specific to shell */
  //
  ///////////////////////////////////////////////////////////////////////////
  virtual void materialProblemSetup(const ProblemSpecP& prob_spec, 
                                    SimulationStateP& sharedState,
                                    MPMLabel* lb, MPMFlags* flags);
         
  ///////////////////////////////////////////////////////////////////////////
  //
  /*! Schedule interpolation from particles to the grid */
  //
  ///////////////////////////////////////////////////////////////////////////
  virtual void scheduleInterpolateParticlesToGrid(SchedulerP& sched,
                                                  const PatchSet* patches,
                                                  const MaterialSet* matls);

  ///////////////////////////////////////////////////////////////////////////
  //
  /*! Schedule interpolation of rotation from particles to the grid */
  //
  ///////////////////////////////////////////////////////////////////////////
  void schedInterpolateParticleRotToGrid(SchedulerP& sched,
                                         const PatchSet* patches,
                                         const MaterialSet* matls);

  ///////////////////////////////////////////////////////////////////////////
  //
  /*! Actually interpolate normal rotation from particles to the grid */
  //
  ///////////////////////////////////////////////////////////////////////////
  void interpolateParticleRotToGrid(const ProcessorGroup*,
                                    const PatchSubset* patches,
                                    const MaterialSubset* ,
                                    DataWarehouse* old_dw,
                                    DataWarehouse* new_dw);

  ///////////////////////////////////////////////////////////////////////////
  //
  /*! Schedule computation of Internal Force */
  //
  ///////////////////////////////////////////////////////////////////////////
  virtual void scheduleComputeInternalForce(SchedulerP& sched,
                                            const PatchSet* patches,
                                            const MaterialSet* matls);

  ///////////////////////////////////////////////////////////////////////////
  //
  /*! Schedule computation of rotational internal moment */
  //
  ///////////////////////////////////////////////////////////////////////////
  void schedComputeRotInternalMoment(SchedulerP& sched,
                                     const PatchSet* patches,
                                     const MaterialSet* matls);

  ///////////////////////////////////////////////////////////////////////////
  //
  /*! Actually compute rotational internal moment */
  //
  ///////////////////////////////////////////////////////////////////////////
  void computeRotInternalMoment(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* ,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw);

  ///////////////////////////////////////////////////////////////////////////
  //
  /*! Schedule Calculation of acceleration */
  //
  virtual void scheduleComputeAndIntegrateAcceleration(SchedulerP& sched,
                                            const PatchSet* patches,
                                            const MaterialSet* matls);

  ///////////////////////////////////////////////////////////////////////////
  //
  /*! Schedule calculation of rotational acceleration of shell normal */
  //
  void schedComputeRotAcceleration(SchedulerP& sched,
                                   const PatchSet* patches,
                                   const MaterialSet* matls);

  ///////////////////////////////////////////////////////////////////////////
  //
  /*! Actually calculate of rotational acceleration of shell normal */
  //
  void computeRotAcceleration(const ProcessorGroup*,
                              const PatchSubset* patches,
                              const MaterialSubset*,
                              DataWarehouse* old_dw,
                              DataWarehouse* new_dw);

  ///////////////////////////////////////////////////////////////////////////
  //
  /*! Schedule interpolation from grid to particles and update */
  //
  virtual void scheduleInterpolateToParticlesAndUpdate(SchedulerP& sched,
                                               const PatchSet* patches,
                                               const MaterialSet* matls);

  ///////////////////////////////////////////////////////////////////////////
  //
  /*! Schedule update of the particle normal rotation rate */
  //
  void schedParticleNormalRotRateUpdate(SchedulerP& sched,
                                        const PatchSet* patches,
                                        const MaterialSet* matls);

  ///////////////////////////////////////////////////////////////////////////
  //
  /*! Actually update the particle normal rotation rate */
  //
  void particleNormalRotRateUpdate(const ProcessorGroup*,
                                   const PatchSubset* patches,
                                   const MaterialSubset* ,
                                   DataWarehouse* old_dw,
                                   DataWarehouse* new_dw);
private:

  ///////////////////////////////////////////////////////////////////////////
  //
  /*! Forbid copying of this class */
  //
  ShellMPM(const ShellMPM&);
  ShellMPM& operator=(const ShellMPM&);
         
};
      
} // end namespace Uintah

#endif
