/*
 * The MIT License
 *
 * Copyright (c) 1997-2025 The University of Utah
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

#include <CCA/Components/MPM/ShellMPM.h>
#include <CCA/Components/MPM/Materials/MPMMaterial.h>
#include <CCA/Components/MPM/Materials/ConstitutiveModel/ShellMaterial.h>
#include <CCA/Components/MPM/Materials/ConstitutiveModel/ConstitutiveModel.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/Task.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Util/DebugStream.h>

using namespace Uintah;

using namespace std;


static DebugStream cout_doing("ShellMPM", false);


///////////////////////////////////////////////////////////////////////////
//
// Construct ShellMPM using the SerialMPM constructor
//
ShellMPM::ShellMPM(const ProcessorGroup* myworld,
                   const MaterialManagerP materialManager) :
  SerialMPM(myworld, materialManager)
{
}

///////////////////////////////////////////////////////////////////////////
//
// Destruct ShellMPM using the SerialMPM destructor
//
ShellMPM::~ShellMPM()
{
}

///////////////////////////////////////////////////////////////////////////
//
// Setup problem -- additional set-up parameters may be added here
// for the shell problem
//
void 
ShellMPM::problemSetup(const ProblemSpecP& prob_spec, 
                       const ProblemSpecP& restart_prob_spec, 
                       GridP& grid)
{
  SerialMPM::problemSetup(prob_spec, restart_prob_spec, grid);
}

///////////////////////////////////////////////////////////////////////////
//
// Setup material part of the problem specific to the shell formulation
// Nothing special right now .. but option of adding stuff is made available
//
// Commenting this out since it isn't called and I don't want to maintain it. JG
#if 0 
void 
ShellMPM::materialProblemSetup(const ProblemSpecP& prob_spec, 
                               MPMLabel* lb, 
                               MPMFlags* flags)
{
  // Search for the MaterialProperties block and then get the MPM section
  ProblemSpecP mat_ps     = prob_spec->findBlockWithOutAttribute("MaterialProperties");
  ProblemSpecP mpm_mat_ps = mat_ps->findBlock("MPM");
  for( ProblemSpecP ps = mpm_mat_ps->findBlock("material"); ps != nullptr; ps = ps->findNextBlock("material") ) {
    MPMMaterial *mat = scinew MPMMaterial(ps, m_materialManager,flags);

    // Register as an MPM material
    mat->registerParticleState( d_particleState,
                                d_particleState_preReloc );

    m_materialManager->registerMaterial( "MPM", mat);
  }
}
#endif

///////////////////////////////////////////////////////////////////////////
//
// Schedule interpolation from particles to the grid
//
void 
ShellMPM::scheduleInterpolateParticlesToGrid(SchedulerP& sched,
                                             const PatchSet* patches,
                                             const MaterialSet* matls)
{
  // First do the standard SerialMPM stuff
  SerialMPM::scheduleInterpolateParticlesToGrid(sched, patches, matls);

  // Then add a task for interpolating shell normal rotations to the grid
  schedInterpolateParticleRotToGrid(sched, patches, matls);
}

///////////////////////////////////////////////////////////////////////////
//
// Schedule interpolation of rotation from particles to the grid
//
void 
ShellMPM::schedInterpolateParticleRotToGrid(SchedulerP& sched,
                                            const PatchSet* patches,
                                            const MaterialSet* matls)
{
  Task* t = scinew Task("ShellMPM::interpolateParticleRotToGrid",
                        this,&ShellMPM::interpolateParticleRotToGrid);
  int numMatls = m_materialManager->getNumMatls( "MPM" );
  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM", m);
    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
    //cerr << "Material = " << m << " numMatls = " << numMatls 
    //   << " mpm_matl = " << mpm_matl << " Cm = " << cm  << endl;
    ShellMaterial* smcm = dynamic_cast<ShellMaterial*>(cm);
    if (smcm) smcm->addComputesRequiresParticleRotToGrid(t, mpm_matl, patches);
  }
  sched->addTask(t, patches, matls);
}

///////////////////////////////////////////////////////////////////////////
//
// Actually interpolate normal rotation from particles to the grid
//
void 
ShellMPM::interpolateParticleRotToGrid(const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset* ,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw)
{
  int numMatls = m_materialManager->getNumMatls( "MPM" );
  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM", m);
    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
    ShellMaterial* smcm = dynamic_cast<ShellMaterial*>(cm);
    if (smcm) smcm->interpolateParticleRotToGrid(patches, mpm_matl, 
                                                 old_dw, new_dw);
  }
}

///////////////////////////////////////////////////////////////////////////
//
// Schedule computation of Internal Force
//
void 
ShellMPM::scheduleComputeInternalForce(SchedulerP& sched,
                                       const PatchSet* patches,
                                       const MaterialSet* matls)
{
  // Call the SerialMPM version first
  SerialMPM::scheduleComputeInternalForce(sched, patches, matls);

  // Add task for computing internal moment for the shell particles
  schedComputeRotInternalMoment(sched, patches, matls);
}

///////////////////////////////////////////////////////////////////////////
//
// Schedule computation of rotational internal moment
//
void 
ShellMPM::schedComputeRotInternalMoment(SchedulerP& sched,
                                        const PatchSet* patches,
                                        const MaterialSet* matls)
{
  Task* t = scinew Task("MPM::computeRotInternalMoment",
                        this, &ShellMPM::computeRotInternalMoment);
  int numMatls = m_materialManager->getNumMatls( "MPM" );
  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM", m);
    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
    ShellMaterial* smcm = dynamic_cast<ShellMaterial*>(cm);
    if (smcm) smcm->addComputesRequiresRotInternalMoment(t, mpm_matl, patches);
  }
  sched->addTask(t, patches, matls);
}

///////////////////////////////////////////////////////////////////////////
//
// Actually compute rotational Internal moment
//
void 
ShellMPM::computeRotInternalMoment(const ProcessorGroup*,
                                   const PatchSubset* patches,
                                   const MaterialSubset* ,
                                   DataWarehouse* old_dw,
                                   DataWarehouse* new_dw)
{
  // Loop over materials
  int numMatls = m_materialManager->getNumMatls( "MPM" );
  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM", m);
    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
    ShellMaterial* smcm = dynamic_cast<ShellMaterial*>(cm);
    if (smcm) smcm->computeRotInternalMoment(patches, mpm_matl, old_dw, new_dw);
  } 
}

///////////////////////////////////////////////////////////////////////////
//
// Schedule Calculation of acceleration
//
void 
ShellMPM::scheduleComputeAndIntegrateAcceleration(SchedulerP& sched,
                                                  const PatchSet* patches,
                                                  const MaterialSet* matls)
{
  SerialMPM::scheduleComputeAndIntegrateAcceleration(sched, patches, matls);

  // Add a task for the rotational acceleration for the shell
  schedComputeRotAcceleration(sched, patches, matls);
}

///////////////////////////////////////////////////////////////////////////
//
// Schedule calculation of rotational acceleration of shell normal
//
void 
ShellMPM::schedComputeRotAcceleration(SchedulerP& sched,
                                      const PatchSet* patches,
                                      const MaterialSet* matls)
{
  Task* t = scinew Task("MPM::computeRotAcceleration",
                        this, &ShellMPM::computeRotAcceleration);

  int numMatls = m_materialManager->getNumMatls( "MPM" );
  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM", m);
    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
    ShellMaterial* smcm = dynamic_cast<ShellMaterial*>(cm);
    if (smcm) smcm->addComputesRequiresRotAcceleration(t, mpm_matl, patches);
  }

  sched->addTask(t, patches, matls);
}

///////////////////////////////////////////////////////////////////////////
//
// Actually calculate of rotational acceleration of shell normal
//
void 
ShellMPM::computeRotAcceleration(const ProcessorGroup*,
                                 const PatchSubset* patches,
                                 const MaterialSubset*,
                                 DataWarehouse* old_dw,
                                 DataWarehouse* new_dw)
{
  // Loop over materials
  int numMatls = m_materialManager->getNumMatls( "MPM" );
  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM", m);
    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
    ShellMaterial* smcm = dynamic_cast<ShellMaterial*>(cm);
    if (smcm) smcm->computeRotAcceleration(patches, mpm_matl, old_dw, new_dw);
  } 
}

///////////////////////////////////////////////////////////////////////////
//
// Schedule interpolation from grid to particles and update
//
void 
ShellMPM::scheduleInterpolateToParticlesAndUpdate(SchedulerP& sched,
                                                  const PatchSet* patches,
                                                  const MaterialSet* matls)
{
  // First Schedule update of the rate of shell normal rotation
  schedParticleNormalRotRateUpdate(sched, patches, matls);

  // Schedule update of the rest using SerialMPM
  SerialMPM::scheduleInterpolateToParticlesAndUpdate(sched, patches, matls);
}

///////////////////////////////////////////////////////////////////////////
//
// Schedule update of the particle normal rotation rate 
//
void 
ShellMPM::schedParticleNormalRotRateUpdate(SchedulerP& sched,
                                           const PatchSet* patches,
                                           const MaterialSet* matls)
{
  Task* t=scinew Task("ShellMPM::schedParticleNormalRotRateUpdate",
                      this, &ShellMPM::particleNormalRotRateUpdate);

  int numMatls = m_materialManager->getNumMatls( "MPM" );
  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM", m);
    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
    ShellMaterial* smcm = dynamic_cast<ShellMaterial*>(cm);
    if (smcm) {
      //GUVMaterial* guv = dynamic_cast<GUVMaterial*>(cm);
      //if (guv)
      //  guv->addComputesRequiresRotRateUpdate(t, mpm_matl, patches);
      //else
        smcm->addComputesRequiresRotRateUpdate(t, mpm_matl, patches);
    }
  }
  sched->addTask(t, patches, matls);
}

///////////////////////////////////////////////////////////////////////////
//
// Actually update the particle normal rotation rate 
//
void 
ShellMPM::particleNormalRotRateUpdate(const ProcessorGroup*,
                                      const PatchSubset* patches,
                                      const MaterialSubset* ,
                                      DataWarehouse* old_dw,
                                      DataWarehouse* new_dw)
{
  // Loop over materials
  int numMatls = m_materialManager->getNumMatls( "MPM" );
  for(int m = 0; m < numMatls; m++){
    MPMMaterial* mpm_matl = (MPMMaterial*) m_materialManager->getMaterial( "MPM", m);
    ConstitutiveModel* cm = mpm_matl->getConstitutiveModel();
    ShellMaterial* smcm = dynamic_cast<ShellMaterial*>(cm);
    if (smcm) {
      //GUVMaterial* guv = dynamic_cast<GUVMaterial*>(cm);
      //if (guv)
      //  guv->particleNormalRotRateUpdate(patches, mpm_matl, old_dw, new_dw);
      //else
        smcm->particleNormalRotRateUpdate(patches, mpm_matl, old_dw, new_dw);
    }
  } 
}

