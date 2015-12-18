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
 * velocityVerlet.cc
 *
 *  Created on: Oct 21, 2014
 *      Author: jbhooper
 */

#include <CCA/Components/MD/Integrators/velocityVerlet/velocityVerlet.h>

#include <Core/Thread/Mutex.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>

using namespace Uintah;
SCIRun::Mutex forceFileLock("Locks for force output files");

velocityVerlet::velocityVerlet(const VarLabel* _dt_label)
                             :dt_label(_dt_label),
                              d_integratorType("velocityVerlet")

{
  d_firstIntegration = true;
  d_previousKE   = 0.0;
  d_previousPE   = 0.0;
  d_previousTemp = 0.0;
  d_previousMass = 0.0;
  d_dt = 0.0;

  d_previousMomentum = MDConstants::V_ZERO;
  d_currentTimestep = 0;
}

velocityVerlet::~velocityVerlet()
{

}

void velocityVerlet::registerRequiredParticleStates(
                                           varLabelArray& particleState,
                                           varLabelArray& particleState_preReloc,
                                           MDLabel*    labels) const
{

  // Technically, not all integrators must necessarily track velocity
  // explicitly.  So the velocity label probably belongs to the integrator.
  particleState.push_back(labels->global->pV);
  particleState.push_back(labels->global->pID);

  particleState_preReloc.push_back(labels->global->pV_preReloc);
  particleState_preReloc.push_back(labels->global->pID_preReloc);

}

void velocityVerlet::addInitializeRequirements(       Task        * task
                                              ,       MDLabel     * label
                                              , const PatchSet    * patches
                                              , const MaterialSet * matls
                                              , const Level       * level
                                              ) const
{
  // Nothing to add
}

void velocityVerlet::addInitializeComputes(       Task        * task
                                          ,       MDLabel     * label
                                          , const PatchSet    * patches
                                          , const MaterialSet * matls
                                          , const Level       * level
                                           ) const
{
  const MaterialSubset* matl_subset = matls->getUnion();
  task->computes(label->integrator->fPatchFirstIntegration, level, matl_subset, Task::NormalDomain);
}

void velocityVerlet::initialize(const ProcessorGroup*       pg,
                                const PatchSubset*          patches,
                                const MaterialSubset*       atomTypes,
                                      DataWarehouse*      /*oldDW*/,
                                      DataWarehouse*        newDW,
                                const SimulationStateP*     simState,
                                      MDSystem*             systemInfo,
                                const MDLabel*              label,
                                      CoordinateSystem*     coordSys)
{
  // Set the flags on each patch so we know it's the first integration of the
  // particle coordinates on that patch.
  int numPatches = patches->size();
  int numTypes = atomTypes->size();
  for (int patchIndex = 0; patchIndex < numPatches; ++patchIndex) {
    const Patch* currPatch = patches->get(patchIndex);
    for (int typeIndex = 0; typeIndex < numTypes; ++typeIndex) {
      int currType = atomTypes->get(typeIndex);
      PerPatch<bool> patchFirstIntegration = true;
      newDW->put(patchFirstIntegration, label->integrator->fPatchFirstIntegration, currType, currPatch);
    }
  }  // No code
}

void velocityVerlet::addSetupRequirements(     Task*    task,
                                               MDLabel* labels
                                         ) const
{
  // Nothing to add - yet
}

void velocityVerlet::addSetupComputes(         Task*    task,
                                               MDLabel* labels
                                     ) const
{
  // Nothing to add - yet
}

void velocityVerlet::setup(     const ProcessorGroup*       pg,
                                const PatchSubset*          patches,
                                const MaterialSubset*       atomTypes,
                                      DataWarehouse*        oldDW,
                                      DataWarehouse*        newDW,
                                const SimulationStateP*     simState,
                                      MDSystem*             systemInfo,
                                const MDLabel*              label,
                                      CoordinateSystem*     coordSys)
{
  // No code
}

void velocityVerlet::addCalculateRequirements(       Task        * task
                                             ,       MDLabel     * labels
                                             , const PatchSet    * patches
                                             , const MaterialSet * matls
                                             , const Level       * level ) const
{
  const MaterialSubset* matl_subset = matls->getUnion();

  task->requires(Task::OldDW, labels->global->pX, level, Ghost::None, 0);
  task->requires(Task::OldDW, labels->global->pV, level, Ghost::None, 0);
  task->requires(Task::OldDW, labels->global->pID, level, Ghost::None, 0);

  task->requires(Task::OldDW, labels->global->rTotalMomentum, level);
  task->requires(Task::OldDW, labels->global->rTotalMass, level);
  task->requires(Task::OldDW, labels->global->rKineticEnergy, level);

  task->requires(Task::OldDW, labels->nonbonded->rNonbondedEnergy, level);
  task->requires(Task::OldDW, labels->electrostatic->rElectrostaticRealEnergy, level);
  task->requires(Task::OldDW, labels->electrostatic->rElectrostaticInverseEnergy, level);

  task->requires(Task::OldDW, labels->integrator->fPatchFirstIntegration, level, Ghost::None, 0);

  // Eventually the individual components should actually take care of
  // dropping their force contributions into a general "Force" array
  // so that I can just require the current forces here:
  // task->requires(Task::NewDW,
  //                labels->global->pF_preReloc,
  //                Ghost::None,
  //                0);

  // pF_elec_inv+
  task->requires(Task::NewDW,
                 labels->electrostatic->pF_electroInverse_preReloc,
                 level,
                 matl_subset,
                 Task::NormalDomain,
                 Ghost::None,
                 0);

  // pF_nb_MD+
  task->requires(Task::NewDW,
                 labels->nonbonded->pF_nonbonded_preReloc,
                 level,
                 matl_subset,
                 Task::NormalDomain,
                 Ghost::None,
                 0);


  task->requires(Task::NewDW,
                 labels->electrostatic->pF_electroReal_preReloc,
                 level,
                 matl_subset,
                 Task::NormalDomain,
                 Ghost::None,
                 0);

  // Finally, we kinda need the timestep to integrate!
  task->requires(Task::OldDW, dt_label);
}

void velocityVerlet::addCalculateComputes(       Task        * task
                                         ,       MDLabel     * labels
                                         , const PatchSet    * patches
                                         , const MaterialSet * matls
                                         , const Level       * level ) const
{
  const MaterialSubset* matl_subset = matls->getUnion();

  task->computes(labels->global->pX_preReloc,  level, matl_subset, Task::NormalDomain);
  task->computes(labels->global->pV_preReloc,  level, matl_subset, Task::NormalDomain);
  task->computes(labels->global->pID_preReloc, level, matl_subset, Task::NormalDomain);

  task->computes(labels->global->rKineticEnergy, level);
  task->computes(labels->global->rKineticStress, level);
  task->computes(labels->global->rTotalMomentum, level);
  task->computes(labels->global->rTotalMass,     level);

  task->computes(labels->integrator->fPatchFirstIntegration, level, matl_subset, Task::NormalDomain);

}

void velocityVerlet::calculate( const ProcessorGroup*       pg,
                                const PatchSubset*          patches,
                                const MaterialSubset*       atomTypes,
                                      DataWarehouse*        oldDW,
                                      DataWarehouse*        newDW,
                                const SimulationStateP*     simState,
                                      MDSystem*             systemInfo,
                                const MDLabel*              label,
                                      CoordinateSystem*     coordSys)
{

  // Because of the way Uintah pushes reductions, we only ever have the
  // total energies from the previous timestep.
  sum_vartype       previousMass, previousKE, componentPE;
  sumvec_vartype    previousMomentum;
  const Level* level = getLevel(patches);
  oldDW->get(previousMass,     label->global->rTotalMass    , level);
  oldDW->get(previousKE,       label->global->rKineticEnergy, level);
  oldDW->get(previousMomentum, label->global->rTotalMomentum, level);

  d_previousKE          = previousKE;
  d_previousMomentum    = previousMomentum;
  d_previousMass        = previousMass;

  oldDW->get(componentPE, label->nonbonded->rNonbondedEnergy, level);
  d_previousPE = componentPE;
  oldDW->get(componentPE, label->electrostatic->rElectrostaticRealEnergy, level);
  d_previousPE += componentPE;
  oldDW->get(componentPE, label->electrostatic->rElectrostaticInverseEnergy, level);
  d_previousPE += componentPE;

  delt_vartype delT;
  oldDW->get(delT, (*simState)->get_delt_label(), level);
  d_dt = delT;

  double kineticEnergy = 0.0;
  double totalMass     = 0.0;
  SCIRun::Vector totalMomentum = MDConstants::V_ZERO;
  double normKE    = 0.5*0.001*(1.0/4184.0);

  int numPatches = patches->size();
  int numTypes   = atomTypes->size();

  for (int patchIndex = 0; patchIndex < numPatches; ++patchIndex)
  {
    const Patch* currPatch = patches->get(patchIndex);
    for (int typeIndex = 0; typeIndex < numTypes; ++typeIndex)
    {
      int currType = atomTypes->get(typeIndex);
      PerPatch<bool> patchFirstIntegration;
      oldDW->get(patchFirstIntegration,
                 label->integrator->fPatchFirstIntegration,
                 currType,
                 currPatch);
      if (patchFirstIntegration)
      {
        firstIntegratePatch(currPatch, typeIndex,
                            oldDW, newDW, simState, label,
                            kineticEnergy, totalMass, totalMomentum);
        patchFirstIntegration = false;

      }
      else
      {
        integratePatch(currPatch, typeIndex,
                       oldDW, newDW, simState, label,
                       kineticEnergy, totalMass, totalMomentum);
      }
      newDW->put(patchFirstIntegration,
                 label->integrator->fPatchFirstIntegration,
                 currType,
                 currPatch);
      ParticleSubset* delset = scinew ParticleSubset(0, currType, currPatch);
      newDW->deleteParticles(delset);
    }
  }

  kineticEnergy *= normKE;
  newDW->put( sum_vartype(kineticEnergy),    label->global->rKineticEnergy, level);
  newDW->put( sum_vartype(totalMass),        label->global->rTotalMass,     level);
  newDW->put( sumvec_vartype(totalMomentum), label->global->rTotalMomentum, level);
//  newDW->put( matrix_sum(kineticStress),     label->global->rKineticStress);

//  // FIXME TODO We should be checking firstIntegration on a PER PATCH basis here
//  // rather than as a global variable.  The global is a race condition and may
//  // be why multi-patch with ghost cells in a single patch is not consistent.
//  if (d_firstIntegration)
//  {
//    d_firstIntegration = false;
//    firstIntegrate(patches, atomTypes, oldDW, newDW, simState, label);
//
//
//  }
//  else
//  {
//    integrate(patches, atomTypes, oldDW, newDW, simState, label);
//  }
}

void velocityVerlet::addFinalizeRequirements(  Task*    task,
                                               MDLabel* labels
                                            ) const
{
//  task->requires(Task::NewDW,
//                  labels->nonbonded->pF_nonbonded_preReloc,
//                  Ghost::None,
//                  0);
//  task->requires(Task::NewDW,
//                  labels->global->pX_preReloc,
//                  Ghost::None,
//                  0);
//  task->requires(Task::NewDW,
//                  labels->global->pV_preReloc,
//                  Ghost::None,
//                  0);
//  task->requires(Task::NewDW,
//                 labels->global->pID_preReloc,
//                 Ghost::None,
//                 0);
  // No code
}

void velocityVerlet::addFinalizeComputes(      Task*    task,
                                               MDLabel* labels
                                        ) const
{
  // No code
}

void velocityVerlet::finalize(  const ProcessorGroup*       pg,
                                const PatchSubset*          patches,
                                const MaterialSubset*       atomTypes,
                                      DataWarehouse*        oldDW,
                                      DataWarehouse*        newDW,
                                const SimulationStateP*     simState,
                                      MDSystem*             systemInfo,
                                const MDLabel*              label,
                                      CoordinateSystem*     coordSys)
{
//  size_t numPatches = patches->size();
//  size_t numAtomTypes = atomTypes->size();
//
//  std::ostringstream forceOutName;
//  forceOutName << "forceOutput.txt_" << std::left << Uintah::Parallel::getMPIRank();
//  std::ofstream forceOutFile;
//  forceFileLock.lock();
//  forceOutFile.open(forceOutName.str().c_str(), std::fstream::out | std::fstream::app );
//  for (size_t patchIndex = 0; patchIndex < numPatches; ++patchIndex)
//  {
//    const Patch* currPatch = patches->get(patchIndex);
//    for (size_t atomIndex = 0; atomIndex < numAtomTypes; ++atomIndex)
//    {
//      int atomType = atomTypes->get(atomIndex);
//      ParticleSubset* atomSet = oldDW->getParticleSubset(atomType,currPatch);
//
//      constParticleVariable<long64> atomID;
//      constParticleVariable<SCIRun::Vector> atomF, atomV;
//      constParticleVariable<SCIRun::Point> atomX;
//
//      newDW->get(atomID, label->global->pID_preReloc, atomSet);
//      newDW->get(atomX, label->global->pX_preReloc, atomSet);
//      newDW->get(atomV, label->global->pV_preReloc, atomSet);
//      newDW->get(atomF, label->nonbonded->pF_nonbonded_preReloc, atomSet);
//
//      size_t numAtoms = atomSet->numParticles();
//      std::cerr  << "Timestep: " << timestep
//                 << " Patch: " << currPatch->getID()
//                 << " sees " << numAtoms << " atoms." << std::endl;
//      for (size_t atom = 0; atom < numAtoms; ++atom)
//      {
//        forceOutFile << "t: " << std::setw(5) << std::right << std::fixed
//                   << timestep
//                   << " Source Atom: " << std::setw(8) << std::right << std::fixed
//                   << atomID[atom]
//                   << " F: " << atomF[atom]
//                   << " V: " << atomV[atom]
//                   << " X: " << atomX[atom].asVector() << std::endl;
//      }
//    }
//  }
//  forceOutFile.close();
//  forceFileLock.unlock();
  // No Code
}

void velocityVerlet::integratePatch(const Patch*            workPatch,
                                    const int               localMDMaterialIndex,
                                          DataWarehouse*    oldDW,
                                          DataWarehouse*    newDW,
                                    const SimulationStateP* simState,
                                    const MDLabel*          label,
                                          double&           kineticEnergy,
                                          double&           totalMass,
                                          SCIRun::Vector&   totalMomentum)
{
//  SCIRun::Vector momentumFraction = d_previousMomentum/d_previousMass;
  // Normalization constants
  double forceNorm = 41.84;
  double velocNorm = 1.0e-5;

  SCIRun::Vector F_nPlus1;
  double atomMass = (*simState)->getMDMaterial(localMDMaterialIndex)->getMass();
  int    globalMDMaterialIndex = (*simState)->getMDMaterial(localMDMaterialIndex)->getDWIndex();
  double massInv  = 1.0/atomMass;

  ParticleSubset* atomSet = oldDW->getParticleSubset(globalMDMaterialIndex, workPatch);
  constParticleVariable<long64> pID_n;
       ParticleVariable<long64> pID_nPlus1;
  constParticleVariable<Point>  pX_n;
       ParticleVariable<Point>  pX_nPlus1;
  constParticleVariable<SCIRun::Vector> pV_n;
       ParticleVariable<SCIRun::Vector> pV_nPlus1;

  constParticleVariable<SCIRun::Vector> pF_nb_nPlus1,
                                        pF_eReal_nPlus1,
                                        pF_eInv_nPlus1;

  oldDW->get(pID_n, label->global->pID, atomSet);
  oldDW->get(pX_n,  label->global->pX,  atomSet);
  oldDW->get(pV_n,  label->global->pV,  atomSet);

  newDW->allocateAndPut(pID_nPlus1,label->global->pID_preReloc, atomSet);
  newDW->allocateAndPut(pX_nPlus1, label->global->pX_preReloc,  atomSet);
  newDW->allocateAndPut(pV_nPlus1, label->global->pV_preReloc,  atomSet);

  newDW->get(pF_nb_nPlus1,    label->nonbonded->pF_nonbonded_preReloc,          atomSet);
  newDW->get(pF_eReal_nPlus1, label->electrostatic->pF_electroReal_preReloc,    atomSet);
  newDW->get(pF_eInv_nPlus1,  label->electrostatic->pF_electroInverse_preReloc, atomSet);

//  int timestep = (*simState)->getCurrentTopLevelTimeStep();
//  std::ostringstream forceOutName;
//  std::fstream forceOutFile;
//  forceOutName << "forceOutput.txt_" << std::left << Uintah::Parallel::getMPIRank();
//  forceOutFile.open(forceOutName.str().c_str(), std::fstream::out | std::fstream::app);

  ParticleSubset::iterator atomBegin, atomEnd, atom;
  atomBegin = atomSet->begin();
  atomEnd   = atomSet->end();
  for (atom = atomBegin; atom != atomEnd; ++atom)
  {
    F_nPlus1 = pF_nb_nPlus1[*atom] + pF_eReal_nPlus1[*atom] + pF_eInv_nPlus1[*atom];
//    pV_nPlus1[*atom] = (pV_n[*atom] - momentumFraction)
//                     + 0.5 * F_nPlus1 * d_dt * forceNorm * massInv;

    pV_nPlus1[*atom] = (pV_n[*atom])
                     + 0.5 * F_nPlus1 * d_dt * forceNorm * massInv;


    kineticEnergy += atomMass * pV_nPlus1[*atom].length2();
    totalMomentum += atomMass * pV_nPlus1[*atom];
    totalMass     += atomMass;

    pV_nPlus1[*atom]  +=  0.5 * F_nPlus1 * d_dt * forceNorm * massInv;

    pX_nPlus1[*atom]   =  pX_n[*atom] + d_dt * velocNorm * pV_nPlus1[*atom];
    pID_nPlus1[*atom]  =  pID_n[*atom];

//    forceFileLock.lock();
//    forceOutFile << "t: " << std::setw(5) << std::right << std::fixed
//               << timestep
//               << " Source Atom: " << std::setw(8) << std::right << std::fixed
//               << pID_nPlus1[*atom]
//               << " F: " << pF_nb_nPlus1[*atom]
//               << " V: " << pV_nPlus1[*atom]
//               << " X: " << pX_nPlus1[*atom].asVector()
//               << " mom: " << momentumFraction
//               << " F_diff: " << F_nPlus1-pF_nb_nPlus1[*atom]
//               << std::endl;
//    forceFileLock.unlock();
  } // Loop over atom subset
}

void velocityVerlet::firstIntegratePatch(const Patch*           workPatch,
                                         const int              localMDMaterialIndex,
                                          DataWarehouse*        oldDW,
                                          DataWarehouse*        newDW,
                                    const SimulationStateP*     simState,
                                    const MDLabel*              label,
                                          double&               kineticEnergy,
                                          double&               totalMass,
                                          SCIRun::Vector&       totalMomentum)
{
//  SCIRun::Vector momentumFraction = d_previousMomentum/d_previousMass;
  // Normalization constants
  double forceNorm = 41.84;
  double velocNorm = 1.0e-5;

  SCIRun::Vector F_nPlus1;
  double atomMass = (*simState)->getMDMaterial(localMDMaterialIndex)->getMass();
  double globalAtomIndex = (*simState)->getMDMaterial(localMDMaterialIndex)->getDWIndex();

  double massInv  = 1.0/atomMass;


  ParticleSubset* atomSet = oldDW->getParticleSubset(globalAtomIndex, workPatch);
  constParticleVariable<long64> pID_n;
       ParticleVariable<long64> pID_nPlus1;
  constParticleVariable<Point>  pX_n;
       ParticleVariable<Point>  pX_nPlus1;
  constParticleVariable<SCIRun::Vector> pV_n;
       ParticleVariable<SCIRun::Vector> pV_nPlus1;

  constParticleVariable<SCIRun::Vector> pF_nb_nPlus1,
                                        pF_eReal_nPlus1,
                                        pF_eInv_nPlus1;

  oldDW->get(pID_n, label->global->pID, atomSet);
  oldDW->get(pX_n,  label->global->pX,  atomSet);
  oldDW->get(pV_n,  label->global->pV,  atomSet);

  newDW->allocateAndPut(pID_nPlus1,label->global->pID_preReloc, atomSet);
  newDW->allocateAndPut(pX_nPlus1, label->global->pX_preReloc,  atomSet);
  newDW->allocateAndPut(pV_nPlus1, label->global->pV_preReloc,  atomSet);

  newDW->get(pF_nb_nPlus1,    label->nonbonded->pF_nonbonded_preReloc,          atomSet);
  newDW->get(pF_eReal_nPlus1, label->electrostatic->pF_electroReal_preReloc,    atomSet);
  newDW->get(pF_eInv_nPlus1,  label->electrostatic->pF_electroInverse_preReloc, atomSet);

  // Loop to spit out the total integrated quantities before reloc:  TODO delete this
//  int timestep = (*simState)->getCurrentTopLevelTimeStep();
//  forceFileLock.lock();
//  std::ostringstream forceOutName;
//  std::fstream forceOutFile;
//  forceOutName << "forceOutput.txt_" << std::left << Uintah::Parallel::getMPIRank();
//  forceOutFile.open(forceOutName.str().c_str(), std::fstream::out | std::fstream::app);

  ParticleSubset::iterator atomBegin, atomEnd, atom;
  atomBegin = atomSet->begin();
  atomEnd   = atomSet->end();
  for (atom = atomBegin; atom != atomEnd; ++atom)
  {
    F_nPlus1 = pF_nb_nPlus1[*atom] + pF_eReal_nPlus1[*atom] + pF_eInv_nPlus1[*atom];
//    pV_nPlus1[*atom] = (pV_n[*atom] - momentumFraction)
//                     + 0.5 * F_nPlus1 * d_dt * forceNorm * massInv;
//
    pV_nPlus1[*atom] = (pV_n[*atom])
                     + 0.5 * F_nPlus1 * d_dt * forceNorm * massInv;
    kineticEnergy += atomMass * pV_nPlus1[*atom].length2();
    totalMomentum += atomMass * pV_nPlus1[*atom];
    totalMass     += atomMass;

    pX_nPlus1[*atom]   =  pX_n[*atom] + d_dt * velocNorm * pV_nPlus1[*atom];
    pID_nPlus1[*atom]  =  pID_n[*atom];

//    forceOutFile << "t: " << std::setw(5) << std::right << std::fixed
//               << timestep
//               << " Source Atom: " << std::setw(8) << std::right << std::fixed
//               << pID_nPlus1[*atom]
//               << " F: " << pF_nb_nPlus1[*atom]
//               << " V: " << pV_nPlus1[*atom]
//               << " X: " << pX_nPlus1[*atom].asVector()
//               << " mom: " << momentumFraction
//               << " F_diff: " << F_nPlus1-pF_nb_nPlus1[*atom]
//               << std::endl;

  } // Loop over atom subset
//  forceFileLock.unlock();
}
void velocityVerlet::firstIntegrate(const PatchSubset*          patches,
                                    const MaterialSubset*       localMDMaterialIndex,
                                          DataWarehouse*        oldDW,
                                          DataWarehouse*        newDW,
                                    const SimulationStateP*     simState,
                                    const MDLabel*              label)
{
  SCIRun::Vector momentumFraction = d_previousMomentum/d_previousMass;
  // Accumulators for reduction variables
  double          totalMass     = 0.0;
  double          kineticEnergy = 0.0;
  Uintah::Matrix3 kineticStress = MDConstants::M3_0;
  SCIRun::Vector  totalMomentum = MDConstants::V_ZERO;

  // Normalization constants
  double forceNorm = 41.84;
  double velocNorm = 1.0e-5;
  double normKE    = 0.5*0.001*(1.0/4184.0);

  // Temporary vectors to avoid inner loop temp creation
  SCIRun::Vector F_nPlus1;
  int numPatches = patches->size();
  int numTypes   = localMDMaterialIndex->size();

  for (int patchIndex = 0; patchIndex < numPatches; ++patchIndex)
  {
    const Patch* currPatch = patches->get(patchIndex);
    for (int typeIndex = 0; typeIndex < numTypes; ++typeIndex)
    {
      int       atomType    =   localMDMaterialIndex->get(typeIndex);
      double    atomMass    =   (*simState)->getMDMaterial(typeIndex)->getMass();
      double    massInv     =   1.0/atomMass;

      ParticleSubset* atomSet = oldDW->getParticleSubset(atomType, currPatch);

      constParticleVariable<long64>         pID_n;
           ParticleVariable<long64>         pID_nPlus1;
      constParticleVariable<Point>          pX_n;
           ParticleVariable<Point>          pX_nPlus1;
      constParticleVariable<SCIRun::Vector> pV_n;
           ParticleVariable<SCIRun::Vector> pV_nPlus1;
      constParticleVariable<SCIRun::Vector> pF_nb_nPlus1, pF_eReal_nPlus1, pF_eInv_nPlus1;

      oldDW->get(pID_n,label->global->pID,atomSet);
      oldDW->get(pX_n, label->global->pX, atomSet);
      oldDW->get(pV_n, label->global->pV, atomSet);

      newDW->allocateAndPut(pID_nPlus1,label->global->pID_preReloc, atomSet);
      newDW->allocateAndPut(pX_nPlus1, label->global->pX_preReloc,  atomSet);
      newDW->allocateAndPut(pV_nPlus1, label->global->pV_preReloc,  atomSet);

      newDW->get(pF_nb_nPlus1,    label->nonbonded->pF_nonbonded_preReloc, atomSet);
      newDW->get(pF_eReal_nPlus1, label->electrostatic->pF_electroReal_preReloc, atomSet);
      newDW->get(pF_eInv_nPlus1,  label->electrostatic->pF_electroInverse_preReloc, atomSet);

      ParticleSubset::iterator atomBegin, atomEnd, atom;
      atomBegin = atomSet->begin();
      atomEnd   = atomSet->end();
      for (atom = atomBegin; atom != atomEnd; ++atom)
      {
        F_nPlus1 = pF_nb_nPlus1[*atom] + pF_eReal_nPlus1[*atom] + pF_eInv_nPlus1[*atom];
        pV_nPlus1[*atom] = (pV_n[*atom] - momentumFraction)
                         + 0.5 * F_nPlus1 * d_dt * forceNorm * massInv;

        kineticEnergy += atomMass * pV_nPlus1[*atom].length2();
        totalMomentum += atomMass * pV_nPlus1[*atom];
        totalMass     += atomMass;

        pX_nPlus1[*atom]   =  pX_n[*atom] + d_dt * velocNorm * pV_nPlus1[*atom];
        pID_nPlus1[*atom]  =  pID_n[*atom];
      } // Loop over atom subset
      ParticleSubset* delset = scinew ParticleSubset(0, atomType, currPatch);
      newDW->deleteParticles(delset);
    } // Loop over atom type
  } // Loop over patch
  kineticEnergy *= normKE;
  const Level* level = getLevel(patches);
  newDW->put( sum_vartype(kineticEnergy),    label->global->rKineticEnergy, level);
  newDW->put( sum_vartype(totalMass),        label->global->rTotalMass    , level);
  newDW->put( sumvec_vartype(totalMomentum), label->global->rTotalMomentum, level);
  newDW->put( matrix_sum(kineticStress),     label->global->rKineticStress, level);
} // velocityVerlet::firstIntegrate

void velocityVerlet::integrate(     const PatchSubset*          patches,
                                    const MaterialSubset*       atomTypes,
                                          DataWarehouse*        oldDW,
                                          DataWarehouse*        newDW,
                                    const SimulationStateP*     simState,
                                    const MDLabel*              label)
{
  SCIRun::Vector momentumFraction = d_previousMomentum/d_previousMass;
  // Accumulators for reduction variables
  double          totalMass     = 0.0;
  double          kineticEnergy = 0.0;
  Uintah::Matrix3 kineticStress = MDConstants::M3_0;
  SCIRun::Vector  totalMomentum = MDConstants::V_ZERO;

  // Normalization constants
  double forceNorm = 41.84;
  double velocNorm = 1.0e-5;
  double normKE    = 0.5*0.001*(1.0/4184.0);

  // Temporary vectors to avoid inner loop temp creation
  SCIRun::Vector F_nPlus1;
  int numPatches = patches->size();
  int numTypes   = atomTypes->size();

  for (int patchIndex = 0; patchIndex < numPatches; ++patchIndex)
  {
    const Patch* currPatch = patches->get(patchIndex);
    for (int typeIndex = 0; typeIndex < numTypes; ++typeIndex)
    {
      int       atomType    =   atomTypes->get(typeIndex);
      double    atomMass    =   (*simState)->getMDMaterial(typeIndex)->getMass();
      double    massInv     =   1.0/atomMass;

      ParticleSubset* atomSet = oldDW->getParticleSubset(atomType, currPatch);

      constParticleVariable<long64>         pID_n;
           ParticleVariable<long64>         pID_nPlus1;
      constParticleVariable<Point>          pX_n;
           ParticleVariable<Point>          pX_nPlus1;
      constParticleVariable<SCIRun::Vector> pV_n;
           ParticleVariable<SCIRun::Vector> pV_nPlus1;
      constParticleVariable<SCIRun::Vector> pF_nb_nPlus1, pF_eReal_nPlus1, pF_eInv_nPlus1;

      oldDW->get(pID_n,label->global->pID,atomSet);
      oldDW->get(pX_n, label->global->pX, atomSet);
      oldDW->get(pV_n, label->global->pV, atomSet);

      newDW->allocateAndPut(pID_nPlus1,label->global->pID_preReloc, atomSet);
      newDW->allocateAndPut(pX_nPlus1, label->global->pX_preReloc,  atomSet);
      newDW->allocateAndPut(pV_nPlus1, label->global->pV_preReloc,  atomSet);

      newDW->get(pF_nb_nPlus1,    label->nonbonded->pF_nonbonded_preReloc, atomSet);
      newDW->get(pF_eReal_nPlus1, label->electrostatic->pF_electroReal_preReloc, atomSet);
      newDW->get(pF_eInv_nPlus1,  label->electrostatic->pF_electroInverse_preReloc, atomSet);

      ParticleSubset::iterator atomBegin, atomEnd, atom;
      atomBegin = atomSet->begin();
      atomEnd   = atomSet->end();
      for (atom = atomBegin; atom != atomEnd; ++atom)
      {
        F_nPlus1 = pF_nb_nPlus1[*atom] + pF_eReal_nPlus1[*atom] + pF_eInv_nPlus1[*atom];
        pV_nPlus1[*atom] = (pV_n[*atom] - momentumFraction)
                         + 0.5 * F_nPlus1 * d_dt * forceNorm * massInv;

        kineticEnergy += atomMass * pV_nPlus1[*atom].length2();
        totalMomentum += atomMass * pV_nPlus1[*atom];
        totalMass     += atomMass;

        pV_nPlus1[*atom]  +=  0.5 * F_nPlus1 * d_dt * forceNorm * massInv;

        pX_nPlus1[*atom]   =  pX_n[*atom] + d_dt * velocNorm * pV_nPlus1[*atom];
        pID_nPlus1[*atom]  =  pID_n[*atom];
      } // Loop over atom subset
      ParticleSubset* delset = scinew ParticleSubset(0, atomType, currPatch);
      newDW->deleteParticles(delset);
    } // Loop over atom type
  } // Loop over patch
  kineticEnergy *= normKE;
  const Level* level = getLevel(patches);
  newDW->put( sum_vartype(kineticEnergy),    label->global->rKineticEnergy, level);
  newDW->put( sum_vartype(totalMass),        label->global->rTotalMass    , level);
  newDW->put( sumvec_vartype(totalMomentum), label->global->rTotalMomentum, level);
  newDW->put( matrix_sum(kineticStress),     label->global->rKineticStress, level);
} // velocityVerlet::integrate
