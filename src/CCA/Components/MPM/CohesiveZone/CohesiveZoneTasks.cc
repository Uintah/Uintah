/*
 * The MIT License
 *
 * Copyright (c) 1997-2021 The University of Utah
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
#include <CCA/Components/MPM/CohesiveZone/CohesiveZoneTasks.h>
#include <CCA/Components/MPM/CohesiveZone/CZMaterial.h>
#include <CCA/Components/MPM/Materials/MPMMaterial.h>
#include <CCA/Components/MPM/Core/MPMLabel.h>
#include <CCA/Components/MPM/Core/CZLabel.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/LinearInterpolator.h>
#include <Core/Grid/DbgOutput.h>
#include <fstream>

using namespace Uintah;
using namespace std;

static DebugStream cout_doing("MPM", false);

//______________________________________________________________________
//  Reference: N. P. Daphalapukar, Hongbing Lu, Demir Coker, Ranga Komanduri,
// " Simulation of dynamic crack growth using the generalized interpolation material
// point (GIMP) method," Int J. Fract, 2007, 143:79-102
//______________________________________________________________________
CohesiveZoneTasks::CohesiveZoneTasks(MaterialManagerP& ss, MPMFlags* flags)
{
  d_lb = scinew MPMLabel();
  d_Cl = scinew CZLabel();

  d_flags = flags;

  if(flags->d_8or27==8){
    NGP=1;
    NGN=1;
  } else{
    NGP=2;
    NGN=2;
  }

  d_materialManager = ss;
}

CohesiveZoneTasks::~CohesiveZoneTasks()
{
  delete d_lb;
  delete d_Cl;
}
//______________________________________________________________________

void CohesiveZoneTasks::scheduleAddCohesiveZoneForces(SchedulerP& sched,
                                              const PatchSet* patches,
                                              const MaterialSubset* mpm_matls,
                                              const MaterialSubset* cz_matls,
                                              const MaterialSet* matls)
{
  if (!d_flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                           getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,
                "CohesiveZoneTasks::scheduleAddCohesiveZoneForces");

  Task* t = scinew Task("CohesiveZoneTasks::addCohesiveZoneForces",
                        this,&CohesiveZoneTasks::addCohesiveZoneForces);

  Ghost::GhostType  gan = Ghost::AroundNodes;
  Ghost::GhostType  gac = Ghost::AroundCells;
  t->requires(Task::OldDW, d_lb->pXLabel,                   cz_matls, gan,NGP);
  t->requires(Task::NewDW, d_Cl->czForceLabel_preReloc,     cz_matls, gan,NGP);
  t->requires(Task::NewDW, d_Cl->czTopMatLabel_preReloc,    cz_matls, gan,NGP);
  t->requires(Task::NewDW, d_Cl->czBotMatLabel_preReloc,    cz_matls, gan,NGP);
  t->requires(Task::NewDW, d_lb->gMassLabel,                mpm_matls,gac,NGN);

  t->modifies(d_lb->gExternalForceLabel, mpm_matls);

  sched->addTask(t, patches, matls);
}

void CohesiveZoneTasks::addCohesiveZoneForces(const ProcessorGroup*,
                                         const PatchSubset* patches,
                                         const MaterialSubset* ,
                                         DataWarehouse* old_dw,
                                         DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing, 
              "Doing CohesiveZoneTasks::addCohesiveZoneForces");

    ParticleInterpolator* interpolator = scinew LinearInterpolator(patch);
    vector<IntVector> ni(interpolator->size());
    vector<double> S(interpolator->size());

    Ghost::GhostType  gan = Ghost::AroundNodes;
    Ghost::GhostType  gac = Ghost::AroundCells;
    unsigned int numMPMMatls = d_materialManager->getNumMatls( "MPM" );

    std::vector<NCVariable<Vector> > gext_force(numMPMMatls);
    std::vector<constNCVariable<double> > gmass(numMPMMatls);
    for(unsigned int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = 
                      (MPMMaterial*) d_materialManager->getMaterial( "MPM", m);
      int dwi = mpm_matl->getDWIndex();

      new_dw->getModifiable(gext_force[m],d_lb->gExternalForceLabel,dwi, patch);
      new_dw->get(gmass[m],               d_lb->gMassLabel, dwi, patch,gac,NGN);
    }

    unsigned int numCZMatls=d_materialManager->getNumMatls( "CZ" );
    for(unsigned int m = 0; m < numCZMatls; m++){
      CZMaterial* cz_matl = 
                         (CZMaterial*) d_materialManager->getMaterial( "CZ", m);
      int dwi = cz_matl->getDWIndex();

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch,
                                                       gan, NGP, d_lb->pXLabel);

      // Get the arrays of particle values to be changed
      constParticleVariable<Point> czx;
      constParticleVariable<Vector> czforce;
      constParticleVariable<int> czTopMat, czBotMat;

      old_dw->get(czx,          d_lb->pXLabel,                          pset);
      new_dw->get(czforce,      d_Cl->czForceLabel_preReloc,            pset);
      new_dw->get(czTopMat,     d_Cl->czTopMatLabel_preReloc,           pset);
      new_dw->get(czBotMat,     d_Cl->czBotMatLabel_preReloc,           pset);

      // Loop over particles
      for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++){
        particleIndex idx = *iter;

        Matrix3 size(0.1,0.,0.,0.,0.1,0.,0.,0.,0.1);

        // Get the node indices that surround the cell
        int NN = interpolator->findCellAndWeights(czx[idx],ni,S,size);

        int TopMat = czTopMat[idx];
        int BotMat = czBotMat[idx];

        double totMassTop = 0.;
        double totMassBot = 0.;
//        double sumSTop = 0.;
//        double sumSBot = 0.;

        for (int k = 0; k < NN; k++) {
          IntVector node = ni[k];
          totMassTop += S[k]*gmass[TopMat][node];
          totMassBot += S[k]*gmass[BotMat][node];
#if 0
          if(gmass[TopMat][node]>d_SMALL_NUM_MPM){
            sumSTop     += S[k];
          }
          if(gmass[BotMat][node]>d_SMALL_NUM_MPM){
            sumSBot     += S[k];
          }
#endif
        }

        // This currently contains three methods for distributing the CZ force
        // to the nodes.
        // The first of these distributes the force from the CZ
        // to the nodes based on a distance*mass weighting.  
        // The second distributes the force to the nodes that have mass,
        // but only uses distance weighting.  So, a node that is near the CZ
        // but relatively far from particles may get a large acceleration
        // compared to other nodes, thereby inducing a velocity gradient.
        // The third simply does a distance weighting from the CZ to the nodes.
        // For this version, it is possible that nodes with no material mass
        // will still acquire force from the CZ, leading to ~infinite
        // acceleration, and thus, badness.

        // Accumulate the contribution from each surrounding vertex
        for (int k = 0; k < NN; k++) {
          IntVector node = ni[k];
          if(patch->containsNode(node)) {
            // Distribute force according to material mass on the nodes
            // to get an approximately equal contribution to the acceleration
            gext_force[BotMat][node] += czforce[idx]*S[k]*gmass[BotMat][node]
                                                                 /totMassBot;
            gext_force[TopMat][node] -= czforce[idx]*S[k]*gmass[TopMat][node]
                                                                 /totMassTop;

//            gext_force[BotMat][node] += czforce[idx]*S[k]/sumSBot;
//            gext_force[TopMat][node] -= czforce[idx]*S[k]/sumSTop;

//            gext_force[BotMat][node] = gext_force[BotMat][node]
//                                     + czforce[idx] * S[k];
//            gext_force[TopMat][node] = gext_force[TopMat][node]
//                                     - czforce[idx] * S[k];
          }
        }
      }
#if 0
      // This is debugging output which is being left in for now (5/10/18)
      // as it may be helpful in generating figures for reports and papers.
      Vector sumForceTop = Vector(0.);
      Vector sumForceBot = Vector(0.);
      for(NodeIterator iter=patch->getExtraNodeIterator();
                       !iter.done();iter++){
        IntVector c = *iter;
        if(gext_force[1][c].length() > 1.e-100){
           cout << "gEF_BM[" << c << "] = " << gext_force[1][c] 
                << ", " << gext_force[1][c]/gmass[1][c] << endl;
           sumForceBot += gext_force[1][c];
        }
        if(gext_force[2][c].length() > 1.e-100){
           cout << "gEF_BM[" << c << "] = " << gext_force[2][c]
                << ", " << gext_force[2][c]/gmass[2][c] << endl;
           sumForceTop += gext_force[2][c];
        }
        if(gext_force[1][c].length() > 1.e-100 &&
           gext_force[2][c].length() > 1.e-100){
           cout << "ratio = " << (gext_force[1][c].x()/gmass[1][c])/
                                 (gext_force[2][c].x()/gmass[2][c]) << endl;
        }
      }
      cout << "SFB = " << sumForceBot << endl;
      cout << "SFT = " << sumForceTop << endl;
#endif
    }
    delete interpolator;
  }
}

void CohesiveZoneTasks::scheduleUpdateCohesiveZones(SchedulerP& sched,
                                            const PatchSet* patches,
                                            const MaterialSubset* mpm_matls,
                                            const MaterialSubset* cz_matls,
                                            const MaterialSet* matls)
{
  if (!d_flags->doMPMOnLevel(getLevel(patches)->getIndex(),
                             getLevel(patches)->getGrid()->numLevels()))
    return;

  printSchedule(patches,cout_doing,
                "CohesiveZoneTasks::scheduleUpdateCohesiveZones");

  Task* t=scinew Task("CohesiveZoneTasks::updateCohesiveZones",
                      this, &CohesiveZoneTasks::updateCohesiveZones);

  t->requires(Task::OldDW, d_lb->delTLabel);

  Ghost::GhostType gac   = Ghost::AroundCells;
  Ghost::GhostType gnone = Ghost::None;
  t->requires(Task::NewDW, d_lb->gVelocityLabel,     mpm_matls,   gac,NGN);
  t->requires(Task::NewDW, d_lb->gMassLabel,         mpm_matls,   gac,NGN);
  t->requires(Task::OldDW, d_lb->pXLabel,            cz_matls,    gnone);
  t->requires(Task::OldDW, d_Cl->czAreaLabel,        cz_matls,    gnone);
  t->requires(Task::OldDW, d_Cl->czNormLabel,        cz_matls,    gnone);
  t->requires(Task::OldDW, d_Cl->czTangLabel,        cz_matls,    gnone);
  t->requires(Task::OldDW, d_Cl->czDispTopLabel,     cz_matls,    gnone);
  t->requires(Task::OldDW, d_Cl->czDispBottomLabel,  cz_matls,    gnone);
  t->requires(Task::OldDW, d_Cl->czSeparationLabel,  cz_matls,    gnone);
  t->requires(Task::OldDW, d_Cl->czForceLabel,       cz_matls,    gnone);
  t->requires(Task::OldDW, d_Cl->czTopMatLabel,      cz_matls,    gnone);
  t->requires(Task::OldDW, d_Cl->czBotMatLabel,      cz_matls,    gnone);
  t->requires(Task::OldDW, d_Cl->czFailedLabel,      cz_matls,    gnone);
  t->requires(Task::OldDW, d_Cl->czIDLabel,          cz_matls,    gnone);

  t->computes(d_lb->pXLabel_preReloc,           cz_matls);
  t->computes(d_Cl->czAreaLabel_preReloc,       cz_matls);
  t->computes(d_Cl->czNormLabel_preReloc,       cz_matls);
  t->computes(d_Cl->czTangLabel_preReloc,       cz_matls);
  t->computes(d_Cl->czDispTopLabel_preReloc,    cz_matls);
  t->computes(d_Cl->czDispBottomLabel_preReloc, cz_matls);
  t->computes(d_Cl->czSeparationLabel_preReloc, cz_matls);
  t->computes(d_Cl->czForceLabel_preReloc,      cz_matls);
  t->computes(d_Cl->czTopMatLabel_preReloc,     cz_matls);
  t->computes(d_Cl->czBotMatLabel_preReloc,     cz_matls);
  t->computes(d_Cl->czFailedLabel_preReloc,     cz_matls);
  t->computes(d_Cl->czIDLabel_preReloc,         cz_matls);

  sched->addTask(t, patches, matls);
}

void CohesiveZoneTasks::updateCohesiveZones(const ProcessorGroup*,
                                            const PatchSubset* patches,
                                            const MaterialSubset* ,
                                            DataWarehouse* old_dw,
                                            DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch,cout_doing, 
              "Doing CohesiveZoneTasks::updateCohesiveZones");

    // The following is adapted from "Simulation of dynamic crack growth
    // using the generalized interpolation material point (GIMP) method"
    // Daphalapurkar, N.P., et al., Int. J. Fracture, 143, 79-102, 2007.

    ParticleInterpolator* interpolator = scinew LinearInterpolator(patch);
    vector<IntVector> ni(interpolator->size());
    vector<double> S(interpolator->size());

    delt_vartype delT;
    old_dw->get(delT, d_lb->delTLabel, getLevel(patches) );

    unsigned int numMPMMatls=d_materialManager->getNumMatls( "MPM" );
    std::vector<constNCVariable<Vector> > gvelocity(numMPMMatls);
    std::vector<constNCVariable<double> > gmass(numMPMMatls);

    for(unsigned int m = 0; m < numMPMMatls; m++){
      MPMMaterial* mpm_matl = (MPMMaterial*) d_materialManager->getMaterial( "MPM",  m );
      int dwi = mpm_matl->getDWIndex();
      Ghost::GhostType  gac = Ghost::AroundCells;
      new_dw->get(gvelocity[m], d_lb->gVelocityLabel,dwi, patch, gac, NGN);
      new_dw->get(gmass[m],     d_lb->gMassLabel,        dwi, patch, gac, NGN);
    }

    unsigned int numCZMatls=d_materialManager->getNumMatls( "CZ" );
    for(unsigned int m = 0; m < numCZMatls; m++){
      CZMaterial* cz_matl = (CZMaterial*) d_materialManager->getMaterial( "CZ",  m );
      int dwi = cz_matl->getDWIndex();

      // Not populating the delset, but we need this to satisfy Relocate
      ParticleSubset* delset = scinew ParticleSubset(0, dwi, patch);
      new_dw->deleteParticles(delset);

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      // Get the arrays of particle values to be changed
      constParticleVariable<Point> czx;
      ParticleVariable<Point> czx_new;
      constParticleVariable<double> czarea;
      ParticleVariable<double> czarea_new;
      constParticleVariable<long64> czids;
      ParticleVariable<long64> czids_new;
      constParticleVariable<Vector> cznorm, cztang, czDispTop;
      ParticleVariable<Vector> cznorm_new, cztang_new, czDispTop_new;
      constParticleVariable<Vector> czDispBot, czsep, czforce;
      ParticleVariable<Vector> czDispBot_new, czsep_new, czforce_new;
      constParticleVariable<int> czTopMat, czBotMat, czFailed;
      ParticleVariable<int> czTopMat_new, czBotMat_new, czFailed_new;

      old_dw->get(czx,          d_lb->pXLabel,                         pset);
      old_dw->get(czarea,       d_Cl->czAreaLabel,                     pset);
      old_dw->get(cznorm,       d_Cl->czNormLabel,                     pset);
      old_dw->get(cztang,       d_Cl->czTangLabel,                     pset);
      old_dw->get(czDispTop,    d_Cl->czDispTopLabel,                  pset);
      old_dw->get(czDispBot,    d_Cl->czDispBottomLabel,               pset);
      old_dw->get(czsep,        d_Cl->czSeparationLabel,               pset);
      old_dw->get(czforce,      d_Cl->czForceLabel,                    pset);
      old_dw->get(czids,        d_Cl->czIDLabel,                       pset);
      old_dw->get(czTopMat,     d_Cl->czTopMatLabel,                   pset);
      old_dw->get(czBotMat,     d_Cl->czBotMatLabel,                   pset);
      old_dw->get(czFailed,     d_Cl->czFailedLabel,                   pset);

      new_dw->allocateAndPut(czx_new,      d_lb->pXLabel_preReloc,        pset);
      new_dw->allocateAndPut(czarea_new,   d_Cl->czAreaLabel_preReloc,    pset);
      new_dw->allocateAndPut(cznorm_new,   d_Cl->czNormLabel_preReloc,    pset);
      new_dw->allocateAndPut(cztang_new,   d_Cl->czTangLabel_preReloc,    pset);
      new_dw->allocateAndPut(czDispTop_new,d_Cl->czDispTopLabel_preReloc, pset);
      new_dw->allocateAndPut(czforce_new,  d_Cl->czForceLabel_preReloc,   pset);
      new_dw->allocateAndPut(czids_new,    d_Cl->czIDLabel_preReloc,      pset);
      new_dw->allocateAndPut(czTopMat_new, d_Cl->czTopMatLabel_preReloc,  pset);
      new_dw->allocateAndPut(czBotMat_new, d_Cl->czBotMatLabel_preReloc,  pset);
      new_dw->allocateAndPut(czFailed_new, d_Cl->czFailedLabel_preReloc,  pset);
      new_dw->allocateAndPut(czDispBot_new,d_Cl->czDispBottomLabel_preReloc,
                                                                          pset);
      new_dw->allocateAndPut(czsep_new,    d_Cl->czSeparationLabel_preReloc,
                                                                          pset);

      czarea_new.copyData(czarea);
      czids_new.copyData(czids);
      czTopMat_new.copyData(czTopMat);
      czBotMat_new.copyData(czBotMat);

      double sig_max = cz_matl->getCohesiveNormalStrength();
      double delta_n = cz_matl->getCharLengthNormal();
      double delta_t = cz_matl->getCharLengthTangential();
      double tau_max = cz_matl->getCohesiveTangentialStrength();
      double delta_s = delta_t;
      double delta_n_fail = cz_matl->getNormalFailureDisplacement();
      double delta_t_fail = cz_matl->getTangentialFailureDisplacement();
      bool rotate_CZs= cz_matl->getDoRotation();

      double phi_n = M_E*sig_max*delta_n;
      double phi_t = sqrt(M_E/2)*tau_max*delta_t;
      double q = phi_t/phi_n;
      // From the text following Eq. 15 in Nitin's paper it is a little hard
      // to tell what r should be, but zero seems like a reasonable value
      // based on the example problem in that paper
      double r=0.;

      // Loop over particles
      for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++){
        particleIndex idx = *iter;

        Matrix3 size(0.1,0.,0.,0.,0.1,0.,0.,0.,0.1);

        // Get the node indices that surround the cell
        int NN = interpolator->findCellAndWeights(czx[idx],ni,S,size);

        Vector velTop(0.0,0.0,0.0);
        Vector velBot(0.0,0.0,0.0);
        double massTop = 0.0;
        double massBot = 0.0;
        int TopMat = czTopMat[idx];
        int BotMat = czBotMat[idx];
        double sumSTop = 0.;
        double sumSBot = 0.;

        // Accumulate the contribution from each surrounding vertex
        for (int k = 0; k < NN; k++) {
          IntVector node = ni[k];
          if(gmass[TopMat][node]>2.e-200){
            velTop      += gvelocity[TopMat][node]* S[k];
            sumSTop     += S[k];
          }
          if(gmass[BotMat][node]>2.e-200){
            velBot      += gvelocity[BotMat][node]* S[k];
            sumSBot     += S[k];
          }
          massTop     += gmass[TopMat][node]*S[k];
          massBot     += gmass[BotMat][node]*S[k];
        }
        velTop/=(sumSTop+1.e-100);
        velBot/=(sumSBot+1.e-100);

#if 0
        // I'm not sure what this was here for in the first place,
        // but it is disabled for now
        double mass_ratio = 0.0;
        if (massBot > 0.0) {
          mass_ratio = massTop/massBot;
          mass_ratio = min(mass_ratio,1.0/mass_ratio);
        }
        else {
          mass_ratio = 0.0;
        }

        double mass_correction_factor = mass_ratio;
#endif

        // Update the cohesive zone's position and displacements
        czx_new[idx]         = czx[idx]       + .5*(velTop + velBot)*delT;
        czDispTop_new[idx]   = czDispTop[idx] + velTop*delT;
        czDispBot_new[idx]   = czDispBot[idx] + velBot*delT;
        czsep_new[idx]       = czDispTop_new[idx] - czDispBot_new[idx];

        double disp_old = czsep[idx].length();
        double disp     = czsep_new[idx].length();
        if (disp > 0.0 && rotate_CZs){
          Matrix3 Rotation;
          Matrix3 Rotation_tang;
          cz_matl->computeRotationMatrix(Rotation, Rotation_tang,
                                         cznorm[idx],czsep_new[idx]);

          cznorm_new[idx] = Rotation*cznorm[idx];
          cztang_new[idx] = Rotation_tang*cztang[idx];
        }
        else {
          cznorm_new[idx]=cznorm[idx];
          cztang_new[idx]=cztang[idx];
        }

        Vector cztang2 = Cross(cztang_new[idx],cznorm_new[idx]);

        double D_n  = Dot(czsep_new[idx],cznorm_new[idx]);
        double D_t1 = Dot(czsep_new[idx],cztang_new[idx]);
        double D_t2 = Dot(czsep_new[idx],cztang2);

        // Determine if a CZ has failed.
        double czf=0.0;
        if(czFailed[idx]>0 ){
          if(disp>=disp_old){
           czFailed_new[idx]=min(czFailed[idx]+1,1000);
          } else {
           czFailed_new[idx]=czFailed[idx];
          }
          czf =.001*((double) czFailed_new[idx]);
        }
        else if(fabs(D_n) > delta_n_fail){
          cout << "czFailed, D_n =  " << endl;
          czFailed_new[idx]=1;
        }
        else if( fabs(D_t1) > delta_t_fail){
          czFailed_new[idx]=1;
        }
        else if( fabs(D_t2) > delta_t_fail){
          czFailed_new[idx]=1;
        }
        else {
          czFailed_new[idx]=0;
        }

        double normal_stress  = (phi_n/delta_n)*exp(-D_n/delta_n)*
                              ((D_n/delta_n)*exp((-D_t1*D_t1)/(delta_t*delta_t))
                              + ((1.-q)/(r-1.))
                       *(1.-exp(-D_t1*D_t1/(delta_t*delta_t)))*(r-D_n/delta_n));

        double tang1_stress =(phi_n/delta_n)*(2.*delta_n/delta_t)*(D_t1/delta_t)
                              * (q
                              + ((r-q)/(r-1.))*(D_n/delta_n))
                              * exp(-D_n/delta_n)
                              * exp(-D_t1*D_t1/(delta_t*delta_t));

        double tang2_stress =(phi_n/delta_n)*(2.*delta_n/delta_s)*(D_t2/delta_s)
                              * (q
                              + ((r-q)/(r-1.))*(D_n/delta_n))
                              * exp(-D_n/delta_n)
                              * exp(-D_t2*D_t2/(delta_s*delta_s));

        czforce_new[idx]     = ((normal_stress*cznorm_new[idx]
                             +   tang1_stress*cztang_new[idx]
                             +   tang2_stress*cztang2)*czarea_new[idx])
                             *   (1.0 - czf);
/*
        dest << time << " " << czsep_new[idx].x() << " " << czsep_new[idx].y() << " " << czforce_new[idx].x() << " " << czforce_new[idx].y() << endl;
        if(fabs(normal_force) >= 0.0){
          cout << "czx_new " << czx_new[idx] << endl;
          cout << "czforce_new " << czforce_new[idx] << endl;
          cout << "czsep_new " << czsep_new[idx] << endl;
          cout << "czDispTop_new " << czDispTop_new[idx] << endl;
          cout << "czDispBot_new " << czDispBot_new[idx] << endl;
          cout << "velTop " << velTop << endl;
          cout << "velBot " << velBot << endl;
          cout << "delT " << delT << endl;
        }
*/
      }
    }

    delete interpolator;
  }
}

void CohesiveZoneTasks::cohesiveZoneProblemSetup(const ProblemSpecP& prob_spec, 
                                                 MPMFlags* flags)
{
  // Search for the MaterialProperties block and then get the MPM section
  ProblemSpecP mat_ps     = prob_spec->findBlockWithOutAttribute( "MaterialProperties" );
  ProblemSpecP mpm_mat_ps = mat_ps->findBlock( "MPM" );
  for( ProblemSpecP ps = mpm_mat_ps->findBlock("cohesive_zone"); ps != nullptr; ps = ps->findNextBlock("cohesive_zone") ) {

    string index("");
    ps->getAttribute("index",index);
    stringstream id(index);
    const int DEFAULT_VALUE = -1;
    int index_val = DEFAULT_VALUE;

    id >> index_val;

    if( !id ) {
      // stringstream parsing failed... on many (most) systems, the
      // original value assigned to index_val would be left
      // intact... but on some systems it inserts garbage,
      // so we have to manually restore the value.
      index_val = DEFAULT_VALUE;
    }
    // cout << "Material attribute = " << index_val << ", " << index << ", " << id << "\n";

    // Create and register as an MPM material
    CZMaterial *mat = scinew CZMaterial(ps, d_materialManager, flags);

    mat->registerParticleState( d_cohesiveZoneState,
                                d_cohesiveZoneState_preReloc );

    // When doing restart, we need to make sure that we load the materials
    // in the same order that they were initially created.  Restarts will
    // ALWAYS have an index number as in <material index = "0">.
    // Index_val = -1 means that we don't register the material by its 
    // index number.
    if (index_val > -1){
      d_materialManager->registerMaterial( "CZ", mat,index_val);
    }
    else{
      d_materialManager->registerMaterial( "CZ", mat);
    }
  }
}
