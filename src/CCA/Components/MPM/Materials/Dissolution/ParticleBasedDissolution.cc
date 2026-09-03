/*
 * Copyright © 2026 by Geocosm LLC                                   
 */

// ParticleBasedDissolution.cc
// One of the derived Dissolution classes.
//
// This dissolution model computes a change in particle mass at the
// particle level, based on a "rate of dissolution"
// where "rate of dissolution" is the velocity with which a surface is removed.
// In this model, dissolution occurs if the following criteria are met:
// 1.
// 2.
// The output of this model is pDeltaMassLabel.  This is applied to surface
// particles and comes from the outer surface.

// The dissolution rate is converted to a rate of mass decrease which is
// then applied to identified surface particles in 
// interpolateToParticlesAndUpdate

#include <CCA/Components/MPM/Materials/Dissolution/ParticleBasedDissolution.h>
#include <CCA/Components/MPM/Materials/MPMMaterial.h>
#include <CCA/Components/MPM/Core/MPMLabel.h>
#include <CCA/Components/MPM/Core/MPMFlags.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/MaterialManagerP.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <vector>

using namespace std;
using namespace Uintah;

ParticleBasedDissolution::ParticleBasedDissolution(const ProcessorGroup* myworld,
                                 ProblemSpecP& ps, MaterialManagerP& d_sS, 
                                 MPMLabel* Mlb, MPMFlags* flag)
  : Dissolution(myworld, Mlb, ps, flag)
{
  // Constructor
  d_materialManager = d_sS;
  lb = Mlb;

  ps->require("dLdt",        d_dLdt);
}

ParticleBasedDissolution::~ParticleBasedDissolution()
{
}

void ParticleBasedDissolution::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP dissolution_ps = ps->appendChild("dissolution");
  dissolution_ps->appendElement("type",         "particleBasedDissolution");
  dissolution_ps->appendElement("dLdt",          d_dLdt);
}

void ParticleBasedDissolution::computeMassBurnFraction(const ProcessorGroup*,
                                              const PatchSubset* patches,
                                              const MaterialSubset* matls,
                                              DataWarehouse* old_dw,
                                              DataWarehouse* new_dw)
{
//   int numMatls = d_materialManager->getNumMatls("MPM");
//   ASSERTEQ(numMatls, matls->size());

   Ghost::GhostType gac   = Ghost::AroundCells;
   for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    Vector dx = patch->dCell();

    ParticleInterpolator* interpolator = flag->d_interpolator->clone(patch);
    vector<IntVector> ni(interpolator->size());
    vector<double> S(interpolator->size());
    vector<Vector> d_S(interpolator->size());

    delt_vartype delT;
    old_dw->get(delT, lb->delTLabel, getLevel(patches));

    // Retrieve necessary data from DataWarehouse
    constParticleVariable<double> pmass, psurf, pvolume;
    constParticleVariable<Matrix3> psize, pcursize;
    constParticleVariable<Point> px;
    ParticleVariable<double> pdeltaMass;
    constNCVariable<Vector> gSurfNorm;

    for(int m=0;m<matls->size();m++){
      MPMMaterial* mpm_matl =
                     (MPMMaterial*) d_materialManager->getMaterial( "MPM", m);
      int dwi = mpm_matl->getDWIndex();
      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch);

      old_dw->get(px,                 lb->pXLabel,                   pset);
      old_dw->get(pmass,              lb->pMassLabel,                pset);
      old_dw->get(psize,              lb->pSizeLabel,                pset);
      old_dw->get(pvolume,            lb->pVolumeLabel,              pset);
      new_dw->get(pcursize,           lb->pCurSizeLabel,             pset);
      new_dw->get(psurf,              lb->pSurfLabel_preReloc,      pset);
      new_dw->get(gSurfNorm,          lb->gSurfNormLabel, dwi,patch,gac,2);

      new_dw->allocateAndPut(pdeltaMass,
                                      lb->pDeltaMassLabel,          pset);

      // For the 1-D test case
      // dMdt = dL_dt*area*density
      //        0.1*(0.001*0.001)*1000. = 1.e-4

      for(ParticleSubset::iterator iter = pset->begin();
                                        iter != pset->end(); iter++){
        particleIndex idx = *iter;

        if(psurf[idx] > 0){

          // The surface normal at the grid interpolated to the particle
          Vector pSN(0.0,0.0,0.0);
          // Get the node indices that surround the cell
          int NN = interpolator->findCellAndWeights(px[idx], ni, S,
                                                    pcursize[idx]);
          // Accumulate the contribution from each surrounding vertex
          for (int k = 0; k < NN; k++) {
            IntVector node = ni[k];
            pSN      += gSurfNorm[node]      * S[k];
          }
          double pSNL = pSN.length();
          // Normalize particle surface normal
          pSN /= (pSNL + 1.e-100);
          int maxDir = 0; double maxComp=fabs(pSN.x());
          for(int i = 1; i<3; i++){
            if(fabs(pSN[i])>maxComp){
              maxComp=fabs(pSN[i]);
              maxDir=i;
            }
          }
          int maxDirP1 = (maxDir+1)%3;
          int maxDirP2 = (maxDir+2)%3;

          Vector L[3];
          double Ll[3];
          double dL[3];
          double pSNdotL[3];
          Vector deltaLength;

          for(int i=0;i<3;i++){
            L[i]=Vector(psize[idx](0,i),
                        psize[idx](1,i),
                        psize[idx](2,i));
            Ll[i] = L[i].length();

            L[i]/=Ll[i];
            pSNdotL[i] = fabs(Dot(pSN,L[i]));
          }

//          double dL1overdL0 = pSNdotL[maxDirP1]/(pSNdotL[maxDir]+1.e-100);
//          double dL2overdL0 = pSNdotL[maxDirP2]/(pSNdotL[maxDir]+1.e-100);

          deltaLength = pSN*d_dLdt*delT;
          for(int i=0;i<3;i++){
            deltaLength[i]/=dx[i];
            //cout << "deltaLength[" << i << "] = " << deltaLength[i] << endl;
          }
          for(int i=0;i<3;i++){
            dL[i]=fabs(Dot(deltaLength,L[i]));
            //cout << "dL[" << i << "] = " << dL[i] << endl;
            L[i] *= max(Ll[i] -dL[i],0.);
            //cout << "L[" << i << "] = " << L[i] << endl;
          }

/*
          dL[maxDir] = d_dLdt*delT*(Ll[0]*Ll[1]*Ll[2])/
                                   (Ll[maxDirP1]*Ll[maxDirP2]
                               + dL1overdL0*Ll[maxDir]*Ll[maxDirP2]
                               + dL2overdL0*Ll[maxDir]*Ll[maxDirP1]
                               + 1.e-100);

          dL[maxDirP1] = dL1overdL0*dL[maxDir];
          dL[maxDirP2] = dL2overdL0*dL[maxDir];
          for(int i=0;i<3;i++){
          //cout << "dL[" << i << "] = " << dL[i] << endl;
          }
          L[maxDir]   *= (Ll[maxDir]   - dL[maxDir]);
          L[maxDirP1] *= (Ll[maxDirP1] - dL[maxDirP1]);
          L[maxDirP2] *= (Ll[maxDirP2] - dL[maxDirP2]);
*/

          Matrix3 psizeNew = Matrix3(L[0].x(), L[1].x(), L[2].x(),
                                     L[0].y(), L[1].y(), L[2].y(),
                                     L[0].z(), L[1].z(), L[2].z());

          double massRatio = psizeNew.Determinant()/psize[idx].Determinant();
          pdeltaMass[idx] = pmass[idx]*(1. - massRatio);
          //cout << "pdeltaMassNewWay = " << pdeltaMass[idx] << endl;

          // Need some work here
//          double pEdgeLength = dx.y(); //cbrt(pvolume[idx]);
//          double density = pmass[idx]/pvolume[idx];
//          pdeltaMass[idx] = density*pEdgeLength*pEdgeLength*d_dLdt*delT;
          //cout << "pdeltaMassOldWay = " << pdeltaMass[idx] << endl;
        } else {
          pdeltaMass[idx] = 0.0;
        }
      } // loop over particles
    } // materials
  } // patches
}

void ParticleBasedDissolution::addComputesAndRequiresMassBurnFrac(SchedulerP & sched,
                                                      const PatchSet* patches,
                                                      const MaterialSet* ms)
{
  Task * t = scinew Task("ParticleBasedDissolution::computeMassBurnFraction", 
                      this, &ParticleBasedDissolution::computeMassBurnFraction);
  
  //const MaterialSubset* mss = ms->getUnion();
  Ghost::GhostType gnone = Ghost::None;
  Ghost::GhostType gac   = Ghost::AroundCells;

  t->requiresVar(Task::OldDW, lb->delTLabel);
  t->requiresVar(Task::OldDW, lb->pXLabel,                  gnone);
  t->requiresVar(Task::OldDW, lb->pMassLabel,               gnone);
  t->requiresVar(Task::NewDW, lb->pCurSizeLabel,            gnone);
  t->requiresVar(Task::OldDW, lb->pSizeLabel,               gnone);
  t->requiresVar(Task::NewDW, lb->gSurfNormLabel,           gac, 2);


  t->computesVar(lb->pDeltaMassLabel);
//  t->requiresVar(Task::NewDW, lb->gMassLabel,               Ghost::None);
//  t->requiresVar(Task::NewDW, lb->gVolumeLabel,             Ghost::None);
//  t->requiresVar(Task::NewDW, lb->gSurfaceAreaLabel,        Ghost::None);
//  t->requiresVar(Task::OldDW, lb->NC_CCweightLabel,z_matl,  Ghost::None);

  sched->addTask(t, patches, ms);
}
