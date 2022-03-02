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

// NewQuartzOvergrowth.cc
// One of the derived Dissolution classes.
//
// The dissolution rate is converted to a rate of mass decrease which is
// then applied to identified surface particles in 
// interpolateToParticlesAndUpdate

#include <CCA/Components/MPM/Materials/Dissolution/NewQuartzOvergrowth.h>
#include <CCA/Components/MPM/Materials/MPMMaterial.h>
#include <CCA/Components/MPM/Core/MPMLabel.h>
#include <CCA/Components/MPM/Core/MPMBoundCond.h>
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

NewQuartzOvergrowth::NewQuartzOvergrowth(const ProcessorGroup* myworld,
                                 ProblemSpecP& ps, MaterialManagerP& d_sS, 
                                 MPMLabel* Mlb)
  : Dissolution(myworld, Mlb, ps)
{
  // Constructor
  d_materialManager = d_sS;
  lb = Mlb;
  ps->require("masterModalID",             d_masterModalID);
  ps->require("GrowthRate_cmPerMY",        d_growthRate);
}

NewQuartzOvergrowth::~NewQuartzOvergrowth()
{
}

void NewQuartzOvergrowth::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP dissolution_ps = ps->appendChild("dissolution");
  dissolution_ps->appendElement("type",                  "NewQuartzOvergrowth");
  dissolution_ps->appendElement("masterModalID",          d_masterModalID);
  dissolution_ps->appendElement("GrowthRate_cmPerMY",     d_growthRate);
}

void NewQuartzOvergrowth::computeMassBurnFraction(const ProcessorGroup*,
                                              const PatchSubset* patches,
                                              const MaterialSubset* matls,
                                              DataWarehouse* old_dw,
                                              DataWarehouse* new_dw)
{
   int numMatls = d_materialManager->getNumMatls("MPM");
   ASSERTEQ(numMatls, matls->size());

   delt_vartype delT;
   old_dw->get(delT, lb->delTLabel, getLevel(patches) );

//   string interp_type = flags->d_interpolator_type;

//  proc0cout << "phase = " << d_phase << endl;
  if(d_phase=="dissolution"){

   for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    Ghost::GhostType  gnone = Ghost::None;

    // Retrieve necessary data from DataWarehouse
    std::vector<constNCVariable<double> > gmass(numMatls);
    std::vector<constNCVariable<Vector> > gCemVec(numMatls);
    std::vector<constNCVariable<double> > gSurfaceArea(numMatls);
    std::vector<constNCVariable<Vector> > gContactForce(numMatls);
    std::vector<NCVariable<double> >  massBurnRate(numMatls);
    std::vector<NCVariable<double> >  dLdt(numMatls);
    std::vector<NCVariable<Vector> >  gSurfNorm(numMatls);
    constNCVariable<double> NC_CCweight;
    std::vector<bool> masterMatls(numMatls);
    std::vector<double> rho(numMatls);
    old_dw->get(NC_CCweight,  lb->NC_CCweightLabel, 0, patch, gnone, 0);

    for(int m=0;m<matls->size();m++){
      int dwi = matls->get(m);
      new_dw->get(gmass[m],     lb->gMassLabel,           dwi, patch, gnone, 0);
      new_dw->get(gCemVec[m],   lb->gCemVecLabel,         dwi, patch, gnone, 0);
      new_dw->get(gSurfaceArea[m],
                                lb->gSurfaceAreaLabel,    dwi, patch, gnone, 0);
      new_dw->get(gContactForce[m],
                                lb->gLSContactForceLabel, dwi, patch, gnone, 0);
      new_dw->getModifiable(massBurnRate[m],
                                lb->massBurnFractionLabel,dwi, patch);
      new_dw->getModifiable(dLdt[m],
                                lb->dLdtDissolutionLabel, dwi, patch);
      new_dw->getModifiable(gSurfNorm[m],
                                lb->gSurfNormLabel,       dwi, patch);

      MPMMaterial* mat=(MPMMaterial *) d_materialManager->getMaterial("MPM", m);
      rho[m] = mat->getInitialDensity();
      if(mat->getModalID()==d_masterModalID){
        mat->setNeedSurfaceParticles(true);
        masterMatls[m]=true;
      } else{
        masterMatls[m]=false;
      }
    } // loop over matls to fill arrays


    // Surface motion occurs in a direction that is not normal to the surface
    // The mass added will be the projection of the surface area into the
    // direction of the surface motion.
    for(int m=0; m < numMatls; m++){
     if(masterMatls[m]){
      int md=m;
//      double dL_dt      =  -d_growthRate*3.1536e19*d_timeConversionFactor;
      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
        IntVector c = *iter;

        if(gmass[md][c] > 2.e-100 && gContactForce[md][c].length() < 1.e-8
                                  && NC_CCweight[c] < 0.2) {

          double localSurfRate = 0.1*Dot(gCemVec[md][c],gSurfNorm[md][c]);
          massBurnRate[md][c] -= rho[m]*localSurfRate*gSurfaceArea[md][c];
          dLdt[md][c] -= 0.1*gCemVec[md][c].length();
        } // mass is present
        gSurfNorm[md][c] = gCemVec[md][c]/(gCemVec[md][c].length() + 1.e-100);
      } // nodes
     } // endif a masterMaterial
    } // materials
  } // patches
 } // if dissolution
}

void NewQuartzOvergrowth::addComputesAndRequiresMassBurnFrac(
                                                      SchedulerP & sched,
                                                      const PatchSet* patches,
                                                      const MaterialSet* ms)
{
  Task * t = scinew Task("NewQuartzOvergrowth::computeMassBurnFraction", 
                      this, &NewQuartzOvergrowth::computeMassBurnFraction);
  
  const MaterialSubset* mss = ms->getUnion();
  MaterialSubset* z_matl = scinew MaterialSubset();
  z_matl->add(0);
  z_matl->addReference();

  t->requires(Task::OldDW, lb->delTLabel );
  t->requires(Task::NewDW, lb->gMassLabel,               Ghost::None);
  t->requires(Task::NewDW, lb->gCemVecLabel,             Ghost::None);
  t->requires(Task::NewDW, lb->gSurfaceAreaLabel,        Ghost::None);
  t->requires(Task::NewDW, lb->gLSContactForceLabel,     Ghost::None);
  t->requires(Task::OldDW, lb->NC_CCweightLabel,z_matl,  Ghost::None);

  t->modifies(lb->massBurnFractionLabel, mss);
  t->modifies(lb->dLdtDissolutionLabel,  mss);
  t->modifies(lb->gSurfNormLabel,        mss);

  sched->addTask(t, patches, ms);

  if (z_matl->removeReference())
    delete z_matl;
}
