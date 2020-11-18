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

// ContactStressDependent.cc
// One of the derived Dissolution classes.
//
// This dissolution model generates a rate of dissolution that is proportional
// to the difference between the stress at an interface and a user specified 
// threshold stress.
// "rate of dissolution" is the velocity with which a surface is removed.
// In this model, dissolution occurs if the following criteria are met:
// 1.  The "master_material" (the material being dissolved), and at least
//     one of the materials in the "materials" list is present.
// 2.  The pressure exceeds the "thresholdPressure"
// The dissolution rate is converted to a rate of mass decrease which is
// then applied to identified surface particles in 
// interpolateToParticlesAndUpdate

#include <CCA/Components/MPM/Materials/Dissolution/ContactStressDependent.h>
#include <CCA/Components/MPM/Materials/MPMMaterial.h>
#include <CCA/Components/MPM/Core/MPMLabel.h>
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

ContactStressDependent::ContactStressDependent(const ProcessorGroup* myworld,
                                 ProblemSpecP& ps, MaterialManagerP& d_sS, 
                                 MPMLabel* Mlb)
  : Dissolution(myworld, Mlb, ps)
{
  // Constructor
  d_materialManager = d_sS;
  lb = Mlb;
  ps->require("masterModalID",        d_masterModalID);
  ps->require("InContactWithModalID", d_inContactWithModalID);
  ps->require("Ao_mol_cm2-us",        d_Ao);
  ps->require("Ea_ug-cm2_us2-mol",    d_Ea);
  ps->require("R_ug-cm2_us2-mol-K",   d_R);
  ps->require("Vm_cm3_mol",           d_Vm);
  ps->require("StressThreshold",      d_StressThresh);
  ps->getWithDefault("Temperature",   d_temperature, 300.0);
}

ContactStressDependent::~ContactStressDependent()
{
}

void ContactStressDependent::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP dissolution_ps = ps->appendChild("dissolution");
  dissolution_ps->appendElement("type",         "contactStressDependent");
  dissolution_ps->appendElement("masterModalID",        d_masterModalID);
  dissolution_ps->appendElement("InContactWithModalID", d_inContactWithModalID);
  dissolution_ps->appendElement("Ao_mol_cm2-us",        d_Ao);
  dissolution_ps->appendElement("Ea_ug-cm2_us2-mol",    d_Ea);
  dissolution_ps->appendElement("R_ug-cm2_us2-mol-K",   d_R);
  dissolution_ps->appendElement("Vm_cm3_mol",           d_Vm);
  dissolution_ps->appendElement("StressThreshold",      d_StressThresh);
  dissolution_ps->appendElement("Temperature",          d_temperature);
}

void ContactStressDependent::computeMassBurnFraction(const ProcessorGroup*,
                                              const PatchSubset* patches,
                                              const MaterialSubset* matls,
                                              DataWarehouse* old_dw,
                                              DataWarehouse* new_dw)
{
   int numMatls = d_materialManager->getNumMatls("MPM");
   ASSERTEQ(numMatls, matls->size());

  if(d_phase=="hold"){
   for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    Vector dx = patch->dCell();
    double area = dx.x()*dx.y();

    Ghost::GhostType  gnone = Ghost::None;

    // Retrieve necessary data from DataWarehouse
    std::vector<constNCVariable<double> > gmass(numMatls),gvolume(numMatls);
    std::vector<constNCVariable<Vector> > gContactForce(numMatls);
    std::vector<constNCVariable<double> > gSurfaceArea(numMatls);
    std::vector<NCVariable<double> >  massBurnRate(numMatls);
    std::vector<NCVariable<double> >  dLdt(numMatls);
    constNCVariable<double> NC_CCweight;
    std::vector<bool> masterMatls(numMatls);
    std::vector<bool> inContactWithMatls(numMatls);
    old_dw->get(NC_CCweight,  lb->NC_CCweightLabel,0, patch, gnone,0);

    for(int m=0;m<matls->size();m++){
      int dwi = matls->get(m);
      new_dw->get(gmass[m],     lb->gMassLabel,           dwi, patch, gnone, 0);
      new_dw->get(gvolume[m],   lb->gVolumeLabel,         dwi, patch, gnone, 0);
      new_dw->get(gContactForce[m],
                                lb->gLSContactForceLabel, dwi, patch, gnone, 0);
      new_dw->get(gSurfaceArea[m],
                                lb->gSurfaceAreaLabel,    dwi, patch, gnone, 0);

      new_dw->getModifiable(massBurnRate[m],
                                lb->massBurnFractionLabel,dwi, patch);
      new_dw->getModifiable(dLdt[m],
                                lb->dLdtDissolutionLabel, dwi, patch);

      MPMMaterial* mat=(MPMMaterial *) d_materialManager->getMaterial("MPM", m);
      if(mat->getModalID()==d_masterModalID){
        mat->setNeedSurfaceParticles(true);
        masterMatls[m]=true;
      } else{
        masterMatls[m]=false;
      }

      if(mat->getModalID()==d_inContactWithModalID) {
        inContactWithMatls[m]=true;
      } else{
        inContactWithMatls[m]=false;
      }
    }

    for(int m=0; m < numMatls; m++){
     if(masterMatls[m]){
      int md=m;

      // Extra factor of 8.0 in the following equation is there to
      // account for the NC_CCweight. This could still use some fine tuning.
      // In this stress dependent version of the model, dL_dt and rate are
      // multiplied by a stress difference below.
      double dL_dt = (0.75*M_PI)
                   * ((d_Vm*d_Vm)*d_Ao)/(d_R*d_temperature)
                   * exp(-d_Ea/(d_R*d_temperature))
                   * 8.0*3.1536e19*d_timeConversionFactor;
      double rate = 0.25*dL_dt*area;
//      int numNodesMBRGT0 = 0;
//      double mBRSum = 0.;
//      double normtrac_mean = 0.;
//      double surfArea_mean = 0.;
//      double contForceDis = 0.;
//      double contForceTot = 0.;
      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
        IntVector c = *iter;

        double sumMass=0.0;
        int inContactMatl=-999;
        for(int n = 0; n < numMatls; n++){
          if(n==md || inContactWithMatls[n]) {
            sumMass+=gmass[n][c]; 
          }
          if(n!=md && inContactWithMatls[n] && gmass[n][c] > 1.e-100) {
            inContactMatl = n;
          }
        }

        if(gmass[md][c] > 1.e-100  && inContactMatl >= 0) {
          double surfArea = max(gSurfaceArea[md][c] + 
                                gSurfaceArea[inContactMatl][c], 1.e-2*area);
          double normtrac_ave = -1.*(gContactForce[md][c].length() + 
                                     gContactForce[inContactMatl][c].length())
                                      /surfArea;
          if( gSurfaceArea[md][c] > 1.e-3*area &&
              gSurfaceArea[inContactMatl][c] > 1.e-3*area &&
             -normtrac_ave > d_StressThresh){   // Compressive stress is neg
            double rho = gmass[md][c]/gvolume[md][c];
            double stressDiff = (-normtrac_ave - d_StressThresh);
            massBurnRate[md][c] += NC_CCweight[c]*rate*stressDiff*rho;
            dLdt[md][c] += NC_CCweight[c]*dL_dt*stressDiff;
//           cout << "c = " << c << endl;
//           cout << "gContactForce[md][c] = " << gContactForce[md][c] << endl;
//           cout << "gContactForce[inContactMatl][c] = " 
//                <<  gContactForce[inContactMatl][c] << endl;
//            cout << "normtrac_ave = " << normtrac_ave << endl;
//            cout << "surfArea = " << surfArea << endl;
//            cout << "gSurfaceArea["<<md<<"]["<<c<<"] = " 
//                 <<  gSurfaceArea[md][c] << endl;
//            cout << "gSurfaceArea["<<inContactMatl<<"]["<<c<<"] = " 
//                 <<  gSurfaceArea[inContactMatl][c] << endl;
//          mBRSum += massBurnRate[md][c];
//          surfArea_mean += surfArea;
//          contForceDis += normtrac_ave*surfArea;
//          cout << "mBR["<<md<<"]["<<c<<"] = " << massBurnRate[md][c] << endl;
//          cout << "NC_CCweight["<<c<<"] = " << NC_CCweight[c] << endl;
//            numNodesMBRGT0++;
//            normtrac_mean += normtrac_ave;
          } // Stress threshold
        } // mass is present
      } // nodes
//      if(numNodesMBRGT0 > 0){
//        cout << "md = " << md << endl;
//        cout << "numNodesMBRGT0 = " << numNodesMBRGT0 << endl;
//        cout << "mBRSum =" << mBRSum << endl;
//        cout << "normtrac_mean = " << normtrac_mean/((double) numNodesMBRGT0) << endl;
//        cout << "surfArea_total = " << surfArea_mean << endl;
//        cout << "surfArea_mean = " << surfArea_mean/((double) numNodesMBRGT0) << endl;
//        cout << "contForceDis = " << contForceDis << endl;
//        cout << "contForceTot = " << contForceTot << endl;
//      }
     } // endif a masterMaterial
    } // materials
  } // patches
 } // if hold
}

void ContactStressDependent::addComputesAndRequiresMassBurnFrac(
                                                      SchedulerP & sched,
                                                      const PatchSet* patches,
                                                      const MaterialSet* ms)
{
  Task * t = scinew Task("ContactStressDependent::computeMassBurnFraction", 
                      this, &ContactStressDependent::computeMassBurnFraction);
  
  const MaterialSubset* mss = ms->getUnion();
  MaterialSubset* z_matl = scinew MaterialSubset();
  z_matl->add(0);
  z_matl->addReference();

  t->requires(Task::NewDW, lb->gMassLabel,               Ghost::None);
  t->requires(Task::NewDW, lb->gVolumeLabel,             Ghost::None);
  t->requires(Task::NewDW, lb->gSurfaceAreaLabel,        Ghost::None);
  t->requires(Task::NewDW, lb->gLSContactForceLabel,     Ghost::None);
  t->requires(Task::OldDW, lb->NC_CCweightLabel,z_matl,  Ghost::None);

  t->modifies(lb->massBurnFractionLabel, mss);
  t->modifies(lb->dLdtDissolutionLabel,  mss);

  sched->addTask(t, patches, ms);

  if (z_matl->removeReference())
    delete z_matl;
}
