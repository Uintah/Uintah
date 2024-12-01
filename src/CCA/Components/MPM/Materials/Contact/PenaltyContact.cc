/*
 * The MIT License
 *
 * Copyright (c) 1997-2024 The University of Utah
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

#include <Core/Math/Matrix3.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/MaterialManagerP.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <CCA/Components/MPM/Core/MPMLabel.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Components/MPM/Materials/MPMMaterial.h>
#include <CCA/Components/MPM/Materials/Contact/PenaltyContact.h>
#include <vector>
#include <iostream>

using namespace Uintah;
using std::vector;
using std::string;

using namespace std;


PenaltyContact::PenaltyContact(const ProcessorGroup* myworld,
                               ProblemSpecP& ps,MaterialManagerP& d_sS,
                               MPMLabel* Mlb, MPMFlags* MFlag, int numMatls)
  : Contact(myworld, Mlb, MFlag, ps, numMatls)
{
  // Constructor
  d_vol_const=0.;
  d_oneOrTwoStep = 1;

  ps->require("mu",d_mu);

  d_materialManager = d_sS;

  if(flag->d_8or27==8){
    NGP=1;
    NGN=1;
  } else{
    NGP=2;
    NGN=2;
  }
}

PenaltyContact::~PenaltyContact()
{
  // Destructor
}

void PenaltyContact::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP contact_ps = ps->appendChild("contact");
  contact_ps->appendElement("type", "penalty");
  contact_ps->appendElement("mu",                d_mu);
  d_matls.outputProblemSpec(contact_ps);
}

void PenaltyContact::exMomInterpolated(const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw)
{
}

void PenaltyContact::exMomIntegrated(const ProcessorGroup*,
                                     const PatchSubset* patches,
                                     const MaterialSubset* matls,
                                     DataWarehouse* old_dw,
                                     DataWarehouse* new_dw)
{
  Ghost::GhostType  gnone = Ghost::None;

  int numMatls = d_materialManager->getNumMatls( "MPM" );
  ASSERTEQ(numMatls, matls->size());

  // Need access to all velocity fields at once, so store in
  // vectors of NCVariables
  std::vector<constNCVariable<double> > gmass(numMatls);
  std::vector<constNCVariable<double> > gvolume(numMatls);
  std::vector<constNCVariable<Vector> > gtrcontactforce(numMatls);
  std::vector<constNCVariable<int> >    gInContactMatl(numMatls);
  std::vector<NCVariable<Vector> >      gvelocity_star(numMatls);
  std::vector<bool> PistonMaterial(numMatls);

  delt_vartype delT;
  old_dw->get(delT, lb->delTLabel, getLevel(patches));

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

//    bool useMinNormalDv = true;
    
//    if(flag->d_doingDissolution && flag->d_currentPhase=="hold"){
//       useMinNormalDv = false;
//    }

    // Retrieve necessary data from DataWarehouse
    for(int m=0;m<matls->size();m++){
      MPMMaterial* mpm_matl =
                     (MPMMaterial*) d_materialManager->getMaterial( "MPM",  m );
      PistonMaterial[m] = mpm_matl->getIsPistonMaterial();
      int dwi = matls->get(m);
      new_dw->get(gmass[m],          lb->gMassLabel,     dwi, patch, gnone, 0);
      new_dw->get(gvolume[m],        lb->gVolumeLabel,   dwi, patch, gnone, 0);
      new_dw->get(gtrcontactforce[m],lb->gLSContactForceLabel,
                                                         dwi, patch, gnone, 0);
      new_dw->get(gInContactMatl[m], lb->gInContactMatlLabel,
                                                         dwi, patch, gnone, 0);
      new_dw->getModifiable(gvelocity_star[m], lb->gVelocityStarLabel,
                                                         dwi, patch);
    }
    Vector dx = patch->dCell();

    if(flag->d_axisymmetric){
      ostringstream warn;
      warn << "Penalty contact not implemented for axisymmetry\n";
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }

    for(NodeIterator iter = patch->getNodeIterator();!iter.done();iter++){
      IntVector c = *iter;
      Vector centerOfMassVelocity(0.,0.,0.);
      double centerOfMassMass=0.0; 
      for(int  n = 0; n < numMatls; n++){
        if(!d_matls.requested(n)) continue;
        if(gtrcontactforce[n][c].length2() > 0.0){
          centerOfMassVelocity+=gvelocity_star[n][c] * gmass[n][c];
          centerOfMassMass+= gmass[n][c]; 
        }
      }

      centerOfMassVelocity/=centerOfMassMass;

      // Loop over materials.  Only proceed if velocity field mass
      // is nonzero (not numerical noise) and the difference from
      // the centerOfMassVelocity is nonzero (More than one velocity
      // field is contributing to grid vertex).
      for(int n = 0; n < numMatls; n++){
        if(!d_matls.requested(n)) continue;
        double mass=gmass[n][c];
        if(gtrcontactforce[n][c].length2() > 0.0 &&
           !compare(mass,0.0) /*&& mass!=centerOfMassMass*/){
          // Dv is the change in velocity due to penalty forces at tracers
          Vector Dv = (gtrcontactforce[n][c]/mass)*delT;
          double normalDv = Dv.length();
          // This is an inward facing normal
          Vector normal = Dv/(normalDv+1.e-100);

          // deltaVelocity is the difference in velocity of this material
          // relative to the centerOfMassVelocity
          Vector deltaVelocity=gvelocity_star[n][c] - centerOfMassVelocity;
          // Get the normal component of deltaVelocity
          double normalDeltaVel = Dot(deltaVelocity,normal);

          // Take the minimum of these two measures of normal velocity.
          // This prevents the penalty force (normalDv) from pushing too hard,
          // and thereby creating space between the materials.  The
          // force based on the velocity (normalDeltaVel) is just enough 
          // to bring the material's velocity to the center of mass velocity.
//          if(!compare(mass-centerOfMassMass,0.0)){
//            normalDv = Min(fabs(normalDv), fabs(normalDeltaVel));
//          } else {
//            normalDv = fabs(normalDv);
//          }
//          if(useMinNormalDv){
            normalDv = Min(fabs(normalDv), fabs(normalDeltaVel));
//          } else {
//            normalDv = fabs(normalDv);
//          }

          // Change in velocity in the normal direction
          Dv = normalDv*normal;

          // Create a vector that is the normal part of deltaVelocity
          Vector normalXnormaldV = normal*normalDeltaVel;
          // Subtract the normal part from the entire deltaVelocity
          Vector dV_normalDV = deltaVelocity - normalXnormaldV;
          // Compute tangent unit vector in the direction of the
          // non-normal part of deltaVelocity
          Vector surfaceTangent = dV_normalDV/(dV_normalDV.length()+1.e-100);

#if 0
          if(fabs(Dot(surfaceTangent,normal)) > 1.e-9){
            cout << "n = "  << n << endl;
            cout << "c = "  << c << endl;
            cout << "gtrcontactforce = "  << gtrcontactforce[n][c] << endl;
            cout << "Dot(surfaceTangent,normal) = "
                 <<  Dot(surfaceTangent,normal) << endl;
            cout << "deltaVelocity = "  << deltaVelocity << endl;
            cout << "normalDeltaVel = " << normalDeltaVel << endl;
            cout << "Dv = " << Dv << endl;
            cout << "normalDv = " << normalDv << endl;
            cout << "normal = " << normal << endl;
            cout << "normalXnormaldV = " << normalXnormaldV << endl;
            cout << "dV_normalDV = " << dV_normalDV << endl;
            cout << "surfaceTangent = " << surfaceTangent << endl;
          }
#endif

          double tangentDeltaVelocity=Dot(deltaVelocity,surfaceTangent);
          double frictionCoefficient=
                    Min(d_mu,tangentDeltaVelocity/(normalDv + 1.e-100));

          if(PistonMaterial[n] || PistonMaterial[gInContactMatl[n][c]]){
           frictionCoefficient=0.0;
          }

          // Change in velocity in the tangential direction
          Dv -= surfaceTangent*frictionCoefficient*normalDv;

          // Define contact algorithm imposed strain, find maximum
          Vector epsilon=(Dv/dx)*delT;
          double epsilon_max=
            Max(fabs(epsilon.x()),fabs(epsilon.y()),fabs(epsilon.z()));
          if(!compare(epsilon_max,0.0)){
            // Scale velocity change if contact imposed strain is too large
            double ff=Min(epsilon_max,.1)/epsilon_max;
            Dv=Dv*ff;
          }
          gvelocity_star[n][c]    +=Dv;
        }   // if gtrcontactforce>0
      }     // matls
    }       // nodeiterator
  } // patches
}

void PenaltyContact::addComputesAndRequiresInterpolated(SchedulerP & sched,
                                                        const PatchSet* patches,
                                                        const MaterialSet* ms)
{
  Task * t = scinew Task("Penalty::exMomInterpolated", 
                      this, &PenaltyContact::exMomInterpolated);
  sched->addTask(t, patches, ms);
}

void PenaltyContact::addComputesAndRequiresIntegrated(SchedulerP & sched,
                                                      const PatchSet* patches,
                                                      const MaterialSet* ms) 
{
  Task * t = scinew Task("Penalty::exMomIntegrated", 
                      this, &PenaltyContact::exMomIntegrated);

  const MaterialSubset* mss = ms->getUnion();
  t->requires(Task::OldDW, lb->delTLabel);
  t->requires(Task::NewDW, lb->gMassLabel,                  Ghost::None);
  t->requires(Task::NewDW, lb->gVolumeLabel,                Ghost::None);
  t->requires(Task::NewDW, lb->gLSContactForceLabel,        Ghost::None);
  t->requires(Task::NewDW, lb->gInContactMatlLabel,         Ghost::None);

  t->modifies(             lb->gVelocityStarLabel,  mss);

  sched->addTask(t, patches, ms);
}
