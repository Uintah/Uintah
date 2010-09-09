/*

The MIT License

Copyright (c) 1997-2010 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/



// NodalSVF.cc
// This is a contact model developed by Peter Mackenzie. Details about the
// derivation can be found in "Modeling Strategies for Multiphase Drag 
// Interactions Using the Material Point Method" (Mackenzie, et al; 2010).
// Of the interaction models proposed in their paper, this particular 
// contact model for MPM in Uintah can simulate the Nodal Bang Bang method 
// OR the Nodal Smoothed Volume (SVF) Fraction Method. The Nodal Bang Bang 
// method is less expensive than Nodal SVF, but much less accurate. These 
// two methods, which are of the Node-based type, register interaction 
// proportional to the cell volume (dx*dy*dz) between two or more phases. 
// As a result, over-estimates of the interaction occur in cells where the
// interacting materials do not completely occupy the computational cell. 
// Interaction in this model is quantified by an interaction parameter, 
// mu (N/m^4), and the velocity difference between the phases.  Other 
// simple Coulomb friction or viscous fluid interaction models can be 
// substituted.

#include <CCA/Components/MPM/Contact/NodalSVFContact.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Level.h>
#include <Core/Grid/Variables/NodeIterator.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Labels/MPMLabel.h>
#include <CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <CCA/Ports/DataWarehouse.h>
#include <Core/Containers/StaticArray.h>
#include <vector>
#include <iostream>
#include <fstream>

using namespace std;
using namespace Uintah;
using namespace SCIRun;
using std::vector;

NodalSVFContact::NodalSVFContact(const ProcessorGroup* myworld,
                                   ProblemSpecP& ps, SimulationStateP& d_sS, 
                                   MPMLabel* Mlb,MPMFlags* MFlag)
  : Contact(myworld, Mlb, MFlag, ps)
{
  // Constructor
  d_sharedState = d_sS;
  ps->require("myu",d_myu);
  ps->require("use_svf",b_svf);
  lb = Mlb;
  flag = MFlag;
}

NodalSVFContact::~NodalSVFContact()
{
}

void NodalSVFContact::outputProblemSpec(ProblemSpecP& ps)
{
  ProblemSpecP contact_ps = ps->appendChild("contact");
  contact_ps->appendElement("type","nodal_svf");
  contact_ps->appendElement("myu",d_myu);
  contact_ps->appendElement("use_svf",b_svf);
  d_matls.outputProblemSpec(contact_ps);
}

void NodalSVFContact::exMomInterpolated(const ProcessorGroup*,
                                         const PatchSubset* patches,
                                         const MaterialSubset* matls,
                                         DataWarehouse*,
                                         DataWarehouse* new_dw)
{   }

void NodalSVFContact::exMomIntegrated(const ProcessorGroup*,
                                       const PatchSubset* patches,
                                       const MaterialSubset* matls,
                                       DataWarehouse* old_dw,
                                       DataWarehouse* new_dw)
{
  int numMatls = d_sharedState->getNumMPMMatls();
  ASSERTEQ(numMatls, matls->size());


  for(int p=0;p<patches->size();p++){
    
    //---------------------------------
    //Declare variable types
    //---------------------------------
    const Patch* patch = patches->get(p);
    Ghost::GhostType  gnone = Ghost::None; 
    delt_vartype delT;
    constNCVariable<double> NC_CCweight;

    double dx = patch->dCell().x();
    double dy = patch->dCell().y();
    double dz = patch->dCell().z();
    double cellVol = dx*dy*dz;
    double coeff = cellVol*d_myu;
    double factor;


    StaticArray<constNCVariable<double> > gmass(numMatls);
    StaticArray<constNCVariable<double> > gvolume(numMatls);
    StaticArray<NCVariable<double> > gSVF(numMatls);
    StaticArray<NCVariable<Vector> > gvelocity_star(numMatls);
    StaticArray<NCVariable<Vector> > gvelocity_old(numMatls);
    StaticArray<NCVariable<Vector> > gForce(numMatls);

    //---------- Retrieve necessary data from DataWarehouse ------------------------------------------------
    old_dw->get(delT, lb->delTLabel, getLevel(patches));
    old_dw->get(NC_CCweight, lb->NC_CCweightLabel, 0, patch, gnone, 0);
    
    for(int m=0;m<matls->size();m++){
      int dwi = matls->get(m);
      new_dw->get(gmass[m],lb->gMassLabel, dwi, patch, Ghost::None, 0);
      new_dw->get(gvolume[m],lb->gVolumeLabel, dwi, patch, Ghost::None, 0);
      new_dw->allocateTemporary(gSVF[m], patch, Ghost::None, 0);
      new_dw->allocateTemporary(gvelocity_old[m], patch, Ghost::None, 0);
      new_dw->allocateTemporary(gForce[m], patch, Ghost::None, 0);
      new_dw->getModifiable(gvelocity_star[m],lb->gVelocityStarLabel, dwi,patch);

      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
        IntVector c = *iter;
        gvelocity_old[m][c] = gvelocity_star[m][c];
        gSVF[m][c] =  8.0 * NC_CCweight[c] * gvolume[m][c] / cellVol;
      }
    }
    
    //cout<<"coeff="<<coeff<<endl;
    //----------- Calculate Interaction Force -----------------------------------
    for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
      IntVector c = *iter;
        
      if (b_svf==1) { factor = coeff * gSVF[0][c] * gSVF[1][c]; } 
      else          { factor = coeff; } 


      // If using the model with svf calculation, or if mass is present on 
      // both nodes, calculate a non-zero interaction force based on velocity 
      // difference and the appropriate value of "factor".
      if ((b_svf==1) || (gmass[0][c]>1.0e-100 && gmass[1][c]>1.0e-100)) { 
        
        gForce[0][c] = factor * (gvelocity_old[1][c] - gvelocity_old[0][c]);
        gForce[1][c] = factor * (gvelocity_old[0][c] - gvelocity_old[1][c]);
        
      } else {
        gForce[0][c] = Vector(0.0,0.0,0.0); gForce[1][c] = Vector(0.0,0.0,0.0);
      }
       
    //------------Calculate Updated Velocity ------------------------------------
      gvelocity_star[0][c] = gvelocity_old[0][c] + (gForce[0][c]/( 8.0 * NC_CCweight[c] * gmass[0][c])) * delT;    
      gvelocity_star[1][c] = gvelocity_old[1][c] + (gForce[1][c]/( 8.0 * NC_CCweight[c] * gmass[1][c])) * delT;    
    
    }//for nodes
  }//for patches
}//end exmomentumIntegrated


void NodalSVFContact::addComputesAndRequiresInterpolated(
                           SchedulerP & sched, const PatchSet* patches, const MaterialSet* ms) {   }

void NodalSVFContact::addComputesAndRequiresIntegrated(
                           SchedulerP & sched, const PatchSet* patches, const MaterialSet* ms) {
  Task * t = scinew Task("NodalSVFContact::exMomIntegrated", this, &NodalSVFContact::exMomIntegrated);
  
  const MaterialSubset* mss = ms->getUnion();
  t->requires(Task::OldDW, lb->delTLabel);    
  t->requires(Task::NewDW, lb->gMassLabel,Ghost::None);
  t->modifies(             lb->gVelocityStarLabel, mss);
  sched->addTask(t, patches, ms);
}
