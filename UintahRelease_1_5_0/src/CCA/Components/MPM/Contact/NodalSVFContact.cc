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


// NodalSVF_Contact.cc
// This is a contact model developed by Peter Mackenzie. Details about the
// derivation can be found in "Modeling Strategies for Multiphase Drag 
// Interactions Using the Material Point Method" (Mackenzie, et al; 2011).
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
#include <Core/Exceptions/ProblemSetupException.h>
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

NodalSVFContact:: NodalSVFContact( const ProcessorGroup*       myworld,
                                         ProblemSpecP&              ps, 
                                         SimulationStateP&        d_sS, 
                                         MPMLabel*                 Mlb,
                                         MPMFlags*                MFlag)
                             : Contact(myworld,   Mlb,  MFlag,  ps)
{  // Constructor
  d_sharedState = d_sS;
  ps->    require("myu",    d_myu);
  ps->    require("use_svf",b_svf);
  
  vector<int> materials;
  ps->    get("materials", materials);

 int numMatlsUPS=0;
 for(vector<int>::const_iterator mit(materials.begin());mit!=materials.end();mit++) {
      numMatlsUPS++;
 }
 if (numMatlsUPS>2) {
   throw ProblemSetupException(" You may only specify two materials in the input file per contact block for Nodal SVF.", __FILE__, __LINE__);
 }                                 

 lb = Mlb;       flag = MFlag;
  
}

NodalSVFContact::~NodalSVFContact( ){ }

void NodalSVFContact:: outputProblemSpec(ProblemSpecP& ps) {
  ProblemSpecP contact_ps = ps-> appendChild  ("contact");
               contact_ps->      appendElement("type"   ,"nodal_svf");
               contact_ps->      appendElement("myu"    ,d_myu);
               contact_ps->      appendElement("use_svf",b_svf);
  d_matls.outputProblemSpec(contact_ps);
}

void NodalSVFContact:: exMomInterpolated( const ProcessorGroup*         ,
                                          const PatchSubset*     patches,
                                          const MaterialSubset*    matls,
                                                DataWarehouse*          ,
                                                DataWarehouse*     new_dw ) {  }

void NodalSVFContact:: exMomIntegrated( const ProcessorGroup*           ,
                                        const PatchSubset*       patches,
                                        const MaterialSubset*      matls,
                                              DataWarehouse*      old_dw,
                                              DataWarehouse*       new_dw ) {
  int numMatls=matls->size();
  int alpha=0; int beta=0; int n=0;
  for(int m=0;m<numMatls;m++){
    if((d_matls.requested(m)) && (n==0)) { alpha = matls->get(m); n++;}
    else                                 { beta  = matls->get(m); }
  }
 

  for(int p=0;p<patches->size();p++){
    
    const Patch*      patch = patches->get(p);
    Ghost::GhostType  gnone = Ghost::None; 
    delt_vartype      delT;

    double dx      = patch->dCell().x();
    double dy      = patch->dCell().y();
    double dz      = patch->dCell().z();
    double cellVol = dx*dy*dz;
    double coeff   = cellVol*d_myu;
    double factor;

                 constNCVariable<double>    NC_CCweight;
    StaticArray <constNCVariable<double> >  gmass(numMatls);
    StaticArray <constNCVariable<double> >  gvolume(numMatls);
    StaticArray <NCVariable     <double> >  gSVF(numMatls);
    StaticArray <NCVariable     <Vector> >  gvelocity_star(numMatls);
    StaticArray <NCVariable     <Vector> >  gvelocity_old(numMatls);
    StaticArray <NCVariable     <Vector> >  gForce(numMatls);

    //---------- Retrieve necessary data from DataWarehouse ------------------------------------------------
    old_dw-> get(delT,        lb->delTLabel,         getLevel(patches));
    old_dw-> get(NC_CCweight, lb->NC_CCweightLabel, 0, patch, gnone, 0);
     
    for(int m=0;m<numMatls;m++){
      int dwi = matls->get(m);
      new_dw-> get              (gmass[dwi],          lb->gMassLabel,         dwi, patch, gnone, 0);
      new_dw-> get              (gvolume[dwi],        lb->gVolumeLabel,       dwi, patch, gnone, 0);
      new_dw-> getModifiable    (gvelocity_star[dwi], lb->gVelocityStarLabel, dwi, patch);
      new_dw-> allocateTemporary(gSVF[dwi],                                        patch, gnone, 0);
      new_dw-> allocateTemporary(gvelocity_old[dwi],                               patch, gnone, 0);
      new_dw-> allocateTemporary(gForce[dwi],                                      patch, gnone, 0);
    } // for m=0:numMatls
   

    //----------- Calculate Interaction Force -----------------------------------
    for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
      
      IntVector c = *iter;
      gvelocity_old[beta] [c] = gvelocity_star[beta] [c];         
      gvelocity_old[alpha][c] = gvelocity_star[alpha][c];
      gSVF         [beta] [c] =  8.0 * NC_CCweight[c] * gvolume[beta] [c] / cellVol; 
      gSVF         [alpha][c] =  8.0 * NC_CCweight[c] * gvolume[alpha][c] / cellVol;
       
      //Calculate the appropriate value of "factor" based on whether using SVF.
      if (b_svf==1) { factor = coeff * gSVF[beta][c] * gSVF[alpha][c]; } 
      else          { factor = coeff; } 
      
      // "If using the model with svf calculation," or "if mass is present on 
      // both nodes," calculate a non-zero interaction force based on velocity 
      // difference and the appropriate value of "factor".
      if ((b_svf==1) || (gmass[beta][c]>1.0e-100 && gmass[alpha][c]>1.0e-100)) { 
        gForce[beta] [c] = factor * (gvelocity_old[alpha][c] - gvelocity_old[beta] [c]);
        gForce[alpha][c] = factor * (gvelocity_old[beta] [c] - gvelocity_old[alpha][c]);
        
      } else {
        gForce[beta][c]  = Vector(0.0,0.0,0.0);
        gForce[alpha][c] = Vector(0.0,0.0,0.0);
      }
       
      //-- Calculate Updated Velocity ------------------------------------
      gvelocity_star[beta] [c] += (gForce[beta] [c]/( 8.0 * NC_CCweight[c] * gmass[beta] [c])) * delT;    
      gvelocity_star[alpha][c] += (gForce[alpha][c]/( 8.0 * NC_CCweight[c] * gmass[alpha][c])) * delT;    
    
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
  t->requires(Task::NewDW, lb->gMassLabel,         Ghost::None);
  t->modifies(             lb->gVelocityStarLabel, mss);
  sched->addTask(t, patches, ms);
}
