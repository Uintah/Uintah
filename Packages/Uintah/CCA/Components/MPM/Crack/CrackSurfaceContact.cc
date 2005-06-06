/********************************************************************************
    Crack.cc
    PART THREE: CRACK SURFACE CONTACT

    Created by Yajun Guo in 2002-2005.
********************************************************************************/

#include "Crack.h"
#include <Packages/Uintah/Core/Labels/MPMLabel.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/Math/Short27.h>
#include <Core/Geometry/Vector.h>
#include <Core/Geometry/IntVector.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Variables/NCVariable.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Variables/NodeIterator.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/SimulationStateP.h>
#include <Packages/Uintah/CCA/Ports/DataWarehouse.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Core/Containers/StaticArray.h>
#include <sgi_stl_warnings_off.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <sgi_stl_warnings_on.h>

using namespace Uintah;
using namespace SCIRun;
using namespace std;

using std::vector;
using std::string;

void Crack::addComputesAndRequiresAdjustCrackContactInterpolated(
            Task* t,const PatchSet* /*patches*/,
            const MaterialSet* matls) const
{
  const MaterialSubset* mss = matls->getUnion();

  // Nodal solutions above crack
  t->requires(Task::NewDW, lb->gMassLabel,         Ghost::None);
  t->requires(Task::NewDW, lb->gVolumeLabel,       Ghost::None);
  t->requires(Task::NewDW, lb->gNumPatlsLabel,     Ghost::None);
  t->requires(Task::NewDW, lb->gDisplacementLabel, Ghost::None);

  // Nodal solutions below crack
  t->requires(Task::NewDW, lb->GMassLabel,         Ghost::None);
  t->requires(Task::NewDW, lb->GVolumeLabel,       Ghost::None);
  t->requires(Task::NewDW, lb->GNumPatlsLabel,     Ghost::None);
  t->requires(Task::NewDW, lb->GCrackNormLabel,    Ghost::None);
  t->requires(Task::NewDW, lb->GDisplacementLabel, Ghost::None);

  t->modifies(lb->gVelocityLabel, mss);
  t->modifies(lb->GVelocityLabel, mss);

  t->computes(lb->frictionalWorkLabel);
}

void Crack::AdjustCrackContactInterpolated(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* matls,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    
    double mua=0.0,mub=0.0;
    double ma,mb,dvan,dvbn,dvat,dvbt,ratioa,ratiob;
    Vector va,vb,vc,dva,dvb,ta,tb,na,nb,norm;

    int numMatls = d_sharedState->getNumMPMMatls();
    ASSERTEQ(numMatls, matls->size());

    // Nodal solutions above crack
    StaticArray<constNCVariable<int> >    gNumPatls(numMatls);
    StaticArray<constNCVariable<double> > gmass(numMatls);
    StaticArray<constNCVariable<double> > gvolume(numMatls);
    StaticArray<constNCVariable<Vector> > gdisplacement(numMatls);
    StaticArray<NCVariable<Vector> >      gvelocity(numMatls);

    // Nodal solutions below crack
    StaticArray<constNCVariable<int> >    GNumPatls(numMatls);
    StaticArray<constNCVariable<double> > Gmass(numMatls);
    StaticArray<constNCVariable<double> > Gvolume(numMatls);
    StaticArray<constNCVariable<Vector> > GCrackNorm(numMatls);
    StaticArray<constNCVariable<Vector> > Gdisplacement(numMatls);
    StaticArray<NCVariable<Vector> >      Gvelocity(numMatls);
    StaticArray<NCVariable<double> >      frictionWork(numMatls);

    Ghost::GhostType  gnone = Ghost::None;

    for(int m=0;m<matls->size();m++){
      int dwi = matls->get(m);
      // Get data above crack
      new_dw->get(gNumPatls[m], lb->gNumPatlsLabel, dwi, patch, gnone, 0);
      new_dw->get(gmass[m],     lb->gMassLabel,     dwi, patch, gnone, 0);
      new_dw->get(gvolume[m],   lb->gVolumeLabel,   dwi, patch, gnone, 0);
      new_dw->get(gdisplacement[m],lb->gDisplacementLabel,dwi,patch,gnone,0);

      new_dw->getModifiable(gvelocity[m],lb->gVelocityLabel,dwi,patch);

      // Get data below crack
      new_dw->get(GNumPatls[m],lb->GNumPatlsLabel,  dwi, patch, gnone, 0);
      new_dw->get(Gmass[m],     lb->GMassLabel,     dwi, patch, gnone, 0);
      new_dw->get(Gvolume[m],   lb->GVolumeLabel,   dwi, patch, gnone, 0);
      new_dw->get(GCrackNorm[m],lb->GCrackNormLabel,dwi, patch, gnone, 0);
      new_dw->get(Gdisplacement[m],lb->GDisplacementLabel,dwi,patch,gnone,0);
      new_dw->getModifiable(Gvelocity[m],lb->GVelocityLabel,dwi,patch);

      new_dw->allocateAndPut(frictionWork[m],lb->frictionalWorkLabel,dwi,patch);
      frictionWork[m].initialize(0.);

      if(crackType[m]=="NO_CRACK") continue;  // no crack in this material

      // Check if there is contact. If yes, adjust velocity field
      for(NodeIterator iter=patch->getNodeIterator();!iter.done();iter++) {
        IntVector c = *iter;

        // Only one velocity field
        if(gNumPatls[m][c]==0 || GNumPatls[m][c]==0) continue;
        // Nodes in non-crack-zone
        norm=GCrackNorm[m][c];
        if(norm.length()<1.e-16) continue;  // should not happen now, but ...

        ma=gmass[m][c];
        va=gvelocity[m][c];
        mb=Gmass[m][c];
        vb=Gvelocity[m][c];
        vc=(va*ma+vb*mb)/(ma+mb);
	
	// Check if contact
        short contact=YES;
	if(Dot((vb-va),norm) <= 0.) { // Moving apart, no contact 
  	  contact=NO;
	}  
	else { // Approaching, check displacements 
          Vector ua=gdisplacement[m][c];
          Vector ub=Gdisplacement[m][c];
          if(Dot((ub-ua),norm) <= 0. ) contact=NO;
        }
	
	// If contact, adjust velocity field 
        if(!contact) { // No contact
          gvelocity[m][c]=gvelocity[m][c];
          Gvelocity[m][c]=Gvelocity[m][c];
          frictionWork[m][c] += 0.;
        }
        else { // There is contact, apply contact law
          if(crackType[m]=="null") { // Do nothing 
            gvelocity[m][c]=gvelocity[m][c];
            Gvelocity[m][c]=Gvelocity[m][c];
            frictionWork[m][c] += 0.;
          }

          else if(crackType[m]=="stick") { // Assign centerofmass velocity
            gvelocity[m][c]=vc;
            Gvelocity[m][c]=vc;
            frictionWork[m][c] += 0.;
          }

          else if(crackType[m]=="friction") { // Apply friction law
            // For velocity field above crack
            Vector deltva(0.,0.,0.);
            dva=va-vc;
            na=norm;
            dvan=Dot(dva,na);
            if((dva-na*dvan).length()>1.e-16)
               ta=(dva-na*dvan)/(dva-na*dvan).length();
            else
               ta=Vector(0.,0.,0.);
            dvat=Dot(dva,ta);
            ratioa=dvat/dvan;
            if( fabs(ratioa)>cmu[m] ) { // slide
               if(ratioa>0.) mua=cmu[m];
               if(ratioa<0.) mua=-cmu[m];
               deltva=-(na+ta*mua)*dvan;
               gvelocity[m][c]=va+deltva;
               frictionWork[m][c]+=ma*cmu[m]*dvan*dvan*(fabs(ratioa)-cmu[m]);
            }
            else { // stick
               gvelocity[m][c]=vc;
               frictionWork[m][c] += 0.;
            }

            // For velocity field below crack
            Vector deltvb(0.,0.,0.);
            dvb=vb-vc;
            nb=-norm;
            dvbn=Dot(dvb,nb);
            if((dvb-nb*dvbn).length()>1.e-16)
               tb=(dvb-nb*dvbn)/(dvb-nb*dvbn).length();
            else
               tb=Vector(0.,0.,0.);
            dvbt=Dot(dvb,tb);
            ratiob=dvbt/dvbn;
            if(fabs(ratiob)>cmu[m]) { // slide
               if(ratiob>0.) mub=cmu[m];
               if(ratiob<0.) mub=-cmu[m];
               deltvb=-(nb+tb*mub)*dvbn;
               Gvelocity[m][c]=vb+deltvb;
               frictionWork[m][c]+=mb*cmu[m]*dvbn*dvbn*(fabs(ratiob)-cmu[m]);
            }
            else { // stick
               Gvelocity[m][c]=vc;
               frictionWork[m][c] += 0.;
            }
          }
        } // End of if there is !contact
      } //End of loop over nodes
    } //End of loop over materials
  }  //End of loop over patches
}

void Crack::addComputesAndRequiresAdjustCrackContactIntegrated(Task* t,
                                const PatchSet* /*patches*/,
                                const MaterialSet* matls) const
{
  const MaterialSubset* mss = matls->getUnion();

  t->requires(Task::OldDW, lb->delTLabel);
  t->requires(Task::NewDW, lb->gMassLabel,          Ghost::None);
  t->requires(Task::NewDW, lb->gVolumeLabel,        Ghost::None);
  t->requires(Task::NewDW, lb->gNumPatlsLabel,      Ghost::None);
  t->requires(Task::NewDW, lb->gDisplacementLabel,  Ghost::None);
  t->modifies(             lb->gVelocityStarLabel,  mss);
  t->modifies(             lb->gAccelerationLabel,  mss);

  t->requires(Task::NewDW, lb->GMassLabel,          Ghost::None);
  t->requires(Task::NewDW, lb->GVolumeLabel,        Ghost::None);
  t->requires(Task::NewDW, lb->GNumPatlsLabel,      Ghost::None);
  t->requires(Task::NewDW, lb->GCrackNormLabel,     Ghost::None);
  t->requires(Task::NewDW, lb->GDisplacementLabel,  Ghost::None);
  t->modifies(             lb->GVelocityStarLabel,  mss);
  t->modifies(             lb->GAccelerationLabel,  mss);
  t->modifies(             lb->frictionalWorkLabel, mss);

}

void Crack::AdjustCrackContactIntegrated(const ProcessorGroup*,
                                const PatchSubset* patches,
                                const MaterialSubset* matls,
                                DataWarehouse* old_dw,
                                DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
	      
    double mua=0.0,mub=0.0;
    double ma,mb,dvan,dvbn,dvat,dvbt,ratioa,ratiob;
    Vector aa,ab,va,vb,vc,dva,dvb,ta,tb,na,nb,norm;

    int numMatls = d_sharedState->getNumMPMMatls();
    ASSERTEQ(numMatls, matls->size());

    // Nodal solutions above crack
    StaticArray<constNCVariable<double> > gmass(numMatls);
    StaticArray<constNCVariable<double> > gvolume(numMatls);
    StaticArray<constNCVariable<int> >    gNumPatls(numMatls);
    StaticArray<constNCVariable<Vector> > gdisplacement(numMatls);
    StaticArray<NCVariable<Vector> >      gvelocity_star(numMatls);
    StaticArray<NCVariable<Vector> >      gacceleration(numMatls);
    // Nodal solutions below crack
    StaticArray<constNCVariable<double> > Gmass(numMatls);
    StaticArray<constNCVariable<double> > Gvolume(numMatls);
    StaticArray<constNCVariable<int> >    GNumPatls(numMatls);
    StaticArray<constNCVariable<Vector> > GCrackNorm(numMatls);
    StaticArray<constNCVariable<Vector> > Gdisplacement(numMatls);
    StaticArray<NCVariable<Vector> >      Gvelocity_star(numMatls);
    StaticArray<NCVariable<Vector> >      Gacceleration(numMatls);
    // Friction work
    StaticArray<NCVariable<double> >      frictionWork(numMatls);

    Ghost::GhostType  gnone = Ghost::None;

    for(int m=0;m<matls->size();m++){
      int dwi = matls->get(m);
      // Get nodal data above crack
      new_dw->get(gmass[m],     lb->gMassLabel,     dwi, patch, gnone, 0);
      new_dw->get(gvolume[m],   lb->gVolumeLabel,   dwi, patch, gnone, 0);
      new_dw->get(gNumPatls[m], lb->gNumPatlsLabel, dwi, patch, gnone, 0);
      new_dw->get(gdisplacement[m],lb->gDisplacementLabel,dwi,patch,gnone,0);

      new_dw->getModifiable(gvelocity_star[m], lb->gVelocityStarLabel,
                                                         dwi, patch);
      new_dw->getModifiable(gacceleration[m],lb->gAccelerationLabel,
                                                         dwi, patch);
      // Get nodal data below crack
      new_dw->get(Gmass[m],     lb->GMassLabel,     dwi, patch, gnone, 0);
      new_dw->get(Gvolume[m],   lb->GVolumeLabel,   dwi, patch, gnone, 0);
      new_dw->get(GNumPatls[m], lb->GNumPatlsLabel, dwi, patch, gnone, 0);
      new_dw->get(GCrackNorm[m],lb->GCrackNormLabel,dwi, patch, gnone, 0);
      new_dw->get(Gdisplacement[m],lb->GDisplacementLabel,dwi,patch,gnone,0);

      new_dw->getModifiable(Gvelocity_star[m], lb->GVelocityStarLabel,
                                                         dwi, patch);
      new_dw->getModifiable(Gacceleration[m],lb->GAccelerationLabel,
                                                         dwi, patch);
      new_dw->getModifiable(frictionWork[m], lb->frictionalWorkLabel,
                                                         dwi, patch);

      delt_vartype delT;
      old_dw->get(delT, lb->delTLabel, getLevel(patches));

      if(crackType[m]=="NO_CRACK") continue; // No crack(s) in this material

      for(NodeIterator iter=patch->getNodeIterator();!iter.done();iter++) {
        IntVector c = *iter;

        // For nodes in non-crack zone, there is no contact, just continue
        if(gNumPatls[m][c]==0 || GNumPatls[m][c]==0) continue;
        norm=GCrackNorm[m][c];
        if(norm.length()<1.e-16) continue;   // should not happen now, but ...

        ma=gmass[m][c];
        va=gvelocity_star[m][c];
        aa=gacceleration[m][c];
        mb=Gmass[m][c];
        vb=Gvelocity_star[m][c];
        ab=Gacceleration[m][c];
        vc=(va*ma+vb*mb)/(ma+mb);
	
	// Check if contact
        short contact=YES;
        if(Dot((vb-va),norm) <= 0.) { // Moving apart, no contact
	  contact=NO;
	}
	else { // Approaching, check displacements
          Vector ua=gdisplacement[m][c];
                    //+delT*gvelocity_star[m][c];
          Vector ub=Gdisplacement[m][c];
                    //+delT*Gvelocity_star[m][c];
          if(Dot((ub-ua),norm) <= 0.) contact=NO;
        }
	
	// If contact, adjust velocity field  
        if(!contact) { // No contact
          gvelocity_star[m][c]=gvelocity_star[m][c];
          gacceleration[m][c]=gacceleration[m][c];
          Gvelocity_star[m][c]=Gvelocity_star[m][c];
          Gacceleration[m][c]=Gacceleration[m][c];
          frictionWork[m][c]+=0.0;
        }
        else { // There is contact, apply contact law
          if(crackType[m]=="null") { // Do nothing
            gvelocity_star[m][c]=gvelocity_star[m][c];
            gacceleration[m][c]=gacceleration[m][c];
            Gvelocity_star[m][c]=Gvelocity_star[m][c];
            Gacceleration[m][c]=Gacceleration[m][c];
            frictionWork[m][c]+=0.0;
          }

          else if(crackType[m]=="stick") { // Assign centerofmass velocity
            gvelocity_star[m][c]=vc;
            gacceleration[m][c]=aa+(vb-va)*mb/(ma+mb)/delT;
            Gvelocity_star[m][c]=vc;
            Gacceleration[m][c]=ab+(va-vb)*ma/(ma+mb)/delT;
            frictionWork[m][c]+=0.0;
          }

          else if(crackType[m]=="friction") { // Apply friction law
            // for velocity field above crack
            Vector deltva(0.,0.,0.);
            dva=va-vc;
            na=norm;
            dvan=Dot(dva,na);
            if((dva-na*dvan).length()>1.e-16)
               ta=(dva-na*dvan)/(dva-na*dvan).length();
            else
               ta=Vector(0.,0.,0.);
            dvat=Dot(dva,ta);
            ratioa=dvat/dvan;
            if( fabs(ratioa)>cmu[m] ) { // slide
               if(ratioa>0.) mua= cmu[m];
               if(ratioa<0.) mua=-cmu[m];
               deltva=-(na+ta*mua)*dvan;
               gvelocity_star[m][c]=va+deltva;
               gacceleration[m][c]=aa+deltva/delT;
               frictionWork[m][c]+=ma*cmu[m]*dvan*dvan*(fabs(ratioa)-cmu[m]);
            }
            else { // stick
               gvelocity_star[m][c]=vc;
               gacceleration[m][c]=aa+(vb-va)*mb/(ma+mb)/delT;
               frictionWork[m][c]+=0.0;
            }
	    
            // for velocity field below crack
            Vector deltvb(0.,0.,0.);
            dvb=vb-vc;
            nb=-norm;
            dvbn=Dot(dvb,nb);
            if((dvb-nb*dvbn).length()>1.e-16)
               tb=(dvb-nb*dvbn)/(dvb-nb*dvbn).length();
            else
               tb=Vector(0.,0.,0.);
            dvbt=Dot(dvb,tb);
            ratiob=dvbt/dvbn;
            if(fabs(ratiob)>cmu[m]) { // slide
               if(ratiob>0.) mub= cmu[m];
               if(ratiob<0.) mub=-cmu[m];
               deltvb=-(nb+tb*mub)*dvbn;
               Gvelocity_star[m][c]=vb+deltvb;
               Gacceleration[m][c]=ab+deltvb/delT;
               frictionWork[m][c]+=mb*cmu[m]*dvbn*dvbn*(fabs(ratiob)-cmu[m]);
            }
            else {  // stick
               Gvelocity_star[m][c]=vc;
               Gacceleration[m][c]=ab+(va-vb)*ma/(ma+mb)/delT;
               frictionWork[m][c]+=0.0;
            }
          }
        } // End of if there is !contact
      } //End of loop over nodes
    } //End of loop over materials
  }  //End of loop over patches
}

