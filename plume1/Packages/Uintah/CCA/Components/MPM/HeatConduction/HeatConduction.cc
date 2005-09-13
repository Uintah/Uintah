#include <Packages/Uintah/CCA/Components/MPM/HeatConduction/HeatConduction.h>
#include <Packages/Uintah/Core/Math/Short27.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/Core/Grid/Variables/NCVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/NodeIterator.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMBoundCond.h>
#include <Packages/Uintah/CCA/Components/MPM/MPMFlags.h>
#include <Packages/Uintah/Core/Grid/Variables/VarTypes.h>
#include <Packages/Uintah/Core/Labels/MPMLabel.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Core/Util/DebugStream.h>

using namespace Uintah;
using namespace SCIRun;

#define EROSION
#undef EROSION

static DebugStream cout_doing("HeatConduction", false);
static DebugStream cout_heat("MPMHeat", false);

HeatConduction::HeatConduction(SimulationStateP& sS,MPMLabel* labels, 
                               MPMFlags* flags)
{
  d_lb = labels;
  d_flag = flags;
  d_sharedState = sS;

  if(d_flag->d_8or27==8){
    NGP=1;
    NGN=1;
  } else if(d_flag->d_8or27==27){
    NGP=2;
    NGN=2;
  }
}

HeatConduction::~HeatConduction()
{
}

void HeatConduction::scheduleComputeInternalHeatRate(SchedulerP& sched,
                                                const PatchSet* patches,
                                                const MaterialSet* matls)
{  
  Task* t = scinew Task("MPM::computeInternalHeatRate",
                        this, &HeatConduction::computeInternalHeatRate);

  Ghost::GhostType  gan = Ghost::AroundNodes;
  Ghost::GhostType  gac = Ghost::AroundCells;
  Ghost::GhostType  gnone = Ghost::None;
  t->requires(Task::OldDW, d_lb->pXLabel,                         gan, NGP);
  t->requires(Task::OldDW, d_lb->pSizeLabel,                      gan, NGP);
  t->requires(Task::OldDW, d_lb->pMassLabel,                      gan, NGP);
  t->requires(Task::NewDW, d_lb->pVolumeDeformedLabel,            gan, NGP);
  t->requires(Task::NewDW, d_lb->pInternalHeatRateLabel_preReloc, gan, NGP);
#ifdef EROSION  
  t->requires(Task::NewDW, d_lb->pErosionLabel_preReloc,          gan, NGP);
#endif  
  t->requires(Task::NewDW, d_lb->gTemperatureLabel,               gac, 2*NGP);
  t->requires(Task::NewDW, d_lb->gMassLabel,                      gnone);
  t->computes(d_lb->gInternalHeatRateLabel);

  if(d_flag->d_fracture) { // for FractureMPM
    t->requires(Task::NewDW, d_lb->pgCodeLabel,                   gan, NGP);
    t->requires(Task::NewDW, d_lb->GTemperatureLabel,             gac, 2*NGP);
    t->requires(Task::NewDW, d_lb->GMassLabel,                    gnone);
    t->computes(d_lb->GInternalHeatRateLabel);
  }
  
  sched->addTask(t, patches, matls);
}

void HeatConduction::scheduleSolveHeatEquations(SchedulerP& sched,
                                           const PatchSet* patches,
                                           const MaterialSet* matls)
{
  /* solveHeatEquations
   *   in(G.MASS, G.INTERNALHEATRATE, G.EXTERNALHEATRATE)
   *   out(G.TEMPERATURERATE) */

  Task* t = scinew Task("MPM::solveHeatEquations",
                        this, &HeatConduction::solveHeatEquations);

  const MaterialSubset* mss = matls->getUnion();

  Ghost::GhostType  gnone = Ghost::None;
  t->requires(Task::NewDW, d_lb->gMassLabel,                           gnone);
  t->requires(Task::NewDW, d_lb->gVolumeLabel,                         gnone);
  t->requires(Task::NewDW, d_lb->gExternalHeatRateLabel,               gnone);
  t->modifies(             d_lb->gInternalHeatRateLabel,               mss);
  t->requires(Task::NewDW, d_lb->gThermalContactHeatExchangeRateLabel, gnone);
  t->computes(d_lb->gTemperatureRateLabel);

  if(d_flag->d_fracture) { // for FractureMPM
    t->requires(Task::NewDW, d_lb->GMassLabel,                         gnone);
    t->requires(Task::NewDW, d_lb->GVolumeLabel,                       gnone);
    t->requires(Task::NewDW, d_lb->GExternalHeatRateLabel,             gnone);
    t->modifies(             d_lb->GInternalHeatRateLabel,               mss);
    t->requires(Task::NewDW, d_lb->GThermalContactHeatExchangeRateLabel,gnone);
    t->computes(d_lb->GTemperatureRateLabel);
  }

  sched->addTask(t, patches, matls);
}

void HeatConduction::scheduleIntegrateTemperatureRate(SchedulerP& sched,
                                                 const PatchSet* patches,
                                                 const MaterialSet* matls)
{
  /* integrateTemperatureRate
   *   in(G.TEMPERATURE, G.TEMPERATURERATE)
   *   operation(t* = t + t_rate * dt)
   *   out(G.TEMPERATURE_STAR) */

  Task* t = scinew Task("MPM::integrateTemperatureRate",
                        this, &HeatConduction::integrateTemperatureRate);

  const MaterialSubset* mss = matls->getUnion();

  t->requires(Task::OldDW, d_sharedState->get_delt_label() );

  t->requires(Task::NewDW, d_lb->gTemperatureLabel,     Ghost::None);
  t->requires(Task::NewDW, d_lb->gTemperatureNoBCLabel, Ghost::None);
  t->modifies(             d_lb->gTemperatureRateLabel, mss);
  t->computes(d_lb->gTemperatureStarLabel);

  if(d_flag->d_fracture) { // for FractureMPM
    t->requires(Task::NewDW, d_lb->GTemperatureLabel,     Ghost::None);
    t->requires(Task::NewDW, d_lb->GTemperatureNoBCLabel, Ghost::None);
    t->modifies(             d_lb->GTemperatureRateLabel, mss);
    t->computes(d_lb->GTemperatureStarLabel);
  }
                     
  sched->addTask(t, patches, matls);
}



void HeatConduction::computeInternalHeatRate(const ProcessorGroup*,
                                        const PatchSubset* patches,
                                        const MaterialSubset* ,
                                        DataWarehouse* old_dw,
                                        DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    if (cout_doing.active())
      cout_doing <<"Doing computeInternalHeatRate on patch " << patch->getID()<<"\t\t MPM"<< endl;
    if (cout_heat.active())
      cout_heat << " Patch = " << patch->getID() << endl;

    ParticleInterpolator* interpolator = d_flag->d_interpolator->clone(patch);
    vector<IntVector> ni;
    ni.reserve(interpolator->size());
    vector<double> S;
    S.reserve(interpolator->size());
    vector<Vector> d_S;
    d_S.reserve(interpolator->size());

    Vector dx = patch->dCell();
    double oodx[3];
    oodx[0] = 1.0/dx.x();
    oodx[1] = 1.0/dx.y();
    oodx[2] = 1.0/dx.z();

    Ghost::GhostType  gac   = Ghost::AroundCells;
    Ghost::GhostType  gnone = Ghost::None;
    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );

      if (cout_heat.active())
        cout_heat << "  Material = " << m << endl;

      int dwi = mpm_matl->getDWIndex();
      double kappa = mpm_matl->getThermalConductivity();
      double Cv = mpm_matl->getSpecificHeat();
      
      constParticleVariable<Point>  px;
      constParticleVariable<double> pvol;
      constParticleVariable<double> pIntHeatRate;
      constParticleVariable<double> pMass;
      constParticleVariable<Vector> psize;
#ifdef EROSION      
      constParticleVariable<double> pErosion;
#endif      
      ParticleVariable<Vector>      pTemperatureGradient;
      constNCVariable<double>       gTemperature;
      constNCVariable<double>       gMass;
      NCVariable<double>            internalHeatRate;

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch,
                                                       Ghost::AroundNodes, NGP,
                                                       d_lb->pXLabel);

      old_dw->get(px,           d_lb->pXLabel,                         pset);
      new_dw->get(pvol,         d_lb->pVolumeDeformedLabel,            pset);
      new_dw->get(pIntHeatRate, d_lb->pInternalHeatRateLabel_preReloc, pset);
      old_dw->get(pMass,        d_lb->pMassLabel,                      pset);
      old_dw->get(psize,      d_lb->pSizeLabel,           pset);
#ifdef EROSION      
      new_dw->get(pErosion,     d_lb->pErosionLabel_preReloc, pset);
#endif      
      new_dw->get(gTemperature, d_lb->gTemperatureLabel, dwi, patch, gac,2*NGN);
      new_dw->get(gMass,        d_lb->gMassLabel,        dwi, patch, gnone, 0);
      new_dw->allocateAndPut(internalHeatRate, d_lb->gInternalHeatRateLabel,
                             dwi, patch);
      new_dw->allocateTemporary(pTemperatureGradient, pset);
  
      internalHeatRate.initialize(0.);

      // for FractureMPM
      constParticleVariable<Short27> pgCode;
      constNCVariable<double> GTemperature;
      constNCVariable<double> GMass;
      NCVariable<double> GinternalHeatRate;
      if(d_flag->d_fracture) { 
        new_dw->get(pgCode,       d_lb->pgCodeLabel, pset);
        new_dw->get(GTemperature, d_lb->GTemperatureLabel, dwi,patch,gac,2*NGN);
        new_dw->get(GMass,        d_lb->GMassLabel,        dwi,patch,gnone, 0);
        new_dw->allocateAndPut(GinternalHeatRate, d_lb->GInternalHeatRateLabel,
	                       dwi, patch);	      
        GinternalHeatRate.initialize(0.);
      }

      // Create a temporary variable to store the mass weighted grid node
      // internal heat rate that has been projected from the particles
      // to the grid
      NCVariable<double> gPIntHeatRate;
      new_dw->allocateTemporary(gPIntHeatRate, patch, gnone, 0);
      gPIntHeatRate.initialize(0.);
      // for FractureMPM
      NCVariable<double> GPIntHeatRate;
      if(d_flag->d_fracture) { 	
        new_dw->allocateTemporary(GPIntHeatRate, patch, gnone, 0);
        GPIntHeatRate.initialize(0.);
      }

      // First compute the temperature gradient at each particle

      for (ParticleSubset::iterator iter = pset->begin();
           iter != pset->end(); 
           iter++){
        particleIndex idx = *iter;

        // Get the node indices that surround the cell
        interpolator->findCellAndWeightsAndShapeDerivatives(px[idx],ni,S,d_S,
                                                            psize[idx]);

        // Weight the particle internal heat rate with the mass
        double pIntHeatRate_massWt = pIntHeatRate[idx]*pMass[idx];

        if (cout_heat.active()) {
          cout_heat << " Particle = " << idx << endl;
          cout_heat << " pIntHeatRate = " << pIntHeatRate[idx]
                    << " pMass = " << pMass[idx] << endl;
        }


        pTemperatureGradient[idx] = Vector(0.0,0.0,0.0);
        for (int k = 0; k < d_flag->d_8or27; k++){
#ifdef EROSION 	
          d_S[k] *= pErosion[idx];
#endif	  
          for (int j = 0; j<3; j++) {
            if(d_flag->d_fracture) { // for FractureMPM
              if(pgCode[idx][k]==1) { // above crack
                pTemperatureGradient[idx][j] +=
                    gTemperature[ni[k]] * d_S[k][j] * oodx[j];
              }
              else if(pgCode[idx][k]==2) { // below crack
	        pTemperatureGradient[idx][j] +=
	            GTemperature[ni[k]] * d_S[k][j] * oodx[j];
	      }	      
            }
            else { // for SerialMPM	    
              pTemperatureGradient[idx][j] += 
                    gTemperature[ni[k]] * d_S[k][j] * oodx[j];
            }
	    
            if (cout_heat.active()) {
              cout_heat << "   node = " << ni[k]
                        << " gTemp = " << gTemperature[ni[k]]
                        << " idx = " << idx
                        << " pTempGrad = " << pTemperatureGradient[idx][j]
                        << endl;
            }

          }
          // Project the mass weighted particle internal heat rate to
          // the grid
          if(patch->containsNode(ni[k])){
#ifdef EROSION              
            S[k] *= pErosion[idx];
#endif   	      
            if(d_flag->d_fracture) { // for FractureMPM
	      if(pgCode[idx][k]==1) { // above crack
                gPIntHeatRate[ni[k]] +=  (pIntHeatRate_massWt*S[k]);
              }
              else if(pgCode[idx][k]==2) { // below crack	      
	        GPIntHeatRate[ni[k]] +=  (pIntHeatRate_massWt*S[k]);
	      }
            }
            else { // for SerialMPM	     
              gPIntHeatRate[ni[k]] +=  (pIntHeatRate_massWt*S[k]);
            }
	     
            if (cout_heat.active()) {
              cout_heat << "   k = " << k << " node = " << ni[k] 
                        << " gPIntHeatRate = " << gPIntHeatRate[ni[k]] << endl;
            }

          }
        }
      }


      // Get the internal heat rate due to particle deformation at the
      // grid nodes by dividing gPIntHeatRate by the grid mass
      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++) {
        IntVector c = *iter;

        if (cout_heat.active()) {
          cout_heat << " c = " << c << " gPIntHeatRate = " << gPIntHeatRate[c]
                    << " gMass = " << gMass[c] << endl;
        }
	
        if(d_flag->d_fracture) { // for FractureMPM
          // above crack		
          gPIntHeatRate[c] /= gMass[c];
          internalHeatRate[c] = gPIntHeatRate[c];
	  // below crack
          GPIntHeatRate[c] /= GMass[c];
          GinternalHeatRate[c] = GPIntHeatRate[c];
	}
	else { // for SerialMPM 
          gPIntHeatRate[c] /= gMass[c];
          internalHeatRate[c] = gPIntHeatRate[c];
	}
      }

      // Compute T,ii
      for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++){
        particleIndex idx = *iter;
  
        // Get the node indices that surround the cell
        interpolator->findCellAndWeightsAndShapeDerivatives(px[idx],ni,S,d_S,
                                                            psize[idx]);

        // Calculate k/(rho*Cv)
        double alpha = kappa*pvol[idx]/Cv; 
        Vector T_i = pTemperatureGradient[idx];
        double T_ii = 0.0;
        double d_f_aH=d_flag->d_adiabaticHeating;
        IntVector node(0,0,0);
        for (int k = 0; k < d_flag->d_8or27; k++){
          node = ni[k];
          if(patch->containsNode(node)){
           Vector div(d_S[k].x()*oodx[0],d_S[k].y()*oodx[1],d_S[k].z()*oodx[2]);
	    if(d_flag->d_fracture) { // for FractureMPM
	      if(pgCode[idx][k]==1) { // above crack    
                T_ii = Dot(div, T_i)*(alpha/gMass[node])*d_f_aH;
	        internalHeatRate[node] -= T_ii;
	      }
              else if(pgCode[idx][k]==2) { // below crack
	        T_ii = Dot(div, T_i)*(alpha/GMass[node])*d_f_aH;
	        GinternalHeatRate[node] -= T_ii;
              }		      		    
	    }
	    else { // for SerialMPM
              T_ii = Dot(div, T_i)*(alpha/gMass[node])*d_f_aH;
              internalHeatRate[node] -= T_ii;
            }
	    


            if (cout_heat.active()) {
              cout_heat << "   node = " << node << " div = " << div 
                        << " T_i = " << T_i << " alpha = " << alpha*Cv 
                        << " T_ii = " << T_ii*Cv*gMass[node]
                        << " internalHeatRate = " << internalHeatRate[node] 
                        << endl;
            }
          }
        }
      }
    }  // End of loop over materials
    delete interpolator;
  }  // End of loop over patches
}


void HeatConduction::solveHeatEquations(const ProcessorGroup*,
                                   const PatchSubset* patches,
                                   const MaterialSubset* ,
                                   DataWarehouse* /*old_dw*/,
                                   DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    if (cout_doing.active())
      cout_doing <<"Doing solveHeatEquations on patch " << patch->getID() <<"\t\t\t MPM"<< endl;


    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();
      double Cv = mpm_matl->getSpecificHeat();
     
      // Get required variables for this patch
      constNCVariable<double> mass,externalHeatRate,gvolume;
      constNCVariable<double> thermalContactHeatExchangeRate;
      NCVariable<double> internalHeatRate;
            
      new_dw->get(mass,    d_lb->gMassLabel,      dwi, patch, Ghost::None, 0);
      new_dw->get(gvolume, d_lb->gVolumeLabel,    dwi, patch, Ghost::None, 0);
      new_dw->get(externalHeatRate, d_lb->gExternalHeatRateLabel,
                  dwi, patch, Ghost::None, 0);
      new_dw->getModifiable(internalHeatRate, d_lb->gInternalHeatRateLabel,
                            dwi, patch);
      new_dw->get(thermalContactHeatExchangeRate,
                  d_lb->gThermalContactHeatExchangeRateLabel,
                  dwi, patch, Ghost::None, 0);

      // for FractureMPM
      constNCVariable<double> Gmass,GexternalHeatRate,Gvolume;
      constNCVariable<double> GthermalContactHeatExchangeRate;
      NCVariable<double> GinternalHeatRate;
      if(d_flag->d_fracture) {  
        new_dw->get(Gmass,   d_lb->GMassLabel,      dwi, patch, Ghost::None, 0);
        new_dw->get(Gvolume, d_lb->GVolumeLabel,    dwi, patch, Ghost::None, 0);
        new_dw->get(GexternalHeatRate, d_lb->GExternalHeatRateLabel,
                    dwi, patch, Ghost::None, 0);
        new_dw->getModifiable(GinternalHeatRate, d_lb->GInternalHeatRateLabel,
                              dwi, patch);
        new_dw->get(GthermalContactHeatExchangeRate,
                    d_lb->GThermalContactHeatExchangeRateLabel,
                    dwi, patch, Ghost::None, 0);      
      }

      MPMBoundCond bc;
      if(d_flag->d_fracture) { // for FractureMPM
        bc.setBoundaryCondition(patch,dwi,"Temperature",internalHeatRate,
	                        gvolume,d_flag->d_8or27);	      
        bc.setBoundaryCondition(patch,dwi,"Temperature",GinternalHeatRate,
	                        Gvolume,d_flag->d_8or27);
      }
      else { // for SerialMPM
        bc.setBoundaryCondition(patch,dwi,"Temperature",internalHeatRate,
	                        gvolume,d_flag->d_8or27);
      }

      // Create variables for the results
      NCVariable<double> tempRate, GtempRate;
      if(d_flag->d_fracture) { // for FractureMPM
	new_dw->allocateAndPut(tempRate, d_lb->gTemperatureRateLabel,dwi,patch);
        new_dw->allocateAndPut(GtempRate,d_lb->GTemperatureRateLabel,dwi,patch);
	tempRate.initialize(0.0);
	GtempRate.initialize(0.0);
      }
      else { // for SerialMPM
        new_dw->allocateAndPut(tempRate,d_lb->gTemperatureRateLabel,dwi,patch);
        tempRate.initialize(0.0);
      }

      int n8or27=d_flag->d_8or27;
      for(NodeIterator iter=patch->getNodeIterator(n8or27);!iter.done();iter++){
        IntVector c = *iter;
	if(d_flag->d_fracture) { // for FractureMPM
          tempRate[c] = internalHeatRate[c]*((mass[c]-1.e-200)/mass[c]) +
	   (externalHeatRate[c])/(mass[c]*Cv)+thermalContactHeatExchangeRate[c];
	  GtempRate[c]=GinternalHeatRate[c]*((Gmass[c]-1.e-200)/Gmass[c]) +
           (GexternalHeatRate[c])/
                               (Gmass[c]*Cv)+GthermalContactHeatExchangeRate[c];
	}  
	else { // for SerialMPM
          tempRate[c] = internalHeatRate[c]*((mass[c]-1.e-200)/mass[c]) +
           (externalHeatRate[c])/(mass[c]*Cv)+thermalContactHeatExchangeRate[c];
	}
      } // End of loop over iter
    }
  }
}


void HeatConduction::integrateTemperatureRate(const ProcessorGroup*,
                                         const PatchSubset* patches,
                                         const MaterialSubset*,
                                         DataWarehouse* old_dw,
                                         DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    if (cout_doing.active())
      cout_doing <<"Doing integrateTemperatureRate on patch " << patch->getID()<< "\t\t MPM"<< endl;


    Ghost::GhostType  gnone = Ghost::None;
    for(int m = 0; m < d_sharedState->getNumMPMMatls(); m++){
      MPMMaterial* mpm_matl = d_sharedState->getMPMMaterial( m );
      int dwi = mpm_matl->getDWIndex();

      constNCVariable<double> temp_old,temp_oldNoBC;
      NCVariable<double> temp_rate,tempStar;
      delt_vartype delT;
 
      new_dw->get(temp_old,    d_lb->gTemperatureLabel,     dwi,patch,gnone,0);
      new_dw->get(temp_oldNoBC,d_lb->gTemperatureNoBCLabel, dwi,patch,gnone,0);
      new_dw->getModifiable(temp_rate, d_lb->gTemperatureRateLabel,dwi,patch);

      old_dw->get(delT, d_sharedState->get_delt_label(), getLevel(patches) );

      new_dw->allocateAndPut(tempStar, d_lb->gTemperatureStarLabel, dwi,patch);
      tempStar.initialize(0.0);

      // for FractureMPM
      constNCVariable<double> Gtemp_old,Gtemp_oldNoBC;
      NCVariable<double> Gtemp_rate,GtempStar;
      if(d_flag->d_fracture) {
       new_dw->get(Gtemp_old,    d_lb->GTemperatureLabel,    dwi,patch,gnone,0);
       new_dw->get(Gtemp_oldNoBC,d_lb->GTemperatureNoBCLabel,dwi,patch,gnone,0);
       new_dw->getModifiable(Gtemp_rate,d_lb->GTemperatureRateLabel,dwi,patch);
       new_dw->allocateAndPut(GtempStar,d_lb->GTemperatureStarLabel,dwi,patch);
       GtempStar.initialize(0.0);
      }
      
      int n8or27=d_flag->d_8or27;
      for(NodeIterator iter=patch->getNodeIterator(n8or27);!iter.done();iter++){
        IntVector c = *iter;
	if(d_flag->d_fracture) { // for FractureMPM
          tempStar[c] = temp_old[c] + temp_rate[c] * delT;
          GtempStar[c]=Gtemp_old[c] +Gtemp_rate[c] * delT;
	}
	else { // for SerialMPM
          tempStar[c] = temp_old[c] + temp_rate[c] * delT;
	}  
      }

      // Apply grid boundary conditions to the temperature 
      MPMBoundCond bc;
      if(d_flag->d_fracture) { // for FractureMPM
        bc.setBoundaryCondition(patch,dwi,"Temperature",tempStar,n8or27);
        bc.setBoundaryCondition(patch,dwi,"Temperature",GtempStar,n8or27);
      }
      else { // for SerialMPM
        bc.setBoundaryCondition(patch,dwi,"Temperature",tempStar,n8or27);
      }
       
      // Now recompute temp_rate as the difference between the temperature
      // interpolated to the grid (no bcs applied) and the new tempStar
      for(NodeIterator iter=patch->getNodeIterator(n8or27);!iter.done();iter++){
        IntVector c = *iter;
	if(d_flag->d_fracture) { // for FractureMPM
          temp_rate[c] = (tempStar[c] - temp_oldNoBC[c]) / delT;
	  Gtemp_rate[c]= (GtempStar[c]-Gtemp_oldNoBC[c]) / delT;
	}
	else { // for SerialMPM
          temp_rate[c] = (tempStar[c] - temp_oldNoBC[c]) / delT;
	}
      }
    }
  }
}
