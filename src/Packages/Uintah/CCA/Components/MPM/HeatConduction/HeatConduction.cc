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
  t->requires(Task::NewDW, d_lb->pdTdtLabel_preReloc,             gan, NGP);
#ifdef EROSION  
  t->requires(Task::NewDW, d_lb->pErosionLabel_preReloc,          gan, NGP);
#endif  
  t->requires(Task::NewDW, d_lb->gTemperatureLabel,               gac, 2*NGP);
  t->requires(Task::NewDW, d_lb->gMassLabel,                      gnone);
  t->computes(d_lb->gdTdtLabel);

  if(d_flag->d_fracture) { // for FractureMPM
    t->requires(Task::NewDW, d_lb->pgCodeLabel,                   gan, NGP);
    t->requires(Task::NewDW, d_lb->GTemperatureLabel,             gac, 2*NGP);
    t->requires(Task::NewDW, d_lb->GMassLabel,                    gnone);
    t->computes(d_lb->GdTdtLabel);
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
  t->modifies(             d_lb->gdTdtLabel,                           mss);
  t->requires(Task::NewDW, d_lb->gThermalContactTemperatureRateLabel, gnone);
  t->computes(d_lb->gTemperatureRateLabel);

  if(d_flag->d_fracture) { // for FractureMPM
    t->requires(Task::NewDW, d_lb->GMassLabel,                         gnone);
    t->requires(Task::NewDW, d_lb->GVolumeLabel,                       gnone);
    t->requires(Task::NewDW, d_lb->GExternalHeatRateLabel,             gnone);
    t->modifies(             d_lb->GdTdtLabel,                         mss);
    t->requires(Task::NewDW, d_lb->GThermalContactTemperatureRateLabel,gnone);
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
    vector<IntVector> ni(interpolator->size());
    vector<double> S(interpolator->size());
    vector<Vector> d_S(interpolator->size());

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
      constParticleVariable<double> pvol,pdTdt,pMass;
      constParticleVariable<Vector> psize;
#ifdef EROSION      
      constParticleVariable<double> pErosion;
#endif      
      ParticleVariable<Vector>      pTemperatureGradient;
      constNCVariable<double>       gTemperature,gMass;
      NCVariable<double>            gdTdt;

      ParticleSubset* pset = old_dw->getParticleSubset(dwi, patch,
                                                       Ghost::AroundNodes, NGP,
                                                       d_lb->pXLabel);

      old_dw->get(px,           d_lb->pXLabel,                         pset);
      new_dw->get(pvol,         d_lb->pVolumeDeformedLabel,            pset);
      new_dw->get(pdTdt,        d_lb->pdTdtLabel_preReloc,             pset);
      old_dw->get(pMass,        d_lb->pMassLabel,                      pset);
      old_dw->get(psize,        d_lb->pSizeLabel,                      pset);
#ifdef EROSION      
      new_dw->get(pErosion,     d_lb->pErosionLabel_preReloc, pset);
#endif      
      new_dw->get(gTemperature, d_lb->gTemperatureLabel, dwi, patch, gac,2*NGN);
      new_dw->get(gMass,        d_lb->gMassLabel,        dwi, patch, gnone, 0);
      new_dw->allocateAndPut(gdTdt, d_lb->gdTdtLabel,    dwi, patch);
      new_dw->allocateTemporary(pTemperatureGradient, pset);
  
      gdTdt.initialize(0.);

      // for FractureMPM
      constParticleVariable<Short27> pgCode;
      constNCVariable<double> GTemperature;
      constNCVariable<double> GMass;
      NCVariable<double> GdTdt;
      if(d_flag->d_fracture) { 
        new_dw->get(pgCode,       d_lb->pgCodeLabel, pset);
        new_dw->get(GTemperature, d_lb->GTemperatureLabel, dwi,patch,gac,2*NGN);
        new_dw->get(GMass,        d_lb->GMassLabel,        dwi,patch,gnone, 0);
        new_dw->allocateAndPut(GdTdt, d_lb->GdTdtLabel,    dwi,patch);	      
        GdTdt.initialize(0.);
      }

      // Create a temporary variable to store the mass weighted grid node
      // internal heat rate that has been projected from the particles
      // to the grid
      NCVariable<double> gpdTdt;
      new_dw->allocateTemporary(gpdTdt, patch, gnone, 0);
      gpdTdt.initialize(0.);
      // for FractureMPM
      NCVariable<double> GpdTdt;
      if(d_flag->d_fracture) { 	
        new_dw->allocateTemporary(GpdTdt, patch, gnone, 0);
        GpdTdt.initialize(0.);
      }

      // Compute the temperature gradient at each particle and project
      // the particle plastic work temperature rate to the grid
      for (ParticleSubset::iterator iter = pset->begin();
           iter != pset->end(); iter++){
        particleIndex idx = *iter;

        // Get the node indices that surround the cell
        interpolator->findCellAndWeightsAndShapeDerivatives(px[idx],ni,S,d_S,
                                                            psize[idx]);

        // Weight the particle plastic work temperature rate with the mass
        double pdTdt_massWt = pdTdt[idx]*pMass[idx];

        if (cout_heat.active()) {
          cout_heat << " Particle = " << idx << endl;
          cout_heat << " pdTdt = " << pdTdt[idx]
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
          // Project the mass weighted particle plastic work temperature
          // rate to the grid
          if(patch->containsNode(ni[k])){
#ifdef EROSION              
            S[k] *= pErosion[idx];
#endif   	      
            if(d_flag->d_fracture) { // for FractureMPM
	      if(pgCode[idx][k]==1) { // above crack
                gpdTdt[ni[k]] +=  (pdTdt_massWt*S[k]);
              }
              else if(pgCode[idx][k]==2) { // below crack	      
	        GpdTdt[ni[k]] +=  (pdTdt_massWt*S[k]);
	      }
            }
            else { // for SerialMPM	     
              gpdTdt[ni[k]] +=  (pdTdt_massWt*S[k]);
            }
	     
            if (cout_heat.active()) {
              cout_heat << "   k = " << k << " node = " << ni[k] 
                        << " gpdTdt = " << gpdTdt[ni[k]] << endl;
            }

          }
        }
      }


      // Get the plastic work temperature rate due to particle deformation
      // at the grid nodes by dividing gpdTdt by the grid mass
      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++) {
        IntVector c = *iter;

        if (cout_heat.active()) {
          cout_heat << " c = " << c << " gpdTdt = " << gpdTdt[c]
                    << " gMass = " << gMass[c] << endl;
        }
	
        gpdTdt[c] /= gMass[c];
        gdTdt[c] = gpdTdt[c];
        if(d_flag->d_fracture) { // for FractureMPM
	  // below crack
          GpdTdt[c] /= GMass[c];
          GdTdt[c] = GpdTdt[c];
	}
      }

      // Compute rate of temperature change at the grid due to conduction
      // and plastic work
      for(ParticleSubset::iterator iter = pset->begin();
          iter != pset->end(); iter++){
        particleIndex idx = *iter;
  
        // Get the node indices that surround the cell
        interpolator->findCellAndWeightsAndShapeDerivatives(px[idx],ni,S,d_S,
                                                            psize[idx]);

        // Calculate k/(rho*Cv)
        double alpha = kappa*pvol[idx]/Cv; 
        Vector dT_dx = pTemperatureGradient[idx];
        double Tdot_cond = 0.0;
        double d_f_aH=d_flag->d_adiabaticHeating;
        IntVector node(0,0,0);
        for (int k = 0; k < d_flag->d_8or27; k++){
          node = ni[k];
          if(patch->containsNode(node)){
           Vector div(d_S[k].x()*oodx[0],d_S[k].y()*oodx[1],d_S[k].z()*oodx[2]);
	    if(d_flag->d_fracture) { // for FractureMPM
	      if(pgCode[idx][k]==1) { // above crack    
                Tdot_cond = Dot(div, dT_dx)*(alpha/gMass[node])*d_f_aH;
	        gdTdt[node] -= Tdot_cond;
	      }
              else if(pgCode[idx][k]==2) { // below crack
	        Tdot_cond = Dot(div, dT_dx)*(alpha/GMass[node])*d_f_aH;
	        GdTdt[node] -= Tdot_cond;
              }		      		    
	    }
	    else { // for SerialMPM
              Tdot_cond = Dot(div, dT_dx)*(alpha/gMass[node])*d_f_aH;
              gdTdt[node] -= Tdot_cond;
            }

            if (cout_heat.active()) {
              cout_heat << "   node = " << node << " div = " << div 
                        << " dT_dx = " << dT_dx << " alpha = " << alpha*Cv 
                        << " Tdot_cond = " << Tdot_cond*Cv*gMass[node]
                        << " gdTdt = " << gdTdt[node] 
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
      constNCVariable<double> thermalContactTemperatureRate;
      NCVariable<double> gdTdt;
            
      new_dw->get(mass,    d_lb->gMassLabel,      dwi, patch, Ghost::None, 0);
      new_dw->get(gvolume, d_lb->gVolumeLabel,    dwi, patch, Ghost::None, 0);
      new_dw->get(externalHeatRate, d_lb->gExternalHeatRateLabel,
                  dwi, patch, Ghost::None, 0);
      new_dw->getModifiable(gdTdt, d_lb->gdTdtLabel, dwi, patch);
      new_dw->get(thermalContactTemperatureRate,
                  d_lb->gThermalContactTemperatureRateLabel,
                  dwi, patch, Ghost::None, 0);

      // for FractureMPM
      constNCVariable<double> Gmass,GexternalHeatRate,Gvolume;
      constNCVariable<double> GthermalContactTemperatureRate;
      NCVariable<double> GdTdt;
      if(d_flag->d_fracture) {  
        new_dw->get(Gmass,   d_lb->GMassLabel,      dwi, patch, Ghost::None, 0);
        new_dw->get(Gvolume, d_lb->GVolumeLabel,    dwi, patch, Ghost::None, 0);
        new_dw->get(GexternalHeatRate, d_lb->GExternalHeatRateLabel,
                    dwi, patch, Ghost::None, 0);
        new_dw->getModifiable(GdTdt, d_lb->GdTdtLabel, dwi, patch);
        new_dw->get(GthermalContactTemperatureRate,
                    d_lb->GThermalContactTemperatureRateLabel,
                    dwi, patch, Ghost::None, 0);      
      }

      // Create variables for the results
      NCVariable<double> tempRate, GtempRate;
      new_dw->allocateAndPut(tempRate, d_lb->gTemperatureRateLabel,dwi,patch);
      tempRate.initialize(0.0);
      if(d_flag->d_fracture) { // for FractureMPM
        new_dw->allocateAndPut(GtempRate,d_lb->GTemperatureRateLabel,dwi,patch);
	GtempRate.initialize(0.0);
      }

      int n8or27=d_flag->d_8or27;
      for(NodeIterator iter=patch->getNodeIterator(n8or27);!iter.done();iter++){
        IntVector c = *iter;
        tempRate[c] = gdTdt[c]*((mass[c]-1.e-200)/mass[c]) +
	   (externalHeatRate[c])/(mass[c]*Cv)+thermalContactTemperatureRate[c];
	if(d_flag->d_fracture) { // for FractureMPM
	  GtempRate[c]=GdTdt[c]*((Gmass[c]-1.e-200)/Gmass[c]) +
           (GexternalHeatRate[c])/
                               (Gmass[c]*Cv)+GthermalContactTemperatureRate[c];
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
        tempStar[c] = temp_old[c] + temp_rate[c] * delT;
	if(d_flag->d_fracture) { // for FractureMPM
          GtempStar[c]=Gtemp_old[c] +Gtemp_rate[c] * delT;
	}
      }

      // Apply grid boundary conditions to the temperature 
      MPMBoundCond bc;
      bc.setBoundaryCondition(patch,dwi,"Temperature",tempStar,n8or27);
      if(d_flag->d_fracture) { // for FractureMPM
        bc.setBoundaryCondition(patch,dwi,"Temperature",GtempStar,n8or27);
      }

      // Now recompute temp_rate as the difference between the temperature
      // interpolated to the grid (no bcs applied) and the new tempStar
      for(NodeIterator iter=patch->getNodeIterator(n8or27);!iter.done();iter++){
        IntVector c = *iter;
        temp_rate[c] = (tempStar[c] - temp_oldNoBC[c]) / delT;
	if(d_flag->d_fracture) { // for FractureMPM
	  Gtemp_rate[c]= (GtempStar[c]-Gtemp_oldNoBC[c]) / delT;
	}
      }
    }
  }
}
