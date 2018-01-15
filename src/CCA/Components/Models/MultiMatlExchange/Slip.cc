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

#include <CCA/Components/Models/MultiMatlExchange/Slip.h>
#include <CCA/Ports/SchedulerP.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Material.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/Variables/CCVariable.h>

#include <Core/ProblemSpec/ProblemSpec.h>
#include <ostream>                         // for operator<<, basic_ostream
#include <vector>

using namespace Uintah;
using namespace ExchangeModels;
using namespace std;

extern DebugStream dbgExch;
//______________________________________________________________________
//
SlipExch::SlipExch(const ProblemSpecP     & exch_ps,
                   const SimulationStateP & sharedState )
  : ExchangeModel( exch_ps, sharedState )
{ 
  proc0cout << "__________________________________\n";
  proc0cout << " Now creating the Slip Exchange model " << endl;
  d_prob_spec = exch_ps;
}

SlipExch::~SlipExch()
{
}

//______________________________________________________________________
//
void SlipExch::problemSetup()
{
  ProblemSpecP model_ps = d_prob_spec->findBlock( "Model" );
  model_ps->require("fluidMatlIndex",               d_fluidMatlIndx);
  model_ps->require("solidMatlIndex",               d_solidMatlIndx);
  model_ps->require("momentum_accommodation_coeff", d_momentum_accommodation_coeff);
  model_ps->require("thermal_accommodation_coeff",  d_thermal_accommodation_coeff);
  
  
  proc0cout << " fluidMatlIndex: " << d_fluidMatlIndx << " thermal_accommodation_coeff " << d_thermal_accommodation_coeff << endl;
//  d_exchCoeff->problemSetup(mat_ps, d_sharedState);
};

//______________________________________________________________________
//
void SlipExch::scheduleAddExchangeToMomentumAndEnergy(SchedulerP           & sched,
                                                      const PatchSet       * patches,
                                                      const MaterialSubset * ice_matls,
                                                      const MaterialSubset * mpm_matls,
                                                      const MaterialSubset * press_matl,
                                                      const MaterialSet    * all_matls)
{
  //__________________________________
  // compute surface normal gradients
  scheduleComputeSurfaceNormal( sched, patches, mpm_matls, press_matl);
#if 0
  //__________________________________
  //
  string name = "SlipExch::addExchangeToMomentumAndEnergy_Slip";
  Task* t = scinew Task(name, this, &SlipExch::addExchangeToMomentumAndEnergy);
 
  printSchedule( patches, dbgExch, name );

  Ghost::GhostType  gn  = Ghost::None;
  Ghost::GhostType  gac = Ghost::AroundCells;
  
  t->requires( Task::OldDW,  Ilb->delTLabel,getLevel(patches)); 
  t->requires( Task::NewDW,  Mlb->gMassLabel,       mpm_matls,  gac, 1 );     
  t->requires( Task::OldDW,  Mlb->NC_CCweightLabel, press_matl, gac, 1 );
  t->requires( Task::NewDW,  d_surfaceNormLabel, mpm_matls, gn );
                                // I C E
  t->requires( Task::OldDW,  Ilb->temp_CCLabel,       ice_matls, gn );
  t->requires( Task::NewDW,  Ilb->specific_heatLabel, ice_matls, gn );
  t->requires( Task::NewDW,  Ilb->gammaLabel,         ice_matls, gn );
  t->requires( Task::NewDW,  Ilb->meanfreepathLabel,  ice_matls, gac, 1);

                                // A L L  M A T L S
  t->requires( Task::NewDW,  Ilb->mass_L_CCLabel,           gn );      
  t->requires( Task::NewDW,  Ilb->mom_L_CCLabel,            gn );      
  t->requires( Task::NewDW,  Ilb->int_eng_L_CCLabel,        gn );
  t->requires( Task::NewDW,  Ilb->sp_vol_CCLabel,           gn );      
  t->requires( Task::NewDW,  Ilb->vol_frac_CCLabel,         gn );      
 
#if 0
  computesRequires_CustomBCs(t, "CC_Exchange", lb, ice_matls, d_customBC_var_basket); 
#endif

  t->computes( Ilb->Tdot_CCLabel );
  t->computes( Ilb->mom_L_ME_CCLabel );      
  t->computes( Ilb->eng_L_ME_CCLabel ); 
  t->computes( d_vel_CCTransposedLabel );

  
  t->modifies( Ilb->temp_CCLabel, mpm_matls );
  t->modifies( Ilb->vel_CCLabel,  mpm_matls );

  sched->addTask(t, patches, all_matls);
#endif

}

//______________________________________________________________________
//
void SlipExch::addExchangeToMomentumAndEnergy( const ProcessorGroup *,
                                               const PatchSubset    * patches,
                                               const MaterialSubset *,
                                               DataWarehouse        * old_dw,
                                               DataWarehouse        * new_dw)
{
#if 0
  const Level* level = getLevel(patches);
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask(patches, patch, dbgExch,"Doing SlipExch::addExchangeToMomentumAndEnergy");
    
    int numMPMMatls = d_sharedState->getNumMPMMatls();
    int numICEMatls = d_sharedState->getNumICEMatls();
    int numALLMatls = numMPMMatls + numICEMatls;
    Ghost::GhostType  gn = Ghost::None;
    Ghost::GhostType  gac = Ghost::AroundCells;
    
    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label(),level);
    //Vector zero(0.,0.,0.);

    // Create arrays for the grid data
    std::vector< CCVariable<double> > cv(numALLMatls);
    std::vector< CCVariable<double> > Temp_CC(numALLMatls);
    std::vector< constCCVariable<double> > gamma(numALLMatls);  
    std::vector< constCCVariable<double> > vol_frac_CC(numALLMatls);
    std::vector< constCCVariable<double> > sp_vol_CC(numALLMatls);
    std::vector< constCCVariable<Vector> > mom_L(numALLMatls);
    std::vector< constCCVariable<double> > int_eng_L(numALLMatls);
    constCCVariable<double> meanfreepath;   // This mean free path does not have viscosity in it, which is okay per how it is used in this code
    constNCVariable<double> NC_CCweight;
    constNCVariable<double> NCsolidMass;

    // Create variables for the results
    std::vector< CCVariable<Vector> > mom_L_ME(numALLMatls);
    std::vector< CCVariable<Vector> > vel_CC(numALLMatls);
    std::vector< CCVariable<double> > int_eng_L_ME(numALLMatls);
    std::vector< CCVariable<double> > Tdot(numALLMatls);
    std::vector< CCVariable<Vector> > dbg_vel_T(numALLMatls);  
    std::vector< CCVariable<Vector> > surfNormGrads(numALLMatls); // Aimie added this!!  
    
    std::vector< constCCVariable<double> > mass_L(numALLMatls);
    std::vector< constCCVariable<double> > old_temp(numALLMatls);
    vector<double> sp_vol(numALLMatls);

    double tmp[MAX_MATLS], b[MAX_MATLS];
    Vector vel_T[MAX_MATLS]; // is like a vel_CC tranformed vector that is formed for each material (MAX_MATLS???) at each cell c
    Vector notUsed;
    FastMatrix Q(3,3), a(numALLMatls, numALLMatls); // Q is always 3x3, because it is based on 3D, not number of materials
    FastMatrix k(numALLMatls, numALLMatls), K(numALLMatls, numALLMatls), Kslip(numALLMatls, numALLMatls); // Has the size of the number of materials
    FastMatrix h(numALLMatls, numALLMatls), H(numALLMatls, numALLMatls);
    Q.zero(); 
    a.zero(); 
    k.zero(); 
    K.zero(); 
    Kslip.zero(); 
    h.zero(); 
    H.zero(); 

    getConstantExchangeCoefficients(k,h);
    
/*`==========TESTING==========*/
    FastMatrix K_constant(numALLMatls, numALLMatls);
    K_constant.copy(k); 
/*===========TESTING==========`*/
    

    for (int m = 0; m < numALLMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
      int indx = matl->getDWIndex();
      new_dw->allocateTemporary(cv[m], patch);
      
      if(mpm_matl){                 // M P M
        CCVariable<double> oldTemp;
        new_dw->getCopy(oldTemp,          lb->temp_CCLabel,indx,patch,gn,0);
        new_dw->getModifiable(vel_CC[m],  lb->vel_CCLabel, indx,patch);
        new_dw->getModifiable(Temp_CC[m], lb->temp_CCLabel,indx,patch);
        old_temp[m] = oldTemp;
        cv[m].initialize(mpm_matl->getSpecificHeat());
        
        if(d_exchCoeff->slipflow() && indx==d_exchCoeff->slip_solid_matlindex()){
          new_dw->get(NCsolidMass, MIlb->gMassLabel, indx,patch,gac,1);
        }  
      }
      if(ice_matl){                 // I C E
        constCCVariable<double> cv_ice;
        old_dw->get(old_temp[m],   lb->temp_CCLabel,      indx, patch,gn,0);
        new_dw->get(cv_ice,        lb->specific_heatLabel,indx, patch,gn,0);
        new_dw->get(gamma[m],      lb->gammaLabel,        indx, patch,gn,0);
        
        if(d_exchCoeff->slipflow() && indx==d_exchCoeff->slip_fluid_matlindex()){
          new_dw->get(meanfreepath, lb->meanfreepathLabel, indx, patch,gac,1);
        }  
       
        new_dw->allocateTemporary(vel_CC[m],  patch);
        new_dw->allocateTemporary(Temp_CC[m], patch); 
        cv[m].copyData(cv_ice);
      }                             // A L L  M A T L S

      new_dw->get(mass_L[m],        lb->mass_L_CCLabel,   indx, patch,gn, 0);
      new_dw->get(sp_vol_CC[m],     lb->sp_vol_CCLabel,   indx, patch,gn, 0);
      new_dw->get(mom_L[m],         lb->mom_L_CCLabel,    indx, patch,gn, 0);
      new_dw->get(int_eng_L[m],     lb->int_eng_L_CCLabel,indx, patch,gn, 0);
      new_dw->get(vol_frac_CC[m],   lb->vol_frac_CCLabel, indx, patch,gn, 0);
      
      new_dw->allocateAndPut(Tdot[m],         lb->Tdot_CCLabel,         indx,patch);
      new_dw->allocateAndPut(mom_L_ME[m],     lb->mom_L_ME_CCLabel,     indx,patch);
      new_dw->allocateAndPut(int_eng_L_ME[m], lb->eng_L_ME_CCLabel,     indx,patch);
      new_dw->allocateAndPut(dbg_vel_T[m],    lb->vel_CCTransposedLabel,indx,patch);
      new_dw->allocateAndPut(surfNormGrads[m],lb->surfNormGradsLabel,   indx,patch);
      dbg_vel_T[m].initialize(Vector(0,0,0));
      surfNormGrads[m].initialize(Vector(0,0,0));
    }
    
    if(d_exchCoeff->slipflow()){
      old_dw->get(NC_CCweight, MIlb->NC_CCweightLabel, 0, patch,gac,1);
    }  
  
    // Convert momenta to velocities and internal energy to Temp
    for(CellIterator iter = patch->getExtraCellIterator(); !iter.done();iter++){
      IntVector c = *iter;
      for (int m = 0; m < numALLMatls; m++) {
        Temp_CC[m][c] = int_eng_L[m][c]/(mass_L[m][c]*cv[m][c]);
        vel_CC[m][c]  = mom_L[m][c]/mass_L[m][c];
      }
    }
    

    // Begin here: Start iterations over all cells in domain and get an idea of where the slip is occurring 
    for(CellIterator iter = patch->getCellIterator(); !iter.done();iter++){
      IntVector c = *iter;

      Q.identity();  // Transformation matrix is initialized
      Kslip.copy(k); 
      H.copy(h);    
      
      //__________________________________
      //  Determine Exchange Coefficients
      if(d_exchCoeff->slipflow()){     
      IntVector nodeIdx[8];
      patch->findNodesFromCell(c,nodeIdx);

      double MaxMass = d_SMALL_NUM;
      double MinMass = 1.0/d_SMALL_NUM;
      for (int nN=0; nN<8; nN++) {
        MaxMass = std::max(MaxMass,NC_CCweight[nodeIdx[nN]]*NCsolidMass[nodeIdx[nN]]);
        MinMass = std::min(MinMass,NC_CCweight[nodeIdx[nN]]*NCsolidMass[nodeIdx[nN]]);
      }

      Vector dx = patch->dCell();
      double vol = dx.x() * dx.y() * dx.z();

      if ((MaxMass-MinMass)/MaxMass == 1.0 && (MaxMass > (d_TINY_RHO * vol))){   // If the cell c is a surface shell            
	computeSurfaceRotationMatrix(Q, notUsed, nodeIdx, NCsolidMass, NC_CCweight, dx); // Makes Q at each cell c

	surfNormGrads[0][c] = notUsed; 

	double A_V = 1.0/( dx.x()*fabs(Q(1,0)) + dx.y()*fabs(Q(1,1)) + dx.z()*fabs(Q(1,2)) );

	int gm = d_exchCoeff->slip_fluid_matlindex();  
	int sm = d_exchCoeff->slip_solid_matlindex();  

	double av = d_exchCoeff->momentum_accommodation_coeff(); 
	double at = d_exchCoeff->thermal_accommodation_coeff(); 

	double Beta_v = (2 - av)/av;
	double Beta_t = ((2 - at)/at) * (2/(1 + gamma[gm][c])) * (1/cv[gm][c]);

	Kslip(gm,sm) = A_V / (Beta_v * meanfreepath[c] * vol_frac_CC[sm][c]); // fills Kslip using the definition (mu divided out of lambda term)

	if (Kslip(gm,sm) > k(gm,sm)) {
	  Kslip(gm,sm) = k(gm,sm); // K > Kslip in reality, so if Kslip > K in computation, fix this. 
	}
	Kslip(sm,gm) = Kslip(gm,sm); // Make the inverse indices of the Kslip matrix equal to each other
	H(gm,sm) = A_V / (Beta_t * meanfreepath[c] * vol_frac_CC[sm][c]); // The viscosity does not appear here because it's taken out of lambda

	if (H(gm,sm) > h(gm,sm)) {
	  H(gm,sm) = h(gm,sm); // H > Hslip in reality, so if Hslip > H in computation, fix this.
	}
	H(sm,gm) = H(gm,sm); // Make inverse indices of the Kslip matrix equal to each other
     }  
      } // end if ( d_exchCoeff->slipflow() )

      //---------- M O M E N T U M   E X C H A N G E
      for(int i = 0; i < 3; i++) {           // all principle directions 
	K.copy(k);                          // copy k into a matrix K (one will be needed for each principle direction???)
        
	if(d_exchCoeff->slipflow() && i != 1) { // if the principle direction is NOT the normal to the surface
	  K.copy(Kslip); // Then K is a copy of Kslip
	}
        
	/*`==========TESTING==========*/
	  // K.copy(K_constant);
	  // Q.identity(); 
       /*===========TESTING==========`*/
       //__________________________________
       //  coordinate Transformation
	for(int m = 0; m < numALLMatls; m++) { // for all materials
	  tmp[m]    = delT * vol_frac_CC[m][c]; // make some temporary vector for the specified material (???)
	  vel_T[m][i] = 0;                      // initialize the tranformation velocity field vector for the material

	  for(int j = 0; j < 3; j++) {
	    vel_T[m][i] += Q(i,j) * vel_CC[m][c][j]; // Use linear algegra to compute the transformed velocity
	  }
          
	  /*`==========TESTING==========*/
	  dbg_vel_T[m][c][i] = vel_T[m][i];        // Defined earlier that vel_CCTransposed is the label that spits out dbg_vel_T
	  /*===========TESTING==========`*/      
	}
       
	for(int m = 0; m < numALLMatls; m++) {// loop over all materials again
	  double adiag = 1.0;
	  b[m] = 0.0;
          
	  for(int n = 0; n < numALLMatls; n++) { 
	    a(m,n) = -sp_vol_CC[m][c] * tmp[n] * K(m,n);
	    adiag -= a(m,n);
	    b[m]  -= a(m,n) * (vel_T[n][i] - vel_T[m][i]);
	  }
	  a(m,m) = adiag;
	}
      
	a.destructiveSolve(b); 
        
	for(int m = 0; m < numALLMatls; m++) {
	  vel_T[m][i] = b[m]; 
	  // cout_doing << "cell " << c << "\t vel_T_dbg " << dbg_vel_T[m][c][i] << "\t vel_T " << vel_T[m][i] << endl;
	}
      } // finished over all principle directions
       
      //__________________________________
      //  coordinate transformation
      for(int i = 0; i < 3; i++) {// for all principle directions
	for(int m = 0; m < numALLMatls; m++) {// for all materials
	  for(int j = 0; j < 3; j++) {// for all secondary directions
	    vel_CC[m][c][i] += Q(j,i) * vel_T[m][j]; // Use the transpose of Q to back out the velocity in the cartesian system
	  }
	}
      }

      //---------- E N E R G Y   E X C H A N G E   
      if(d_exchCoeff->d_heatExchCoeffModel != "constant"){
        getVariableExchangeCoefficients( K, H, c, mass_L);
      }
      
      for(int m = 0; m < numALLMatls; m++) {
        tmp[m] = delT * vol_frac_CC[m][c];
      }
      
      for(int m = 0; m < numALLMatls; m++) {
        double adiag = 1.0;
        b[m] = 0.0;
        
        for(int n = 0; n < numALLMatls; n++) {
          a(m,n) = -sp_vol_CC[m][c] * tmp[n] * H(m,n) / cv[m][c];  // Here H has already been changed into Hslip, and Hslip is the same in every direction.
          adiag -= a(m,n);
          b[m]  -= a(m,n) * (Temp_CC[n][c] - Temp_CC[m][c]);
        }
        
        a(m,m) = adiag;
      }
      
      a.destructiveSolve(b);
      
      for(int m = 0; m < numALLMatls; m++) {
        Temp_CC[m][c] += b[m];
      }
    }  //end CellIterator loop

/*`==========TESTING==========*/ 
    if(d_customBC_var_basket->usingLodi || 
       d_customBC_var_basket->usingMicroSlipBCs){ 
      std::vector<CCVariable<double> > temp_CC_Xchange(numALLMatls);
      std::vector<CCVariable<Vector> > vel_CC_Xchange(numALLMatls);      
      for (int m = 0; m < numALLMatls; m++) {
        Material* matl = d_sharedState->getMaterial(m);
        int indx = matl->getDWIndex();
        new_dw->allocateAndPut(temp_CC_Xchange[m],lb->temp_CC_XchangeLabel,indx,patch);
        new_dw->allocateAndPut(vel_CC_Xchange[m], lb->vel_CC_XchangeLabel, indx,patch);
        vel_CC_Xchange[m].copy(vel_CC[m]);
        temp_CC_Xchange[m].copy(Temp_CC[m]);
      }
    }
    
    preprocess_CustomBCs("CC_Exchange",old_dw, new_dw, lb, patch, 
                          999,d_customBC_var_basket);
    
/*===========TESTING==========`*/  
    for (int m = 0; m < numALLMatls; m++)  {
      Material* matl = d_sharedState->getMaterial( m );
      int indx = matl->getDWIndex();
      setBC(vel_CC[m], "Velocity",   patch, d_sharedState, indx, new_dw,
                                                        d_customBC_var_basket);
      setBC(Temp_CC[m],"Temperature",gamma[m], cv[m], patch, d_sharedState, 
                                         indx, new_dw,  d_customBC_var_basket);
#if SET_CFI_BC                                         
//      set_CFI_BC<Vector>(vel_CC[m],  patch);
//      set_CFI_BC<double>(Temp_CC[m], patch);
#endif
    }
    
    delete_CustomBCs(d_customBC_var_basket);
    //__________________________________
    // Convert vars. primitive-> flux 
    for(CellIterator iter = patch->getExtraCellIterator(); !iter.done();iter++){
      IntVector c = *iter;
      for (int m = 0; m < numALLMatls; m++) {
        int_eng_L_ME[m][c] = Temp_CC[m][c]*cv[m][c] * mass_L[m][c];
        mom_L_ME[m][c]     = vel_CC[m][c]           * mass_L[m][c];
        Tdot[m][c]         = (Temp_CC[m][c] - old_temp[m][c])/delT;
      }
    }

    //---- P R I N T   D A T A ------ 
    if (switchDebug_MomentumExchange_CC ) {
      for(int m = 0; m < numALLMatls; m++) {
        Material* matl = d_sharedState->getMaterial( m );
        int indx = matl->getDWIndex();
        ostringstream desc;
        desc<<"addExchangeToMomentumAndEnergy_"<<indx<<"_patch_"
            <<patch->getID();
        printVector(indx, patch,1, desc.str(), "mom_L_ME", 0,mom_L_ME[m]);
        printVector( indx, patch,1, desc.str(),"vel_CC", 0,  vel_CC[m]);
        printData(  indx, patch,1, desc.str(),"int_eng_L_ME",int_eng_L_ME[m]);
        printData(  indx, patch,1, desc.str(),"Tdot",        Tdot[m]);
        printData(  indx, patch,1, desc.str(),"Temp_CC",     Temp_CC[m]);
      }
    }
  } //patches
#endif
}
