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
#include <CCA/Components/ICE/CustomBCs/BoundaryCond.h>
#include <CCA/Components/ICE/Materials/ICEMaterial.h>
#include <CCA/Components/MPM/Materials/MPMMaterial.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Material.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/SimulationStateP.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/ProblemSpec/ProblemSpec.h>

#include <ostream>                         // for operator<<, basic_ostream
#include <vector>

#define MAX_MATLS 8
#define d_TINY_RHO 1e-12

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

  d_exchCoeff = scinew ExchangeCoefficients();
  Ilb  = scinew ICELabel();
  Mlb  = scinew MPMLabel();

  d_vel_CCTransposedLabel = VarLabel::create("vel_CCTransposed", CCVariable<Vector>::getTypeDescription());
  d_meanFreePathLabel     = VarLabel::create("meanFreePath",     CCVariable<double>::getTypeDescription());
}

//______________________________________________________________________
//
SlipExch::~SlipExch()
{
  delete d_exchCoeff;
  delete Ilb;
  delete Mlb;
  VarLabel::destroy( d_vel_CCTransposedLabel );
  VarLabel::destroy( d_meanFreePathLabel );
}

//______________________________________________________________________
//
void SlipExch::problemSetup(const ProblemSpecP & exch_ps)
{

  // read in the exchange coefficients
  d_exchCoeff->problemSetup( exch_ps, d_numMatls );

  ProblemSpecP model_ps = exch_ps->findBlock( "Model" );
  model_ps->require("fluidMatlIndex",               d_fluidMatlIndx);
  model_ps->require("solidMatlIndex",               d_solidMatlIndx);
  model_ps->require("momentum_accommodation_coeff", d_momentum_accommodation_coeff);
  model_ps->require("thermal_accommodation_coeff",  d_thermal_accommodation_coeff);


  proc0cout << " fluidMatlIndex: " << d_fluidMatlIndx << " thermal_accommodation_coeff " << d_thermal_accommodation_coeff << endl;
//  d_exchCoeff->problemSetup(mat_ps, d_sharedState);
};


//______________________________________________________________________
//
void SlipExch::sched_AddExch_VelFC(SchedulerP           & sched,
                                   const PatchSet       * patches,
                                   const MaterialSubset * iceMatls,
                                   const MaterialSet    * allMatls,
                                   customBC_globalVars  * BC_globalVars,
                                   const bool recursion)
{
}

//______________________________________________________________________
//
void SlipExch::addExch_VelFC(const ProcessorGroup  * pg,
                             const PatchSubset     * patch,
                             const MaterialSubset  * matls,
                             DataWarehouse         * old_dw,
                             DataWarehouse         * new_dw,
                             customBC_globalVars   * BC_globalVars,
                             const bool recursion)
{
}

//______________________________________________________________________
//
void SlipExch::sched_AddExch_Vel_Temp_CC(SchedulerP           & sched,
                                         const PatchSet       * patches,
                                         const MaterialSubset * ice_matls,
                                         const MaterialSubset * mpm_matls,
                                         const MaterialSubset * press_matl,
                                         const MaterialSet    * all_matls,
                                         customBC_globalVars  * BC_globalVars)
{
  //__________________________________
  // compute surface normal gradients
  schedComputeSurfaceNormal( sched, patches, press_matl);

  //__________________________________
  //  compute Mean Free Path
  schedComputeMeanFreePath( sched, patches );

  //__________________________________
  //
  string name = "SlipExch::addExch_Vel_Temp_CC";
  Task* t = scinew Task(name, this, &SlipExch::addExch_Vel_Temp_CC, BC_globalVars);

  printSchedule( patches, dbgExch, name );

  Ghost::GhostType  gn  = Ghost::None;
  Ghost::GhostType  gac = Ghost::AroundCells;

  t->requires( Task::OldDW,  Ilb->delTLabel,getLevel(patches));
  t->requires( Task::NewDW,  Mlb->gMassLabel,       mpm_matls,  gac, 1 );
  t->requires( Task::OldDW,  Mlb->NC_CCweightLabel, press_matl, gac, 1 );
  t->requires( Task::NewDW,  d_surfaceNormLabel,    mpm_matls,  gn,  0 );
                                // I C E
  t->requires( Task::OldDW,  Ilb->temp_CCLabel,       ice_matls, gn );
  t->requires( Task::NewDW,  Ilb->specific_heatLabel, ice_matls, gn );
  t->requires( Task::NewDW,  Ilb->gammaLabel,         ice_matls, gn );
  t->requires( Task::NewDW,  d_meanFreePathLabel,     ice_matls, gac, 1);

                                // A L L  M A T L S
  t->requires( Task::NewDW,  Ilb->mass_L_CCLabel,    gn );
  t->requires( Task::NewDW,  Ilb->mom_L_CCLabel,     gn );
  t->requires( Task::NewDW,  Ilb->int_eng_L_CCLabel, gn );
  t->requires( Task::NewDW,  Ilb->sp_vol_CCLabel,    gn );
  t->requires( Task::NewDW,  Ilb->vol_frac_CCLabel,  gn );

  computesRequires_CustomBCs(t, "CC_Exchange", Ilb, ice_matls, BC_globalVars);

  t->computes( Ilb->Tdot_CCLabel );
  t->computes( Ilb->mom_L_ME_CCLabel );
  t->computes( Ilb->eng_L_ME_CCLabel );
  t->computes( d_vel_CCTransposedLabel );

  t->modifies( Ilb->temp_CCLabel, mpm_matls );
  t->modifies( Ilb->vel_CCLabel,  mpm_matls );

  sched->addTask(t, patches, all_matls);
}

//______________________________________________________________________
//
void SlipExch::addExch_Vel_Temp_CC(const ProcessorGroup * pg,
                                   const PatchSubset    * patches,
                                   const MaterialSubset * matls,
                                   DataWarehouse        * old_dw,
                                   DataWarehouse        * new_dw,
                                   customBC_globalVars  * BC_globalVars)
{
  timeStep_vartype timeStep;
  old_dw->get(timeStep, Ilb->timeStepLabel);
  bool isNotInitialTimeStep = (timeStep > 0);
  const Level* level = getLevel(patches);

  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask(patches, patch, dbgExch,"Doing SlipExch::addExchangeToMomentumAndEnergy");

    int numMPMMatls = d_sharedState->getNumMPMMatls();
    int numICEMatls = d_sharedState->getNumICEMatls();
    int numMatls    = numMPMMatls + numICEMatls;
    Ghost::GhostType  gn = Ghost::None;
    Ghost::GhostType  gac = Ghost::AroundCells;

    delt_vartype delT;
    old_dw->get(delT, Ilb->delTLabel, level);
    //Vector zero(0.,0.,0.);

    // Create arrays for the grid data
    std::vector< CCVariable<double> > cv(numMatls);
    std::vector< CCVariable<double> > Temp_CC(numMatls);
    std::vector< constCCVariable<double> > gamma(numMatls);
    std::vector< constCCVariable<double> > vol_frac_CC(numMatls);
    std::vector< constCCVariable<double> > sp_vol_CC(numMatls);
    std::vector< constCCVariable<Vector> > mom_L(numMatls);
    std::vector< constCCVariable<double> > int_eng_L(numMatls);
    std::vector< constCCVariable<double> > mass_L(numMatls);
    std::vector< constCCVariable<double> > old_temp(numMatls);
    std::vector< constCCVariable<double> > meanFreePath(numMatls);  // This mean free path does not have viscosity in it, which is okay per how it is used in this code
    constNCVariable<double> NC_CCweight;
    constNCVariable<double> NCsolidMass;

    // Create variables for the results
    std::vector< CCVariable<Vector> > mom_L_ME(numMatls);
    std::vector< CCVariable<Vector> > vel_CC(numMatls);
    std::vector< CCVariable<double> > int_eng_L_ME(numMatls);
    std::vector< CCVariable<double> > Tdot(numMatls);
    std::vector< CCVariable<Vector> > vel_T_CC(numMatls);

    // local variables
    double b[MAX_MATLS];
    Vector vel_T[MAX_MATLS];
    Vector notUsed;

    FastMatrix Q(3,3);                                // Q is always 3x3, because it is based on 3D, not number of materials
    FastMatrix a(numMatls, numMatls);
    FastMatrix k(numMatls, numMatls);
    FastMatrix K(numMatls, numMatls);
    FastMatrix Kslip(numMatls, numMatls);
    FastMatrix h(numMatls, numMatls);
    FastMatrix H(numMatls, numMatls);

    // Initialize
    Q.zero();
    a.zero();
    k.zero();
    K.zero();
    Kslip.zero();
    h.zero();
    H.zero();

    d_exchCoeff->getConstantExchangeCoeff(k,h);

/*`==========TESTING==========*/
    FastMatrix K_constant(numMatls, numMatls);
    K_constant.copy(k);
/*===========TESTING==========`*/


    for (int m = 0; m < numMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);

      int indx = matl->getDWIndex();
      new_dw->allocateTemporary(cv[m], patch);

      if(mpm_matl){                 // M P M
        CCVariable<double> oldTemp;
        new_dw->getCopy(oldTemp,          Ilb->temp_CCLabel,indx, patch,gn,0);
        new_dw->getModifiable(vel_CC[m],  Ilb->vel_CCLabel, indx, patch);
        new_dw->getModifiable(Temp_CC[m], Ilb->temp_CCLabel,indx, patch);
        old_temp[m] = oldTemp;
        cv[m].initialize(mpm_matl->getSpecificHeat());

        if(indx== d_solidMatlIndx){
          new_dw->get(NCsolidMass, Mlb->gMassLabel, indx,patch,gac,1);
        }
      }
      if(ice_matl){                 // I C E
        constCCVariable<double> cv_ice;
        old_dw->get(old_temp[m],     Ilb->temp_CCLabel,      indx, patch, gn, 0);
        new_dw->get(cv_ice,          Ilb->specific_heatLabel,indx, patch, gn, 0);
        new_dw->get(gamma[m],        Ilb->gammaLabel,        indx, patch, gn, 0);
        new_dw->get(meanFreePath[m], d_meanFreePathLabel,    indx, patch, gac,1);

        new_dw->allocateTemporary(vel_CC[m],  patch);
        new_dw->allocateTemporary(Temp_CC[m], patch);
        cv[m].copyData(cv_ice);
      }                             // A L L  M A T L S

      new_dw->get(mass_L[m],        Ilb->mass_L_CCLabel,   indx, patch,gn, 0);
      new_dw->get(sp_vol_CC[m],     Ilb->sp_vol_CCLabel,   indx, patch,gn, 0);
      new_dw->get(mom_L[m],         Ilb->mom_L_CCLabel,    indx, patch,gn, 0);
      new_dw->get(int_eng_L[m],     Ilb->int_eng_L_CCLabel,indx, patch,gn, 0);
      new_dw->get(vol_frac_CC[m],   Ilb->vol_frac_CCLabel, indx, patch,gn, 0);

      new_dw->allocateAndPut(Tdot[m],         Ilb->Tdot_CCLabel,       indx,patch);
      new_dw->allocateAndPut(mom_L_ME[m],     Ilb->mom_L_ME_CCLabel,   indx,patch);
      new_dw->allocateAndPut(int_eng_L_ME[m], Ilb->eng_L_ME_CCLabel,   indx,patch);
      new_dw->allocateAndPut(vel_T_CC[m],     d_vel_CCTransposedLabel, indx,patch);

      vel_T_CC[m].initialize(Vector(0,0,0));
    }  // all matls

    old_dw->get(NC_CCweight, Mlb->NC_CCweightLabel, 0, patch,gac,1);

    // Convert momenta to velocities and internal energy to Temp
    // shouldn't the matl loop be outside of the cell iterator loop for speed?? --Todd

    for(CellIterator iter = patch->getExtraCellIterator(); !iter.done();iter++){
      IntVector c = *iter;
      for (int m = 0; m < numMatls; m++) {
        Temp_CC[m][c] = int_eng_L[m][c]/(mass_L[m][c]*cv[m][c]);
        vel_CC[m][c]  = mom_L[m][c]/mass_L[m][c];
      }
    }

    //__________________________________
    //
    // Begin here: Start iterations over all cells in domain and get an idea of where the slip is occurring
    for(CellIterator iter = patch->getCellIterator(); !iter.done();iter++){
      IntVector c = *iter;

      Q.identity();  // Transformation matrix is initialized
      Kslip.copy(k);
      H.copy(h);

      //__________________________________
      //  Determine Exchange Coefficients
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

      //__________________________________
      //  If cell is a surface cell modify exchange coefficients
      if ((MaxMass-MinMass)/MaxMass == 1.0 && (MaxMass > (d_TINY_RHO * vol))){   // If the cell c is a surface shell

	 computeSurfaceRotationMatrix(Q, notUsed, nodeIdx, NCsolidMass, NC_CCweight, dx); // Makes Q at each cell c

	 double A_V = 1.0/( dx.x()*fabs(Q(1,0)) + dx.y()*fabs(Q(1,1)) + dx.z()*fabs(Q(1,2)) );

	 int gm = d_fluidMatlIndx;
	 int sm = d_solidMatlIndx;
	 double av = d_momentum_accommodation_coeff;
	 double at = d_thermal_accommodation_coeff;

	 double Beta_v = (2 - av)/av;
	 double Beta_t = ((2 - at)/at) * (2/(1 + gamma[gm][c])) * (1/cv[gm][c]);    // Does this assume an ideal gas????  -Todd

	 Kslip(gm,sm) = A_V / (Beta_v * meanFreePath[gm][c] * vol_frac_CC[sm][c]); // DOUBLE CHECK the material index of meanFreePath  -Todd

	 if(Kslip(gm,sm) > k(gm,sm)) {
	   Kslip(gm,sm) = k(gm,sm);                                            // K > Kslip in reality, so if Kslip > K in computation, fix this.
	 }

	 Kslip(sm,gm) = Kslip(gm,sm);                                          // Make the inverse indices of the Kslip matrix equal to each other

        H(gm,sm) = A_V / (Beta_t * meanFreePath[gm][c] * vol_frac_CC[sm][c]); // The viscosity does not appear here because it's taken out of lambda

	 if(H(gm,sm) > h(gm,sm)) {
	   H(gm,sm) = h(gm,sm);                                                // H > Hslip in reality, so if Hslip > H in computation, fix this.
	 }
	 H(sm,gm) = H(gm,sm);                                                  // Make inverse indices of the Kslip matrix equal to each other
      }  // if a surface cell


      //---------- M O M E N T U M   E X C H A N G E
      for(int i = 0; i < 3; i++) {

	 if(i != 1) { // if the principle direction is NOT the normal to the surface
	   K.copy(Kslip);
	 }

	/*`==========TESTING==========*/
	  // K.copy(K_constant);
	  // Q.identity();
        /*===========TESTING==========`*/
        //__________________________________
        //  coordinate Transformation
	 for(int m = 0; m < numMatls; m++) {
	   vel_T[m][i] = 0;

	   for(int j = 0; j < 3; j++) {
	     vel_T[m][i] += Q(i,j) * vel_CC[m][c][j]; // Use linear algegra to compute the transformed velocity
	   }
	 }

	 for(int m = 0; m < numMatls; m++) {// loop over all materials again
	   double adiag = 1.0;
	   b[m] = 0.0;

	   for(int n = 0; n < numMatls; n++) {
	     a(m,n) = - delT * vol_frac_CC[m][c] * sp_vol_CC[m][c] * K(m,n);
	     adiag -= a(m,n);
	     b[m]  -= a(m,n) * (vel_T[n][i] - vel_T[m][i]);
	   }
	   a(m,m) = adiag;
	 }

	 a.destructiveSolve(b);

	 for(int m = 0; m < numMatls; m++) {
	   vel_T[m][i] = b[m];
	 }
      } // loop over directions

      //__________________________________
      //  coordinate transformation
      for(int m = 0; m < numMatls; m++) {
        vel_T_CC[m][c] = vel_T[m];

        for(int i = 0; i < 3; i++) {
	   for(int j = 0; j < 3; j++) {
	     vel_CC[m][c][i] += Q(j,i) * vel_T[m][j]; // Use the transpose of Q to back out the velocity in the cartesian system
	   }
	 }
      }

      //---------- E N E R G Y   E X C H A N G E
      if(d_exchCoeff->d_heatExchCoeffModel != "constant"){
        d_exchCoeff->getVariableExchangeCoeff( K, H, c, mass_L);
      }

      for(int m = 0; m < numMatls; m++) {
        double adiag = 1.0;
        b[m] = 0.0;

        for(int n = 0; n < numMatls; n++) {
          a(m,n) = - delT * vol_frac_CC[m][c] * sp_vol_CC[m][c] * H(m,n) / cv[m][c];  // Here H has already been changed into Hslip, and Hslip is the same in every direction.
          adiag -= a(m,n);
          b[m]  -= a(m,n) * (Temp_CC[n][c] - Temp_CC[m][c]);
        }
        a(m,m) = adiag;
      }

      a.destructiveSolve(b);

      for(int m = 0; m < numMatls; m++) {
        Temp_CC[m][c] += b[m];
      }
    }  //end CellIterator loop

    //__________________________________
    //  Boundary Condition Code
    if( BC_globalVars->usingLodi || BC_globalVars->usingMicroSlipBCs){

      std::vector<CCVariable<double> > temp_CC_Xchange(numMatls);
      std::vector<CCVariable<Vector> > vel_CC_Xchange(numMatls);

      for (int m = 0; m < numMatls; m++) {
        Material* matl = d_sharedState->getMaterial(m);
        int indx = matl->getDWIndex();

        new_dw->allocateAndPut(temp_CC_Xchange[m], Ilb->temp_CC_XchangeLabel, indx, patch);
        new_dw->allocateAndPut(vel_CC_Xchange[m],  Ilb->vel_CC_XchangeLabel,  indx, patch);
        vel_CC_Xchange[m].copy(vel_CC[m]);
        temp_CC_Xchange[m].copy(Temp_CC[m]);
      }
    }

    //__________________________________
    //  Set boundary conditions
    for (int m = 0; m < numMatls; m++)  {
      Material* matl = d_sharedState->getMaterial( m );
      int indx = matl->getDWIndex();

      customBC_localVars* BC_localVars   = scinew customBC_localVars();
      preprocess_CustomBCs("CC_Exchange", old_dw, new_dw, Ilb, patch, indx, BC_globalVars, BC_localVars);

      setBC(vel_CC[m], "Velocity",   patch, d_sharedState, indx, new_dw,
                                                        BC_globalVars, BC_localVars, isNotInitialTimeStep);
      setBC(Temp_CC[m],"Temperature",gamma[m], cv[m], patch, d_sharedState,
                                         indx, new_dw,  BC_globalVars, BC_localVars, isNotInitialTimeStep);
#if SET_CFI_BC
//      set_CFI_BC<Vector>(vel_CC[m],  patch);
//      set_CFI_BC<double>(Temp_CC[m], patch);
#endif
      delete_CustomBCs( BC_globalVars, BC_localVars );
    }

    //__________________________________
    // Convert vars. primitive-> flux
    for(CellIterator iter = patch->getExtraCellIterator(); !iter.done();iter++){
      IntVector c = *iter;                                              // shouldn't the loop over matls be outside the cell iterator for speed. --Todd
      for (int m = 0; m < numMatls; m++) {
        int_eng_L_ME[m][c] = Temp_CC[m][c]*cv[m][c] * mass_L[m][c];
        mom_L_ME[m][c]     = vel_CC[m][c]           * mass_L[m][c];
        Tdot[m][c]         = (Temp_CC[m][c] - old_temp[m][c])/delT;
      }
    }
  } //patches
}

//______________________________________________________________________
//
void SlipExch::schedComputeMeanFreePath(SchedulerP       & sched,
                                        const PatchSet   * patches)
{
  std::string tName =  "SlipExch::computeMeanFreePath";
  Task* t = scinew Task(tName, this, &SlipExch::computeMeanFreePath);

  printSchedule(patches, dbgExch, tName);

  Ghost::GhostType  gn = Ghost::None;
  t->requires(Task::OldDW, Ilb->temp_CCLabel,       gn);
  t->requires(Task::OldDW, Ilb->sp_vol_CCLabel,     gn);
  t->requires(Task::NewDW, Ilb->gammaLabel,         gn);
  t->requires(Task::NewDW, Ilb->specific_heatLabel, gn);

  t->computes(d_meanFreePathLabel);

  const MaterialSet* ice_matls = d_sharedState->allICEMaterials();
  sched->addTask(t, patches, ice_matls);
}

//______________________________________________________________________
//
void SlipExch::computeMeanFreePath(const ProcessorGroup *,
                                   const PatchSubset    * patches,
                                   const MaterialSubset *,
                                   DataWarehouse        * old_dw,
                                   DataWarehouse        * new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    printTask(patches, patch, dbgExch, "Doing SlipExch::computeMeanFreePath" );

    int numMatls = d_sharedState->getNumICEMatls();
    Ghost::GhostType  gn = Ghost::None;

    for (int m = 0; m < numMatls; m++) {
      ICEMaterial* ice_matl = d_sharedState->getICEMaterial(m);
      int indx = ice_matl->getDWIndex();
      constCCVariable<double> temp;
      constCCVariable<double> sp_vol;
      constCCVariable<double> gamma;
      constCCVariable<double> cv;
      CCVariable<double> meanFreePath;

      old_dw->get(temp,   Ilb->temp_CCLabel,      indx,patch, gn,0);
      old_dw->get(sp_vol, Ilb->sp_vol_CCLabel,    indx,patch, gn,0);
      new_dw->get(gamma,  Ilb->gammaLabel,        indx,patch, gn,0);
      new_dw->get(cv,     Ilb->specific_heatLabel,indx,patch, gn,0);

      new_dw->allocateAndPut(meanFreePath, d_meanFreePathLabel, indx, patch);
      meanFreePath.initialize(0.0);

      //This is really the mean free path divided by the dynamic viscosity
      for (CellIterator iter=patch->getExtraCellIterator();!iter.done();iter++){
        IntVector c = *iter;
        meanFreePath[c] = sp_vol[c] /sqrt(2 * cv[c] * (gamma[c] - 1) * temp[c] / M_PI);
      }
    }
  }
}

//______________________________________________________________________
//
void SlipExch::computeSurfaceRotationMatrix(FastMatrix               & Q,
                                            Vector                   & GradRho,
                                            IntVector                *nodeIdx,
                                            constNCVariable<double>  &NCsolidMass,
                                            constNCVariable<double>  &NC_CCweight,
                                            Vector &dx)
{
  double gradRhoX = 0.25 * (
                            (NCsolidMass[nodeIdx[0]]*NC_CCweight[nodeIdx[0]]+
                             NCsolidMass[nodeIdx[1]]*NC_CCweight[nodeIdx[1]]+
                             NCsolidMass[nodeIdx[2]]*NC_CCweight[nodeIdx[2]]+
                             NCsolidMass[nodeIdx[3]]*NC_CCweight[nodeIdx[3]])
                            -
                            (NCsolidMass[nodeIdx[4]]*NC_CCweight[nodeIdx[4]]+
                             NCsolidMass[nodeIdx[5]]*NC_CCweight[nodeIdx[5]]+
                             NCsolidMass[nodeIdx[6]]*NC_CCweight[nodeIdx[6]]+
                             NCsolidMass[nodeIdx[7]]*NC_CCweight[nodeIdx[7]])
                            )/dx.x();

  double gradRhoY = 0.25 * (
                            (NCsolidMass[nodeIdx[0]]*NC_CCweight[nodeIdx[0]]+
                             NCsolidMass[nodeIdx[1]]*NC_CCweight[nodeIdx[1]]+
                             NCsolidMass[nodeIdx[4]]*NC_CCweight[nodeIdx[4]]+
                             NCsolidMass[nodeIdx[5]]*NC_CCweight[nodeIdx[5]])
                            -
                            (NCsolidMass[nodeIdx[2]]*NC_CCweight[nodeIdx[2]]+
                             NCsolidMass[nodeIdx[3]]*NC_CCweight[nodeIdx[3]]+
                             NCsolidMass[nodeIdx[6]]*NC_CCweight[nodeIdx[6]]+
                             NCsolidMass[nodeIdx[7]]*NC_CCweight[nodeIdx[7]])
                            )/dx.y();

/*`==========TESTING==========*/
#if 0

  double gradRhoZ = 0.25 * (
                            (NCsolidMass[nodeIdx[1]]*NC_CCweight[nodeIdx[1]]+
                             NCsolidMass[nodeIdx[3]]*NC_CCweight[nodeIdx[3]]+
                             NCsolidMass[nodeIdx[5]]*NC_CCweight[nodeIdx[5]]+
                             NCsolidMass[nodeIdx[7]]*NC_CCweight[nodeIdx[7]])
                            -
                            (NCsolidMass[nodeIdx[0]]*NC_CCweight[nodeIdx[0]]+
                             NCsolidMass[nodeIdx[2]]*NC_CCweight[nodeIdx[2]]+
                             NCsolidMass[nodeIdx[4]]*NC_CCweight[nodeIdx[4]]+
                             NCsolidMass[nodeIdx[6]]*NC_CCweight[nodeIdx[6]])
                            )/dx.z();
#endif
/*===========TESTING==========`*/


  double gradRhoZ = 0.25 * (
                            (NCsolidMass[nodeIdx[0]]*NC_CCweight[nodeIdx[0]]+
                             NCsolidMass[nodeIdx[2]]*NC_CCweight[nodeIdx[2]]+
                             NCsolidMass[nodeIdx[4]]*NC_CCweight[nodeIdx[4]]+
                             NCsolidMass[nodeIdx[6]]*NC_CCweight[nodeIdx[6]])
                           -
                            (NCsolidMass[nodeIdx[1]]*NC_CCweight[nodeIdx[1]]+
                             NCsolidMass[nodeIdx[3]]*NC_CCweight[nodeIdx[3]]+
                             NCsolidMass[nodeIdx[5]]*NC_CCweight[nodeIdx[5]]+
                             NCsolidMass[nodeIdx[7]]*NC_CCweight[nodeIdx[7]])
                            )/dx.z();


  double absGradRho = sqrt(gradRhoX*gradRhoX + gradRhoY*gradRhoY + gradRhoZ*gradRhoZ );
  GradRho= Vector(gradRhoX/absGradRho, gradRhoY/absGradRho, gradRhoZ/absGradRho);


  if(fabs(GradRho[1]) > 0.9999) {
    Q.identity();
  }else { //if coord.'s are not already aligned with surface
    Q(1,0) = GradRho[0];
    Q(1,1) = GradRho[1];
    Q(1,2) = GradRho[2];

    double sqrtTerm = sqrt(1 - Q(1,1) * Q(1,1));
    double invTerm  = 1.0/sqrtTerm;

    Q(0,0) = Q(1,1) * Q(1,0) * invTerm;
    Q(0,1) =-sqrtTerm;
    Q(0,2) = Q(1,1) * Q(1,2) * invTerm;
    Q(2,0) =-Q(1,2) * invTerm;
    Q(2,1) = 0.0;
    Q(2,2) = Q(1,0) * invTerm;
  }
}

