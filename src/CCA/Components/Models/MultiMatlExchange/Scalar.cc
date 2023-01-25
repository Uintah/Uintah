/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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

#include <CCA/Components/Models/MultiMatlExchange/ExchangeModel.h>
#include <CCA/Components/Models/MultiMatlExchange/Scalar.h>

#include <CCA/Components/ICE/CustomBCs/BoundaryCond.h>
#include <CCA/Components/ICE/Materials/ICEMaterial.h>
#include <CCA/Components/MPM/Materials/MPMMaterial.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/Material.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/MaterialManagerP.h>
#include <Core/Grid/Variables/ComputeSet.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/ProblemSpec/ProblemSpec.h>

#include <ostream>                         // for operator<<, basic_ostream
#include <vector>

#define MAX_MATLS 8

using namespace Uintah;
using namespace ExchangeModels;
using namespace std;
extern DebugStream dbgExch;
//______________________________________________________________________
//
ScalarExch::ScalarExch( const ProblemSpecP     & prob_spec,
                        const MaterialManagerP & materialManager,
                        const bool with_mpm )
  : ExchangeModel( prob_spec, materialManager, with_mpm )
{
  d_exchCoeff = scinew ExchangeCoefficients();
}

ScalarExch::~ScalarExch()
{
  delete d_exchCoeff;
}

//______________________________________________________________________
//
void ScalarExch::problemSetup(const ProblemSpecP & prob_spec)
{
  // read in the exchange coefficients
  ProblemSpecP notUsed;
  d_exchCoeff->problemSetup( prob_spec, d_numMatls, notUsed );
}

//______________________________________________________________________
//
void ScalarExch::outputProblemSpec(ProblemSpecP & matl_ps )
{
  ProblemSpecP notUsed;
  d_exchCoeff->outputProblemSpec(matl_ps, notUsed);
}

//______________________________________________________________________
//  These tasks are called before semi-implicit pressure solve.
//  All computed variables live in the parent NewDW
void ScalarExch::sched_PreExchangeTasks(SchedulerP           & sched,
                                        const PatchSet       * patches,
                                        const MaterialSubset * ice_matls,
                                        const MaterialSubset * mpm_matls,
                                        const MaterialSet    * allMatls)
{

  //__________________________________
  // compute surface normal and isSurfaceCell
  if(d_exchCoeff->convective() && mpm_matls){
    schedComputeSurfaceNormal( sched, patches, mpm_matls );
  }
}
//______________________________________________________________________
//
void ScalarExch::sched_AddExch_VelFC( SchedulerP            & sched,
                                      const PatchSet        * patches,
                                      const MaterialSubset  * ice_matls,
                                      const MaterialSubset  * mpm_matls,
                                      const MaterialSet     * all_matls,
                                      customBC_globalVars   * BC_globalVars,
                                      const bool recursion )
{
  Task* t = scinew Task( "ScalarExch::addExch_VelFC", this, &ScalarExch::addExch_VelFC,
                         BC_globalVars, recursion);

  printSchedule( patches, dbgExch, "ScalarExch::sched_AddExch_VelFC" );

  if(recursion) {
    t->requires(Task::ParentOldDW, Ilb->delTLabel,getLevel(patches));
  } else {
    t->requires(Task::OldDW,       Ilb->delTLabel,getLevel(patches));
  }

  Ghost::GhostType  gac   = Ghost::AroundCells;
  Ghost::GhostType  gaf_X = Ghost::AroundFacesX;
  Ghost::GhostType  gaf_Y = Ghost::AroundFacesY;
  Ghost::GhostType  gaf_Z = Ghost::AroundFacesZ;
  //__________________________________
  // define parent data warehouse
  // change the definition of parent(old/new)DW
  // when using semi-implicit pressure solve
  Task::WhichDW pNewDW = Task::NewDW;
  if(recursion) {
    pNewDW  = Task::ParentNewDW;
  }

  // All matls
  t->requires( pNewDW,      Ilb->sp_vol_CCLabel,  gac,   1);
  t->requires( pNewDW,      Ilb->vol_frac_CCLabel,gac,   1);
  t->requires( Task::NewDW, Ilb->uvel_FCLabel,    gaf_X, 1);
  t->requires( Task::NewDW, Ilb->vvel_FCLabel,    gaf_Y, 1);
  t->requires( Task::NewDW, Ilb->wvel_FCLabel,    gaf_Z, 1);

  computesRequires_CustomBCs(t, "velFC_Exchange", Ilb, ice_matls,
                             BC_globalVars, recursion);

  t->computes( Ilb->sp_volX_FCLabel );
  t->computes( Ilb->sp_volY_FCLabel );
  t->computes( Ilb->sp_volZ_FCLabel );
  t->computes( Ilb->uvel_FCMELabel );
  t->computes( Ilb->vvel_FCMELabel );
  t->computes( Ilb->wvel_FCMELabel );

  sched->addTask(t, patches, all_matls);
}

/* _____________________________________________________________________
 Purpose~   Add the exchange contribution to vel_FC and compute
            sp_vol_FC for implicit Pressure solve
_____________________________________________________________________*/
template<class constSFC, class SFC>
void ScalarExch::vel_FC_exchange( CellIterator    iter,
                                  IntVector       adj_offset,
                                  int             numMatls,
                                  FastMatrix  &   K,
                                  double          delT,
                                  std::vector<constCCVariable<double> >& vol_frac_CC,
                                  std::vector<constCCVariable<double> >& sp_vol_CC,
                                  std::vector< constSFC>               & vel_FC,
                                  std::vector< SFC >                   & sp_vol_FC,
                                  std::vector< SFC >                   & vel_FCME)
{
  //__________________________________
  //          Single Material
  if (numMatls == 1){

    // put in tmp arrays for speed!
    constCCVariable<double>& sp_vol_tmp = sp_vol_CC[0];
    constSFC& vel_FC_tmp  = vel_FC[0];
    SFC& vel_FCME_tmp     = vel_FCME[0];
    SFC& sp_vol_FC_tmp    = sp_vol_FC[0];

    for(;!iter.done(); iter++){
      IntVector c = *iter;
      IntVector adj = c + adj_offset;
      double sp_vol     = sp_vol_tmp[c];
      double sp_vol_adj = sp_vol_tmp[adj];
      double sp_volFC   = 2.0 * (sp_vol_adj * sp_vol)/
                                (sp_vol_adj + sp_vol);

      sp_vol_FC_tmp[c] = sp_volFC;
      vel_FCME_tmp[c] = vel_FC_tmp[c];
    }
  }
  else{         // Multi-material
    double b[MAX_MATLS];
    double b_sp_vol[MAX_MATLS];
    double vel[MAX_MATLS];
    double tmp[MAX_MATLS];
    FastMatrix a(numMatls, numMatls);

    for(;!iter.done(); iter++){
      IntVector c = *iter;
      IntVector adj = c + adj_offset;

      //__________________________________
      //   Compute beta and off diagonal term of
      //   Matrix A, this includes b[m][m].
      //  You need to make sure that mom_exch_coeff[m][m] = 0

      // - Form diagonal terms of Matrix (A)
      //  - Form RHS (b)
      for(int m = 0; m < numMatls; m++)  {
        b_sp_vol[m] = 2.0 * (sp_vol_CC[m][adj] * sp_vol_CC[m][c])/
                            (sp_vol_CC[m][adj] + sp_vol_CC[m][c]);

        tmp[m] = -0.5 * delT * (vol_frac_CC[m][adj] + vol_frac_CC[m][c]);
        vel[m] = vel_FC[m][c];
      }

      for(int m = 0; m < numMatls; m++)  {
        double betasum = 1;
        double bsum    = 0;
        double bm      = b_sp_vol[m];
        double vm      = vel[m];

        for(int n = 0; n < numMatls; n++)  {
          double b = bm * tmp[n] * K(n,m);
          a(m,n)   = b;
          betasum -= b;
          bsum    -= b * (vel[n] - vm);
        }
        a(m,m) = betasum;
        b[m] = bsum;
      }

      //__________________________________
      //  - solve and backout velocities

      a.destructiveSolve(b, b_sp_vol);
      //  For implicit solve we need sp_vol_FC
      for(int m = 0; m < numMatls; m++) {
        vel_FCME[m][c]  = vel_FC[m][c] + b[m];
        sp_vol_FC[m][c] = b_sp_vol[m];        // only needed by implicit Pressure
      }
    }  // iterator
  }  // multiple materials
}

/*_____________________________________________________________________

 Purpose~
   This function adds the momentum exchange contribution to the
   existing face-centered velocities


                   (A)                              (X)
| (1+b12 + b13)     -b12          -b23          |   |del_FC[1]  |
|                                               |   |           |
| -b21              (1+b21 + b23) -b32          |   |del_FC[2]  |
|                                               |   |           |
| -b31              -b32          (1+b31 + b32) |   |del_FC[2]  |

                        =

                        (B)
| b12( uvel_FC[2] - uvel_FC[1] ) + b13 ( uvel_FC[3] -uvel_FC[1])    |
|                                                                   |
| b21( uvel_FC[1] - uvel_FC[2] ) + b23 ( uvel_FC[3] -uvel_FC[2])    |
|                                                                   |
| b31( uvel_FC[1] - uvel_FC[3] ) + b32 ( uvel_FC[2] -uvel_FC[3])    |

 References: see "A Cell-Centered ICE method for multiphase flow simulations"
 by Kashiwa, above equation 4.13.
 _____________________________________________________________________  */
void ScalarExch::addExch_VelFC( const ProcessorGroup * pg,
                                const PatchSubset    * patches,
                                const MaterialSubset * matls,
                                DataWarehouse        * old_dw,
                                DataWarehouse        * new_dw,
                                customBC_globalVars  * BC_globalVars,
                                const bool recursion )
{
  const Level* level = getLevel(patches);
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask(patches, patch, dbgExch, "Doing ScalarExch::addExch_VelFC" );

    // change the definition of parent(old/new)DW
    // when implicit
    DataWarehouse* pNewDW;
    DataWarehouse* pOldDW;
    if(recursion) {
      pNewDW  = new_dw->getOtherDataWarehouse(Task::ParentNewDW);
      pOldDW  = new_dw->getOtherDataWarehouse(Task::ParentOldDW);
    } else {
      pNewDW  = new_dw;
      pOldDW  = old_dw;
    }

    delt_vartype delT;
    pOldDW->get(delT, Ilb->delTLabel, level);

    std::vector< constCCVariable<double> >   sp_vol_CC(d_numMatls);
    std::vector< constCCVariable<double> >   vol_frac_CC(d_numMatls);
    std::vector< constSFCXVariable<double> > uvel_FC(d_numMatls);
    std::vector< constSFCYVariable<double> > vvel_FC(d_numMatls);
    std::vector< constSFCZVariable<double> > wvel_FC(d_numMatls);

    std::vector< SFCXVariable<double> > uvel_FCME(d_numMatls);
    std::vector< SFCYVariable<double> > vvel_FCME(d_numMatls);
    std::vector< SFCZVariable<double> > wvel_FCME(d_numMatls);

    std::vector< SFCXVariable<double> > sp_vol_XFC(d_numMatls);
    std::vector< SFCYVariable<double> > sp_vol_YFC(d_numMatls);
    std::vector< SFCZVariable<double> > sp_vol_ZFC(d_numMatls);

    // lowIndex is the same for all vel_FC
    IntVector lowIndex(patch->getExtraSFCXLowIndex());

    // Extract the momentum exchange coefficients
    FastMatrix K(d_numMatls, d_numMatls), junk(d_numMatls, d_numMatls);

    K.zero();

    d_exchCoeff->getConstantExchangeCoeff( K, junk);

    Ghost::GhostType  gac   = Ghost::AroundCells;
    Ghost::GhostType  gaf_X = Ghost::AroundFacesX;
    Ghost::GhostType  gaf_Y = Ghost::AroundFacesY;
    Ghost::GhostType  gaf_Z = Ghost::AroundFacesZ;

    for(int m = 0; m < d_numMatls; m++) {
      Material* matl = d_matlManager->getMaterial( m );
      int indx = matl->getDWIndex();

      pNewDW->get( sp_vol_CC[m],    Ilb->sp_vol_CCLabel,  indx, patch, gac,   1 );
      pNewDW->get( vol_frac_CC[m],  Ilb->vol_frac_CCLabel,indx, patch, gac,   1 );
      new_dw->get( uvel_FC[m],      Ilb->uvel_FCLabel,    indx, patch, gaf_X, 1 );
      new_dw->get( vvel_FC[m],      Ilb->vvel_FCLabel,    indx, patch, gaf_Y, 1 );
      new_dw->get( wvel_FC[m],      Ilb->wvel_FCLabel,    indx, patch, gaf_Z, 1 );

      new_dw->allocateAndPut( uvel_FCME[m],  Ilb->uvel_FCMELabel, indx, patch );
      new_dw->allocateAndPut( vvel_FCME[m],  Ilb->vvel_FCMELabel, indx, patch );
      new_dw->allocateAndPut( wvel_FCME[m],  Ilb->wvel_FCMELabel, indx, patch );

      new_dw->allocateAndPut( sp_vol_XFC[m], Ilb->sp_volX_FCLabel,indx, patch );
      new_dw->allocateAndPut( sp_vol_YFC[m], Ilb->sp_volY_FCLabel,indx, patch );
      new_dw->allocateAndPut( sp_vol_ZFC[m], Ilb->sp_volZ_FCLabel,indx, patch );

      uvel_FCME[m].initialize(0.0,  lowIndex,patch->getExtraSFCXHighIndex());
      vvel_FCME[m].initialize(0.0,  lowIndex,patch->getExtraSFCYHighIndex());
      wvel_FCME[m].initialize(0.0,  lowIndex,patch->getExtraSFCZHighIndex());

      sp_vol_XFC[m].initialize(0.0, lowIndex,patch->getExtraSFCXHighIndex());
      sp_vol_YFC[m].initialize(0.0, lowIndex,patch->getExtraSFCYHighIndex());
      sp_vol_ZFC[m].initialize(0.0, lowIndex,patch->getExtraSFCZHighIndex());
    }

    vector<IntVector> adj_offset(3);
    adj_offset[0] = IntVector(-1, 0, 0);    // X faces
    adj_offset[1] = IntVector(0, -1, 0);    // Y faces
    adj_offset[2] = IntVector(0,  0, -1);   // Z faces

    CellIterator XFC_iterator = patch->getSFCXIterator();
    CellIterator YFC_iterator = patch->getSFCYIterator();
    CellIterator ZFC_iterator = patch->getSFCZIterator();

    //__________________________________
    //  tack on exchange contribution
    vel_FC_exchange<constSFCXVariable<double>, SFCXVariable<double> >
                    (XFC_iterator,
                    adj_offset[0],  d_numMatls,    K,
                    delT,           vol_frac_CC, sp_vol_CC,
                    uvel_FC,        sp_vol_XFC,  uvel_FCME);

    vel_FC_exchange<constSFCYVariable<double>, SFCYVariable<double> >
                    (YFC_iterator,
                    adj_offset[1],  d_numMatls,    K,
                    delT,           vol_frac_CC, sp_vol_CC,
                    vvel_FC,        sp_vol_YFC,  vvel_FCME);

    vel_FC_exchange<constSFCZVariable<double>, SFCZVariable<double> >
                    (ZFC_iterator,
                    adj_offset[2],  d_numMatls,    K,
                    delT,           vol_frac_CC, sp_vol_CC,
                    wvel_FC,        sp_vol_ZFC,  wvel_FCME);

    //________________________________
    //  Boundary Conditons
    for (int m = 0; m < d_numMatls; m++)  {
      Material* matl = d_matlManager->getMaterial( m );
      int indx = matl->getDWIndex();

      customBC_localVars* BC_localVars = scinew customBC_localVars();
      BC_localVars->recursiveTask = recursion;

      preprocess_CustomBCs("velFC_Exchange",pOldDW, pNewDW, Ilb,  patch, indx,
                            BC_globalVars, BC_localVars);

      setBC<SFCXVariable<double> >(uvel_FCME[m], "Velocity", patch, indx,
                                    d_matlManager, BC_globalVars, BC_localVars);
      setBC<SFCYVariable<double> >(vvel_FCME[m], "Velocity", patch, indx,
                                    d_matlManager, BC_globalVars, BC_localVars);
      setBC<SFCZVariable<double> >(wvel_FCME[m], "Velocity", patch, indx,
                                    d_matlManager, BC_globalVars, BC_localVars);
      delete_CustomBCs( BC_globalVars, BC_localVars );
    }
  }  // patch loop
}
//______________________________________________________________________
//
void ScalarExch::sched_AddExch_Vel_Temp_CC( SchedulerP           & sched,
                                            const PatchSet       * patches,
                                            const MaterialSubset * ice_matls,
                                            const MaterialSubset * mpm_matls,
                                            const MaterialSet    * all_matls,
                                            customBC_globalVars  * BC_globalVars )
{
  Task* t = 0;
  string tName = "null";

  if( d_numMatls == 1 ){
    tName = "ScalarExch::addExch_Vel_Temp_CC_1matl";
    t = scinew Task( tName, this, &ScalarExch::addExch_Vel_Temp_CC_1matl, BC_globalVars );
  } else {
    tName = "ScalarExch::addExch_Vel_Temp_CC";
    t = scinew Task( tName, this, &ScalarExch::addExch_Vel_Temp_CC, BC_globalVars );
  }

  printSchedule(patches, dbgExch, tName);

  Ghost::GhostType gn  = Ghost::None;
  Ghost::GhostType gac = Ghost::AroundCells;

  t->requires(Task::OldDW, Ilb->timeStepLabel);
  t->requires(Task::OldDW, Ilb->delTLabel,getLevel(patches));

  if(d_exchCoeff->convective() && mpm_matls ){
    t->requires( Task::NewDW, d_isSurfaceCellLabel, d_zero_matl, gac, 1 );
  }
                                // I C E
  t->requires(Task::OldDW,  Ilb->temp_CCLabel,      ice_matls, gn);
  t->requires(Task::NewDW,  Ilb->specific_heatLabel,ice_matls, gn);
  t->requires(Task::NewDW,  Ilb->gammaLabel,        ice_matls, gn);
                                // A L L  M A T L S
  t->requires(Task::NewDW,  Ilb->mass_L_CCLabel,    gn);
  t->requires(Task::NewDW,  Ilb->mom_L_CCLabel,     gn);
  t->requires(Task::NewDW,  Ilb->int_eng_L_CCLabel, gn);
  t->requires(Task::NewDW,  Ilb->sp_vol_CCLabel,    gn);
  t->requires(Task::NewDW,  Ilb->vol_frac_CCLabel,  gn);

  computesRequires_CustomBCs(t, "CC_Exchange", Ilb, ice_matls, BC_globalVars);

  t->computes(Ilb->Tdot_CCLabel);
  t->computes(Ilb->mom_L_ME_CCLabel);
  t->computes(Ilb->eng_L_ME_CCLabel);

  if (mpm_matls && mpm_matls->size() > 0){
    t->modifies(Ilb->temp_CCLabel, mpm_matls);
    t->modifies(Ilb->vel_CCLabel,  mpm_matls);
  }
  sched->addTask(t, patches, all_matls);
}

/*_____________________________________________________________________
   This task adds the  exchange contribution to the
   existing cell-centered velocity and temperature

                   (A)                              (X)
| (1+b12 + b13)     -b12          -b23          |   |del_data_CC[1]  |
|                                               |   |                |
| -b21              (1+b21 + b23) -b32          |   |del_data_CC[2]  |
|                                               |   |                |
| -b31              -b32          (1+b31 + b32) |   |del_data_CC[2]  |

                        =

                        (B)
| b12( data_CC[2] - data_CC[1] ) + b13 ( data_CC[3] -data_CC[1])    |
|                                                                   |
| b21( data_CC[1] - data_CC[2] ) + b23 ( data_CC[3] -data_CC[2])    |
|                                                                   |
| b31( data_CC[1] - data_CC[3] ) + b32 ( data_CC[2] -data_CC[3])    |

 Steps for each cell;
    1) Comute the beta coefficients
    2) Form and A matrix and B vector
    3) Solve for X[*]
    4) Add X[*] to the appropriate Lagrangian data
 - apply Boundary conditions to vel_CC and Temp_CC

 References: see "A Cell-Centered ICE method for multiphase flow simulations"
 by Kashiwa, above equation 4.13.
 _____________________________________________________________________  */
void ScalarExch::addExch_Vel_Temp_CC( const ProcessorGroup * pg,
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

    printTask(patches, patch, dbgExch, "Doing ScalarExch::addExch_Vel_Temp_CC" );

    int numMPMMatls = d_matlManager->getNumMatls( "MPM" );
    int numICEMatls = d_matlManager->getNumMatls( "ICE" );
    int numALLMatls = numMPMMatls + numICEMatls;
    Ghost::GhostType  gn = Ghost::None;

    delt_vartype delT;
    old_dw->get(delT, Ilb->delTLabel, level);
    //Vector zero(0.,0.,0.);

    // Create arrays for the grid data
    std::vector<CCVariable<double> > cv(numALLMatls);
    std::vector<CCVariable<double> > Temp_CC(numALLMatls);
    std::vector<constCCVariable<double> > gamma(numALLMatls);
    std::vector<constCCVariable<double> > vol_frac_CC(numALLMatls);
    std::vector<constCCVariable<double> > sp_vol_CC(numALLMatls);
    std::vector<constCCVariable<Vector> > mom_L(numALLMatls);
    std::vector<constCCVariable<double> > int_eng_L(numALLMatls);

    // Create variables for the results
    std::vector<CCVariable<Vector> > mom_L_ME(numALLMatls);
    std::vector<CCVariable<Vector> > vel_CC(numALLMatls);
    std::vector<CCVariable<double> > int_eng_L_ME(numALLMatls);
    std::vector<CCVariable<double> > Tdot(numALLMatls);
    std::vector<constCCVariable<double> > mass_L(numALLMatls);
    std::vector<constCCVariable<double> > old_temp(numALLMatls);

    double b[MAX_MATLS];
    Vector bb[MAX_MATLS];
    vector<double> sp_vol(numALLMatls);

    double tmp;
    FastMatrix beta(numALLMatls, numALLMatls);
    FastMatrix acopy(numALLMatls, numALLMatls);
    FastMatrix K(numALLMatls, numALLMatls);
    FastMatrix H(numALLMatls, numALLMatls);
    FastMatrix a(numALLMatls, numALLMatls);
    
    beta.zero();
    acopy.zero();
    K.zero();
    H.zero();
    a.zero();

    d_exchCoeff->getConstantExchangeCoeff( K, H);

    for (int m = 0; m < numALLMatls; m++) {
      Material* matl = d_matlManager->getMaterial( m );
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
      int indx = matl->getDWIndex();
      new_dw->allocateTemporary(cv[m], patch);

      if(mpm_matl){                 // M P M
        CCVariable<double> oldTemp;
        new_dw->getCopy(oldTemp,          Ilb->temp_CCLabel,indx,patch,gn,0);
        new_dw->getModifiable(vel_CC[m],  Ilb->vel_CCLabel, indx,patch);
        new_dw->getModifiable(Temp_CC[m], Ilb->temp_CCLabel,indx,patch);
        old_temp[m] = oldTemp;
        cv[m].initialize(mpm_matl->getSpecificHeat());
      }
      if(ice_matl){                 // I C E
        constCCVariable<double> cv_ice;
        old_dw->get(old_temp[m],   Ilb->temp_CCLabel,      indx, patch,gn,0);
        new_dw->get(cv_ice,        Ilb->specific_heatLabel,indx, patch,gn,0);
        new_dw->get(gamma[m],      Ilb->gammaLabel,        indx, patch,gn,0);

        new_dw->allocateTemporary(vel_CC[m],  patch);
        new_dw->allocateTemporary(Temp_CC[m], patch);
        cv[m].copyData(cv_ice);
      }                             // A L L  M A T L S

      new_dw->get(mass_L[m],        Ilb->mass_L_CCLabel,   indx, patch,gn, 0);
      new_dw->get(sp_vol_CC[m],     Ilb->sp_vol_CCLabel,   indx, patch,gn, 0);
      new_dw->get(mom_L[m],         Ilb->mom_L_CCLabel,    indx, patch,gn, 0);
      new_dw->get(int_eng_L[m],     Ilb->int_eng_L_CCLabel,indx, patch,gn, 0);
      new_dw->get(vol_frac_CC[m],   Ilb->vol_frac_CCLabel, indx, patch,gn, 0);
      new_dw->allocateAndPut(Tdot[m],        Ilb->Tdot_CCLabel,    indx,patch);
      new_dw->allocateAndPut(mom_L_ME[m],    Ilb->mom_L_ME_CCLabel,indx,patch);
      new_dw->allocateAndPut(int_eng_L_ME[m],Ilb->eng_L_ME_CCLabel,indx,patch);
    }

    // Convert momenta to velocities and internal energy to Temp
    for(CellIterator iter = patch->getExtraCellIterator(); !iter.done();iter++){
      IntVector c = *iter;
      for (int m = 0; m < numALLMatls; m++) {
        Temp_CC[m][c] = int_eng_L[m][c]/(mass_L[m][c]*cv[m][c]);
        vel_CC[m][c]  = mom_L[m][c]/mass_L[m][c];
      }
    }

    //__________________________________
    //
    for(CellIterator iter = patch->getCellIterator(); !iter.done();iter++){
      IntVector c = *iter;

      //---------- M O M E N T U M   E X C H A N G E
      //   Form BETA matrix (a), off diagonal terms
      for(int m = 0; m < numALLMatls; m++)  {
        tmp = delT*sp_vol_CC[m][c];
        for(int n = 0; n < numALLMatls; n++) {
          beta(m,n) = vol_frac_CC[n][c]  * K(n,m) * tmp;
          a(m,n) = -beta(m,n);
        }
      }
      //   Form matrix (a) diagonal terms
      for(int m = 0; m < numALLMatls; m++) {
        a(m,m) = 1.0;
        for(int n = 0; n < numALLMatls; n++) {
          a(m,m) +=  beta(m,n);
        }
      }

      for(int m = 0; m < numALLMatls; m++) {
        Vector sum(0,0,0);
        const Vector& vel_m = vel_CC[m][c];
        for(int n = 0; n < numALLMatls; n++) {
          sum += beta(m,n) *(vel_CC[n][c] - vel_m);
        }
        bb[m] = sum;
      }

      a.destructiveSolve(bb);

      for(int m = 0; m < numALLMatls; m++) {
        vel_CC[m][c] += bb[m];
      }

      //---------- E N E R G Y   E X C H A N G E
      if(d_exchCoeff->d_heatExchCoeffModel != "constant"){
        d_exchCoeff->getVariableExchangeCoeff( K, H, c, mass_L);
      }

      for(int m = 0; m < numALLMatls; m++) {
        tmp = delT*sp_vol_CC[m][c] / cv[m][c];
        for(int n = 0; n < numALLMatls; n++)  {
          beta(m,n) = vol_frac_CC[n][c] * H(n,m)*tmp;
          a(m,n) = -beta(m,n);
        }
      }

      //   Form matrix (a) diagonal terms
      for(int m = 0; m < numALLMatls; m++) {
        a(m,m) = 1.;
        for(int n = 0; n < numALLMatls; n++)   {
          a(m,m) +=  beta(m,n);
        }
      }
      // -  F O R M   R H S   (b)
      for(int m = 0; m < numALLMatls; m++)  {
        b[m] = 0.0;

       for(int n = 0; n < numALLMatls; n++) {
         b[m] += beta(m,n) * (Temp_CC[n][c] - Temp_CC[m][c]);
        }
      }

      //     S O L V E, Add exchange contribution to orig value
      a.destructiveSolve(b);

      for(int m = 0; m < numALLMatls; m++) {
        Temp_CC[m][c] = Temp_CC[m][c] + b[m];
      }
    } // end CellIterator loop

    //__________________________________
    //   C O N V E C T I V E   H E A T   T R A N S F E R
    if(d_exchCoeff->convective()){
      //  Loop over matls
      //    if (mpm_matl)
      //      Loop over cells
      //        find surface and surface normals
      //        choose adjacent cell
      //        find mass weighted average temp in adjacent cell (T_ave)
      //        compute a heat transfer to the container h(T-T_ave)
      //        compute Temp_CC = Temp_CC + h_trans/(mass*cv)
      //      end loop over cells
      //    endif (mpm_matl)
      //  endloop over matls
      FastMatrix cet(2,2), ac(2,2);
      double RHSc[2];
      cet.zero();
      int gm = d_exchCoeff->conv_fluid_matlindex();  // gas matl from which to get heat
      int sm = d_exchCoeff->conv_solid_matlindex();  // solid matl that heat goes to

      Ghost::GhostType  gn = Ghost::None;
      constNCVariable< int > isSurfaceCell;
      new_dw->get( isSurfaceCell, d_isSurfaceCellLabel, 0, patch, gn,0 );
     
      Vector dx    = patch->dCell();
      double dxLen = dx.length();
        
      const Level* level=patch->getLevel();

      for (int m = 0; m < numALLMatls; m++)  {
        
        Material* matl = d_matlManager->getMaterial( m );
        MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);
        int dwindex = matl->getDWIndex();
        
        if(mpm_matl && dwindex==sm){
        
          constCCVariable<Vector> surfaceNorm;
          new_dw->get(surfaceNorm, d_surfaceNormLabel, dwindex,patch, gn, 0);

          for(CellIterator iter = patch->getCellIterator(); !iter.done();iter++){
            IntVector c = *iter;
            
            // surface
            if ( isSurfaceCell[c]){

              Point this_cell_pos = level->getCellPosition(c);
              Point adja_cell_pos = this_cell_pos + .6 * dxLen * surfaceNorm[c];

              IntVector q;
              if(patch->findCell(adja_cell_pos, q)){
              
                cet(0,0) = 0.;
                cet(0,1) = delT * vol_frac_CC[gm][q] * H(sm,gm) * sp_vol_CC[sm][c]/cv[sm][c];
                cet(1,0) = delT * vol_frac_CC[sm][c] * H(gm,sm) * sp_vol_CC[gm][q]/cv[gm][q];
                cet(1,1) = 0.;

                ac(0,1) = -cet(0,1);
                ac(1,0) = -cet(1,0);

                //   Form matrix (a) diagonal terms
                for(int m = 0; m < 2; m++) {
                  ac(m,m) = 1.;
                  for(int n = 0; n < 2; n++)   {
                    ac(m,m) +=  cet(m,n);
                  }
                }

                RHSc[0] = cet(0,1)*(Temp_CC[gm][q] - Temp_CC[sm][c]);
                RHSc[1] = cet(1,0)*(Temp_CC[sm][c] - Temp_CC[gm][q]);
                
                ac.destructiveSolve(RHSc);
                
                Temp_CC[sm][c] += RHSc[0];
                Temp_CC[gm][q] += RHSc[1];
              }
            }  // if a surface cell
          }  // cellIterator
        } // if mpm_matl
      } // for ALL matls
    } // convective heat transfer

    //__________________________________
    //
    if( BC_globalVars->usingLodi || BC_globalVars->usingMicroSlipBCs){

      std::vector<CCVariable<double> > temp_CC_Xchange(numALLMatls);
      std::vector<CCVariable<Vector> > vel_CC_Xchange(numALLMatls);

      for (int m = 0; m < numALLMatls; m++) {
        Material* matl = d_matlManager->getMaterial(m);
        int indx = matl->getDWIndex();

        new_dw->allocateAndPut(temp_CC_Xchange[m],Ilb->temp_CC_XchangeLabel,indx,patch);
        new_dw->allocateAndPut(vel_CC_Xchange[m], Ilb->vel_CC_XchangeLabel, indx,patch);
        vel_CC_Xchange[m].copy(vel_CC[m]);
        temp_CC_Xchange[m].copy(Temp_CC[m]);
      }
    }

    //__________________________________
    //  Set boundary conditions
    for (int m = 0; m < numALLMatls; m++)  {
      Material* matl = d_matlManager->getMaterial( m );
      int indx = matl->getDWIndex();

      customBC_localVars* BC_localVars   = scinew customBC_localVars();
      preprocess_CustomBCs("CC_Exchange", old_dw, new_dw, Ilb, patch, indx, BC_globalVars, BC_localVars);

      setBC(vel_CC[m], "Velocity",   patch, d_matlManager, indx, new_dw,
                                                        BC_globalVars, BC_localVars, isNotInitialTimeStep);
      setBC(Temp_CC[m],"Temperature",gamma[m], cv[m], patch, d_matlManager,
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
      IntVector c = *iter;
      for (int m = 0; m < numALLMatls; m++) {
        int_eng_L_ME[m][c] = Temp_CC[m][c]*cv[m][c] * mass_L[m][c];
        mom_L_ME[m][c]     = vel_CC[m][c]           * mass_L[m][c];
        Tdot[m][c]         = (Temp_CC[m][c] - old_temp[m][c])/delT;
      }
    }
  } //patches
}

//______________________________________________________________________
//
void ScalarExch::addExch_Vel_Temp_CC_1matl( const ProcessorGroup * pg,
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

    printTask(patches, patch, dbgExch, "Doing ScalarExch::addExch_Vel_Temp_CC_1matl" );

    delt_vartype delT;
    old_dw->get(delT, Ilb->delTLabel, level);

    CCVariable<double>  Temp_CC;
    constCCVariable<double>  gamma;
    constCCVariable<Vector>  mom_L;
    constCCVariable<double>  int_eng_L;
    constCCVariable<double>  cv;

    // Create variables for the results
    CCVariable<Vector>  mom_L_ME;
    CCVariable<Vector>  vel_CC;
    CCVariable<double>  int_eng_L_ME;
    CCVariable<double>  Tdot;
    constCCVariable<double>  mass_L;
    constCCVariable<double>  old_temp;

    Ghost::GhostType  gn = Ghost::None;
    ICEMaterial* ice_matl = (ICEMaterial*) d_matlManager->getMaterial( "ICE", 0);
    int indx = ice_matl->getDWIndex();

    old_dw->get(old_temp,  Ilb->temp_CCLabel,      indx, patch, gn, 0);
    new_dw->get(cv,        Ilb->specific_heatLabel,indx, patch, gn, 0);
    new_dw->get(gamma,     Ilb->gammaLabel,        indx, patch, gn, 0);
    new_dw->get(mass_L,    Ilb->mass_L_CCLabel,    indx, patch, gn, 0);
    new_dw->get(mom_L,     Ilb->mom_L_CCLabel,     indx, patch, gn, 0);
    new_dw->get(int_eng_L, Ilb->int_eng_L_CCLabel, indx, patch, gn, 0);

    new_dw->allocateAndPut(Tdot,        Ilb->Tdot_CCLabel,    indx, patch);
    new_dw->allocateAndPut(mom_L_ME,    Ilb->mom_L_ME_CCLabel,indx, patch);
    new_dw->allocateAndPut(int_eng_L_ME,Ilb->eng_L_ME_CCLabel,indx, patch);

    new_dw->allocateTemporary(vel_CC,  patch);
    new_dw->allocateTemporary(Temp_CC, patch);

    //__________________________________
    // Convert momenta to velocities and internal energy to Temp
    for(CellIterator iter = patch->getExtraCellIterator(); !iter.done();iter++){
      IntVector c = *iter;
      Temp_CC[c] = int_eng_L[c]/(mass_L[c]*cv[c]);
      vel_CC[c]  = mom_L[c]/mass_L[c];
    }

    //__________________________________
    //  Apply boundary conditions
    if(BC_globalVars->usingLodi ||
       BC_globalVars->usingMicroSlipBCs){
      CCVariable<double> temp_CC_Xchange;
      CCVariable<Vector> vel_CC_Xchange;

      new_dw->allocateAndPut( temp_CC_Xchange,Ilb->temp_CC_XchangeLabel,indx,patch );
      new_dw->allocateAndPut( vel_CC_Xchange, Ilb->vel_CC_XchangeLabel, indx,patch );
      vel_CC_Xchange.copy(  vel_CC  );
      temp_CC_Xchange.copy( Temp_CC );
    }

    customBC_localVars* BC_localVars = scinew customBC_localVars();
    preprocess_CustomBCs("CC_Exchange",old_dw, new_dw, Ilb, patch, indx,
                          BC_globalVars, BC_localVars );

    setBC(vel_CC, "Velocity",   patch, d_matlManager,
                                       indx, new_dw,  BC_globalVars, BC_localVars, isNotInitialTimeStep);
    setBC(Temp_CC,"Temperature",gamma, cv, patch, d_matlManager,
                                       indx, new_dw,  BC_globalVars, BC_localVars, isNotInitialTimeStep );
#if SET_CFI_BC
//      set_CFI_BC<Vector>(vel_CC[m],  patch);
//      set_CFI_BC<double>(Temp_CC[m], patch);
#endif
    delete_CustomBCs( BC_globalVars, BC_localVars );

    //__________________________________
    // Convert vars. primitive-> flux
    for(CellIterator iter = patch->getExtraCellIterator(); !iter.done();iter++){
      IntVector c = *iter;
      int_eng_L_ME[c] = Temp_CC[c]*cv[c] * mass_L[c];
      mom_L_ME[c]     = vel_CC[c]        * mass_L[c];
      Tdot[c]         = (Temp_CC[c] - old_temp[c])/delT;
    }
  } //patches
}
