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

#include <CCA/Components/Models/MultiMatlExchange/Slip.h>
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
SlipExch::SlipExch(const ProblemSpecP     & exch_ps,
                   const MaterialManagerP & materialManager,
                   const bool with_mpm )
  : ExchangeModel( exch_ps, materialManager, with_mpm )
{
  proc0cout << "__________________________________\n";
  proc0cout << " Now creating the Slip Exchange model " << endl;

  d_exchCoeff = scinew ExchangeCoefficients();
  d_vel_CCTransLabel  = VarLabel::create("vel_CCTransposed", CCVariable<Vector>::getTypeDescription());
  d_meanFreePathLabel = VarLabel::create("meanFreePath",     CCVariable<double>::getTypeDescription());
}

//______________________________________________________________________
//
SlipExch::~SlipExch()
{
  delete d_exchCoeff;
  VarLabel::destroy( d_vel_CCTransLabel );
  VarLabel::destroy( d_meanFreePathLabel );
}

//______________________________________________________________________
//
void SlipExch::problemSetup(const ProblemSpecP & matl_ps)
{

  // read in the exchange coefficients
  ProblemSpecP exch_ps;
  d_exchCoeff->problemSetup( matl_ps, d_numMatls, exch_ps );

  ProblemSpecP model_ps = exch_ps->findBlock( "Model" );
  model_ps->require("fluidMatlIndex",               d_fluidMatlIndx);
  model_ps->require("solidMatlIndex",               d_solidMatlIndx);
  model_ps->require("momentum_accommodation_coeff", d_momentum_accommodation_coeff);
  model_ps->require("thermal_accommodation_coeff",  d_thermal_accommodation_coeff);
  model_ps->get( "useSlipCoeffs", d_useSlipCoeffs );


  proc0cout << " fluidMatlIndex: " << d_fluidMatlIndx << " thermal_accommodation_coeff " << d_thermal_accommodation_coeff << endl;
//  d_exchCoeff->problemSetup(mat_ps, d_matlManager);
}

//______________________________________________________________________
//
void SlipExch::outputProblemSpec(ProblemSpecP & matl_ps )
{
  ProblemSpecP exch_prop_ps;
  d_exchCoeff->outputProblemSpec(matl_ps, exch_prop_ps);

  // <Model type="slip">
  ProblemSpecP model_ps = exch_prop_ps->appendChild("Model");
  model_ps->setAttribute("type","slip");

  model_ps->appendElement( "fluidMatlIndex", d_fluidMatlIndx );
  model_ps->appendElement( "solidMatlIndex", d_solidMatlIndx );
  model_ps->appendElement( "thermal_accommodation_coeff",  d_thermal_accommodation_coeff);
  model_ps->appendElement( "momentum_accommodation_coeff", d_momentum_accommodation_coeff);
}

//______________________________________________________________________
//  These tasks are called before semi-implicit pressure solve.
//  All computed variables live in the parent NewDW
void SlipExch::sched_PreExchangeTasks(SchedulerP           & sched,
                                      const PatchSet       * patches,
                                      const MaterialSubset * ice_matls,
                                      const MaterialSubset * mpm_matls,
                                      const MaterialSet    * allMatls)
{
  //__________________________________
  // compute surface normal and isSurfaceCell
  schedComputeSurfaceNormal( sched, patches, mpm_matls );

  //__________________________________
  //  compute Mean Free Path
  schedComputeMeanFreePath( sched, patches );
}

//______________________________________________________________________
//  This method requires variables from inside the semi-implicit pressure
// solve sub-scheduler.  Put variables that are required from the
// Parent OldDW and NewDW
void SlipExch::addExchangeModelRequires ( Task* t,
                                          const MaterialSubset * zeroMatl,
                                          const MaterialSubset * ice_matls,
                                          const MaterialSubset * mpm_matls)
{
  Ghost::GhostType  gac = Ghost::AroundCells;
  t->requires( Task::NewDW, d_meanFreePathLabel,   ice_matls, gac, 1 );
  t->requires( Task::NewDW, d_isSurfaceCellLabel,  zeroMatl,  gac, 1 );
  t->requires( Task::NewDW, d_surfaceNormLabel,    mpm_matls, gac, 1 );
}


//______________________________________________________________________
//
void SlipExch::sched_AddExch_VelFC(SchedulerP           & sched,
                                   const PatchSet       * patches,
                                   const MaterialSubset * ice_matls,
                                   const MaterialSubset * mpm_matls,
                                   const MaterialSet    * all_matls,
                                   customBC_globalVars  * BC_globalVars,
                                   const bool recursion)
{
  //__________________________________
  //
  Task* t = scinew Task( "SlipExch::addExch_VelFC", this, &SlipExch::addExch_VelFC,
                         BC_globalVars, recursion);

  printSchedule( patches, dbgExch, "SlipExch::sched_AddExch_VelFC" );

  if(recursion) {
    t->requires(Task::ParentOldDW, Ilb->delTLabel,getLevel(patches));
  } else {
    t->requires(Task::OldDW,       Ilb->delTLabel,getLevel(patches));
  }

  Ghost::GhostType  gac = Ghost::AroundCells;
  Ghost::GhostType  gaf_X = Ghost::AroundFacesX;
  Ghost::GhostType  gaf_Y = Ghost::AroundFacesY;
  Ghost::GhostType  gaf_Z = Ghost::AroundFacesZ;

  //__________________________________
  // define parent data warehouse
  // change the definition of parent(old/new)DW
  // when using semi-implicit pressure solve
  Task::WhichDW pNewDW = Task::NewDW;
  Task::WhichDW pOldDW = Task::OldDW;
  if(recursion) {
    pNewDW  = Task::ParentNewDW;
    pOldDW  = Task::ParentOldDW;
  }

  // All matls
  t->requires( pNewDW,      Ilb->rho_CCLabel,     gac,   1);
  t->requires( pNewDW,      Ilb->sp_vol_CCLabel,  gac,   1);
  t->requires( pNewDW,      Ilb->vol_frac_CCLabel,gac,   1);
  t->requires( Task::NewDW, Ilb->uvel_FCLabel,    gaf_X, 1);
  t->requires( Task::NewDW, Ilb->vvel_FCLabel,    gaf_Y, 1);
  t->requires( Task::NewDW, Ilb->wvel_FCLabel,    gaf_Z, 1);

  t->requires( pNewDW,      d_meanFreePathLabel,   ice_matls,  gac, 1 );
  t->requires( pNewDW,      d_surfaceNormLabel,    mpm_matls,  gac, 1 );
  t->requires( pNewDW,      d_isSurfaceCellLabel,  d_zero_matl,gac, 1 );
  t->requires( pOldDW,      Ilb->vel_CCLabel,      ice_matls,  gac, 1 );
  t->requires( pNewDW,      Ilb->vel_CCLabel,      mpm_matls,  gac, 1 );

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
void SlipExch::vel_FC_exchange( CellIterator      iter,
                                const Patch     * patch,
                                const IntVector   adj_offset,
                                const double      delT,
                                constCCVariable<int> & isSurfaceCell,
                                std::vector<constCCVariable<Vector> >& surfaceNorm,
                                std::vector<constCCVariable<double> >& meanFreePath,
                                std::vector<constCCVariable<double> >& vol_frac_CC,
                                std::vector<constCCVariable<double> >& sp_vol_CC,
                                std::vector< constSFC>               & vel_FC,
                                std::vector< SFC >                   & sp_vol_FC,
                                std::vector< SFC >                   & vel_FCME)

{
  double b[MAX_MATLS];
  FastMatrix Q(3,3);
  double b_sp_vol[MAX_MATLS];
  double vel[MAX_MATLS];
  double tmp[MAX_MATLS];
  FastMatrix a(d_numMatls, d_numMatls);

  int gm = d_fluidMatlIndx;
  int sm = d_solidMatlIndx;

  Vector dx = patch->dCell();

  //__________________________________
  //
  for(;!iter.done(); iter++){
    IntVector c   = *iter;
    IntVector adj = c + adj_offset;

    double K_R = 1e15;
    double K_L = 1e15;

   //  Q.identity();         // Transformation matrix is initialized

/***** Compute K value at current cell [c] (R) *****/

    if( isSurfaceCell[c] && d_useSlipCoeffs ){

//__________________________________
//  <start>                This should be put in a function, since it's a duplicate of the code below except for the cell index  --Todd
      Q.identity();

      computeSurfaceRotationMatrix(Q, surfaceNorm[sm][c]); // Makes Q at each cell c

      double A_V = 1.0/( dx.x() * fabs(Q(1,0))+
                         dx.y() * fabs(Q(1,1))+
                         dx.z() * fabs(Q(1,2)) );

      double av     = d_momentum_accommodation_coeff;
      double Beta_v = (2 - av)/av;

      K_R = A_V / (Beta_v * meanFreePath[gm][c] * vol_frac_CC[sm][c]);

      if(K_R > 1e15) {
        K_R = 1e15;                          // K > Kslip in reality, so if Kslip > K in computation, fix this.
      }

//  < end>
//__________________________________
    }  // if a surface cell


/***** Compute K value at adjacent cell [adj] (L) *****/

    if( isSurfaceCell[adj] && d_useSlipCoeffs ){

      Q.identity();

      // This should work for more that one solid.  It's hard wired for 2 matls-- Todd

      computeSurfaceRotationMatrix(Q, surfaceNorm[sm][adj]); // Makes Q at each cell c

      double A_V = 1.0/( dx.x() * fabs(Q(1,0))+
                         dx.y() * fabs(Q(1,1))+
                         dx.z() * fabs(Q(1,2)) );

      double av     = d_momentum_accommodation_coeff;
      double Beta_v = (2 - av)/av;

      K_L = A_V / (Beta_v * meanFreePath[gm][adj] * vol_frac_CC[sm][adj]);

      if(K_L > 1e15) {
        K_L = 1e15;                      // K > Kslip in reality, so if Kslip > K in computation, fix this.
      }

    }  // if a surface cell


    //__________________________________
    //   Compute beta and off diagonal term of
    //   Matrix A, this includes b[m][m].
    //   You need to make sure that mom_exch_coeff[m][m] = 0

    // - Form diagonal terms of Matrix (A)
    //  - Form RHS (b)
    for(int m = 0; m < d_numMatls; m++)  {

      b_sp_vol[m] = 2.0 * (sp_vol_CC[m][adj] * sp_vol_CC[m][c])/
                          (sp_vol_CC[m][adj] + sp_vol_CC[m][c]);

      tmp[m] = -0.5 * delT * (vol_frac_CC[m][adj]*K_L + vol_frac_CC[m][c]*K_R);
      vel[m] = vel_FC[m][c];
    }

    for(int m = 0; m < d_numMatls; m++)  {
      double betasum = 1;
      double bsum    = 0;
      double bm      = b_sp_vol[m];
      double vm      = vel[m];

      for(int n = 0; n < d_numMatls; n++)  {
         if ( n!=m ) {
           double b = bm * tmp[n];
           a(m,n)   = b;
           betasum -= b;
           bsum    -= b * (vel[n] - vm);
          }
        else{
        double b = 0;
        a(m,n)   = b;
        betasum -= b;
        bsum    -= b * (vel[n] - vm);
        }
      }
      a(m,m) = betasum;
      b[m] = bsum;
    }

    //__________________________________
    //  - solve and backout velocities

    a.destructiveSolve(b, b_sp_vol);
    //  For implicit solve we need sp_vol_FC
    for(int m = 0; m < d_numMatls; m++) {
      vel_FCME[m][c]  = vel_FC[m][c] + b[m];
      sp_vol_FC[m][c] = b_sp_vol[m];        // only needed by implicit Pressure

    }
  }  // iterator
}

//______________________________________________________________________
//
void SlipExch::addExch_VelFC(const ProcessorGroup  * pg,
                             const PatchSubset     * patches,
                             const MaterialSubset  * matls,
                             DataWarehouse         * old_dw,
                             DataWarehouse         * new_dw,
                             customBC_globalVars   * BC_globalVars,
                             const bool recursion)
{
  const Level* level = getLevel(patches);
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);

    printTask(patches, patch, dbgExch, "Doing SlipExch::addExch_VelFC" );

    // change the definition of parent(old/new)DW
    // if using semi-implicit pressure solve
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

    constCCVariable<int> isSurfaceCell;

    std::vector< constCCVariable<double> > sp_vol_CC(    d_numMatls  );
    std::vector< constCCVariable<double> > mass_L(       d_numMatls  );
    std::vector< constCCVariable<double> > vol_frac_CC(  d_numMatls  );

    std::vector< constCCVariable<double> > meanFreePath( d_numMatls );
    std::vector< constCCVariable<Vector> > vel_CC(       d_numMatls );
    std::vector< constCCVariable<Vector> > surfaceNorm(  d_numMatls );

    std::vector< constSFCXVariable<double> > uvel_FC( d_numMatls );
    std::vector< constSFCYVariable<double> > vvel_FC( d_numMatls );
    std::vector< constSFCZVariable<double> > wvel_FC( d_numMatls );

    std::vector< SFCXVariable<double> >uvel_FCME( d_numMatls ), sp_vol_XFC( d_numMatls );
    std::vector< SFCYVariable<double> >vvel_FCME( d_numMatls ), sp_vol_YFC( d_numMatls );
    std::vector< SFCZVariable<double> >wvel_FCME( d_numMatls ), sp_vol_ZFC( d_numMatls );

    Ghost::GhostType  gac = Ghost::AroundCells;
    Ghost::GhostType  gaf_X = Ghost::AroundFacesX;
    Ghost::GhostType  gaf_Y = Ghost::AroundFacesY;
    Ghost::GhostType  gaf_Z = Ghost::AroundFacesZ;
    pNewDW->get( isSurfaceCell, d_isSurfaceCellLabel, 0, patch, gac, 1);

    //__________________________________
    //  Multimaterial arrays
    for(int m = 0; m < d_numMatls; m++) {
      Material* matl = d_matlManager->getMaterial( m );

      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);

      int indx = matl->getDWIndex();

      // retreive from dw
      pNewDW->get( sp_vol_CC[m],   Ilb->sp_vol_CCLabel,  indx, patch,gac, 1);
      pNewDW->get( vol_frac_CC[m], Ilb->vol_frac_CCLabel,indx, patch,gac, 1);

      new_dw->get( uvel_FC[m],     Ilb->uvel_FCLabel,    indx, patch,gaf_X, 1);
      new_dw->get( vvel_FC[m],     Ilb->vvel_FCLabel,    indx, patch,gaf_Y, 1);
      new_dw->get( wvel_FC[m],     Ilb->wvel_FCLabel,    indx, patch,gaf_Z, 1);

      if(mpm_matl) {
        pNewDW->get( vel_CC[m],         Ilb->vel_CCLabel, indx, patch,gac, 1);
        pNewDW->get( surfaceNorm[m], d_surfaceNormLabel,  indx, patch,gac, 1);
      }

      if(ice_matl) {
        pOldDW->get( vel_CC[m],          Ilb->vel_CCLabel, indx, patch,gac, 1);
        pNewDW->get( meanFreePath[m], d_meanFreePathLabel, indx, patch,gac, 1);
      }

      new_dw->allocateAndPut( uvel_FCME[m], Ilb->uvel_FCMELabel, indx, patch );
      new_dw->allocateAndPut( vvel_FCME[m], Ilb->vvel_FCMELabel, indx, patch );
      new_dw->allocateAndPut( wvel_FCME[m], Ilb->wvel_FCMELabel, indx, patch );

      new_dw->allocateAndPut( sp_vol_XFC[m],Ilb->sp_volX_FCLabel,indx, patch );
      new_dw->allocateAndPut( sp_vol_YFC[m],Ilb->sp_volY_FCLabel,indx, patch );
      new_dw->allocateAndPut( sp_vol_ZFC[m],Ilb->sp_volZ_FCLabel,indx, patch );

      // lowIndex is the same for all face centered vars
      IntVector lowIndex(patch->getExtraSFCXLowIndex());
      uvel_FCME[m].initialize(0.0,  lowIndex,patch->getExtraSFCXHighIndex() );
      vvel_FCME[m].initialize(0.0,  lowIndex,patch->getExtraSFCYHighIndex() );
      wvel_FCME[m].initialize(0.0,  lowIndex,patch->getExtraSFCZHighIndex() );

      sp_vol_XFC[m].initialize(0.0, lowIndex,patch->getExtraSFCXHighIndex() );
      sp_vol_YFC[m].initialize(0.0, lowIndex,patch->getExtraSFCYHighIndex() );
      sp_vol_ZFC[m].initialize(0.0, lowIndex,patch->getExtraSFCZHighIndex() );
    }

    vector<IntVector> adj_offset(3);
    adj_offset[0] = IntVector(-1, 0, 0);    // X faces
    adj_offset[1] = IntVector(0, -1, 0);    // Y faces
    adj_offset[2] = IntVector(0,  0, -1);   // Z faces

    CellIterator XFC_iterator = patch->getSFCXIterator();
    CellIterator YFC_iterator = patch->getSFCYIterator();
    CellIterator ZFC_iterator = patch->getSFCZIterator();

    //__________________________________
    //  tack on exchange contribution to FC velocities
    vel_FC_exchange<constSFCXVariable<double>, SFCXVariable<double> >
                    (XFC_iterator, patch, adj_offset[0], delT, isSurfaceCell,
                    surfaceNorm, meanFreePath, vol_frac_CC, sp_vol_CC, uvel_FC, sp_vol_XFC, uvel_FCME);

    vel_FC_exchange<constSFCYVariable<double>, SFCYVariable<double> >
                    (YFC_iterator, patch, adj_offset[1], delT, isSurfaceCell,
                    surfaceNorm, meanFreePath, vol_frac_CC, sp_vol_CC, vvel_FC, sp_vol_YFC, vvel_FCME);

    vel_FC_exchange<constSFCZVariable<double>, SFCZVariable<double> >
                    (ZFC_iterator, patch, adj_offset[2], delT, isSurfaceCell,
                    surfaceNorm, meanFreePath, vol_frac_CC, sp_vol_CC, wvel_FC, sp_vol_ZFC, wvel_FCME);

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
//  This method computes the cell centered exchange
void SlipExch::vel_CC_exchange( CellIterator  iter,
                                const Patch  * patch,
                                FastMatrix   & k_org,
                                const double   delT,
                                constCCVariable<int> & isSurfaceCell,
                                std::vector< constCCVariable<Vector> > & surfaceNorm,
                                std::vector< constCCVariable<double> > & vol_frac_CC,
                                std::vector< constCCVariable<double> > & sp_vol_CC,
                                std::vector< constCCVariable<double> > & meanFreePath,
                                std::vector< constCCVariable<Vector> > & vel_CC,
                                std::vector< CCVariable<Vector> >      & vel_T_CC,
                                std::vector< CCVariable<Vector> >      & delta_vel_exch)
{
  double b[MAX_MATLS];
  Vector bb[MAX_MATLS];
  double tmp;
  FastMatrix beta(  d_numMatls, d_numMatls );
  FastMatrix Q(3,3);
  FastMatrix K(     d_numMatls, d_numMatls );
  FastMatrix Kslip( d_numMatls, d_numMatls );
  FastMatrix junk(  d_numMatls, d_numMatls );
  FastMatrix a(     d_numMatls, d_numMatls );
  Vector vel_T[MAX_MATLS];                    // Transposed velocity
  Vector vel_T_dbg[MAX_MATLS];                // Transposed velocity for visulaization

  // for readability
  int gm = d_fluidMatlIndx;
  int sm = d_solidMatlIndx;

  Vector dx = patch->dCell();

  for(;!iter.done(); iter++){
    IntVector c = *iter;

    Q.identity();         // Transformation matrix is initialized
    Kslip.copy(k_org);
    K.copy(k_org);

    //__________________________________
    //  If cell is a surface cell modify exchange coefficients
    if( isSurfaceCell[c] && d_useSlipCoeffs ){

      //************************This looks almost identical to the code in vel_FC_exchange.  Should consider putting in a function --Todd.

      computeSurfaceRotationMatrix(Q, surfaceNorm[sm][c]); // Makes Q at each cell c

      double A_V = 1.0/( dx.x()*fabs(Q(1,0)) +
                         dx.y()*fabs(Q(1,1)) +
                         dx.z()*fabs(Q(1,2)) );

      double av     = d_momentum_accommodation_coeff;
      double Beta_v = (2 - av)/av;

      Kslip(gm,sm) = A_V / (Beta_v * meanFreePath[gm][c] * vol_frac_CC[sm][c]);

      if(Kslip(gm,sm) > k_org(gm,sm)) {
        Kslip(gm,sm) = k_org(gm,sm);        // K > Kslip in reality, so if Kslip > K in computation, fix this.
      }

      Kslip(sm,gm) = Kslip(gm,sm);         // Make the inverse indices of the Kslip matrix equal to each other


      //__________________________________
      //
      for(int i = 0; i < 3; i++) {

        if(i != 1) {         // if the direction is NOT the normal to the surface
           K.copy(Kslip);
        } else {
           K.copy(k_org);
        }

        //__________________________________
        //  coordinate Transformation
        for(int m = 0; m < d_numMatls; m++) {
          vel_T[m][i]     = 0;
          vel_T_dbg[m][i] = 0;

          for(int j = 0; j < 3; j++) {
            vel_T[m][i] += Q(i,j) * vel_CC[m][c][j];
          }
          vel_T_dbg[m][i] = vel_T[m][i];
        }

        //__________________________________
        //  compute exchange using transposed velocity

        a.zero();

        for(int m = 0; m < d_numMatls; m++) {
          double adiag = 1.0;
          b[m] = 0.0;

          for(int n = 0; n < d_numMatls; n++) {
            a(m,n) = - delT * vol_frac_CC[n][c] * sp_vol_CC[m][c] * K(n,m);
            adiag -= a(m,n);
            b[m]  -= a(m,n) * (vel_T[n][i] - vel_T[m][i]);
          }
          a(m,m) = adiag;
        }

        a.destructiveSolve(b);

        for(int m = 0; m < d_numMatls; m++) {
          vel_T[m][i] = b[m];
        }
      } // loop over directions

    //__________________________________
    //  coordinate transformation
    for(int m = 0; m < d_numMatls; m++) {
      vel_T_CC[m][c] = vel_T_dbg[m];               // for visualization

      Vector vel_exch( Vector(0.0) );

      for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 3; j++) {
          vel_exch[i] += Q(j,i) * vel_T[m][j]; // Use the transpose of Q to back out the velocity in the cartesian system
        }
      }
      delta_vel_exch[m][c] = vel_exch;

     }
   }  // if a surface cell
    else{

      a.zero();
      beta.zero();

      for(int m = 0; m < d_numMatls; m++)  {
        tmp = delT*sp_vol_CC[m][c];
        for(int n = 0; n < d_numMatls; n++) {
          beta(m,n) = vol_frac_CC[n][c]  * K(n,m) * tmp;
          a(m,n) = -beta(m,n);
        }
      }
      //   Form matrix (a) diagonal terms
      for(int m = 0; m < d_numMatls; m++) {
        a(m,m) = 1.0;
        for(int n = 0; n < d_numMatls; n++) {
          a(m,m) +=  beta(m,n);
        }
      }

      for(int m = 0; m < d_numMatls; m++) {
        Vector sum(0,0,0);
        const Vector& vel_m = vel_CC[m][c];

        for(int n = 0; n < d_numMatls; n++) {
          sum += beta(m,n) *(vel_CC[n][c] - vel_m);
        }
        bb[m] = sum;
      }

      a.destructiveSolve(bb);

      //__________________________________
      //  save exchange contribution of each material
      for(int m = 0; m < d_numMatls; m++) {
        delta_vel_exch[m][c] = bb[m];
      }
    }
  }
}

//______________________________________________________________________
//
void SlipExch::sched_AddExch_Vel_Temp_CC(SchedulerP           & sched,
                                         const PatchSet       * patches,
                                         const MaterialSubset * ice_matls,
                                         const MaterialSubset * mpm_matls,
                                         const MaterialSet    * all_matls,
                                         customBC_globalVars  * BC_globalVars)
{


  //__________________________________
  //
  string name = "SlipExch::addExch_Vel_Temp_CC";
  Task* t = scinew Task(name, this, &SlipExch::addExch_Vel_Temp_CC, BC_globalVars);

  printSchedule( patches, dbgExch, name );

  Ghost::GhostType  gn  = Ghost::None;

  t->requires( Task::OldDW,  Ilb->delTLabel,getLevel(patches));
  t->requires( Task::NewDW,  d_surfaceNormLabel,    mpm_matls,   gn, 0 );
  t->requires( Task::NewDW,  d_isSurfaceCellLabel,  d_zero_matl, gn, 0 );
                                // I C E
  t->requires( Task::OldDW,  Ilb->temp_CCLabel,       ice_matls, gn );
  t->requires( Task::NewDW,  Ilb->specific_heatLabel, ice_matls, gn );
  t->requires( Task::NewDW,  Ilb->gammaLabel,         ice_matls, gn );
  t->requires( Task::NewDW,  d_meanFreePathLabel,     ice_matls, gn );
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
  t->computes( d_vel_CCTransLabel );

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

    printTask(patches, patch, dbgExch,"Doing SlipExch::addExch_Vel_Temp_CC");

    //__________________________________
    // Declare variables
    constCCVariable<int> isSurfaceCell;

    std::vector< CCVariable<double> >      cv(          d_numMatls );
    std::vector< CCVariable<double> >      Temp_CC(     d_numMatls );
    std::vector< constCCVariable<double> > gamma(       d_numMatls );
    std::vector< constCCVariable<double> > vol_frac_CC( d_numMatls );
    std::vector< constCCVariable<double> > sp_vol_CC(   d_numMatls );
    std::vector< constCCVariable<Vector> > mom_L(       d_numMatls );
    std::vector< constCCVariable<double> > int_eng_L(   d_numMatls );
    std::vector< constCCVariable<double> > mass_L(      d_numMatls );
    std::vector< constCCVariable<double> > old_temp(    d_numMatls );

    std::vector< constCCVariable<double> > meanFreePath( d_numMatls );  // This mean free path does not have viscosity in it, which is okay per how it is used in this code
    std::vector< constCCVariable<Vector> > surfaceNorm(  d_numMatls );
    std::vector< constCCVariable<Vector> > const_vel_CC( d_numMatls );

    // Create variables for the results
    std::vector< CCVariable<Vector> > mom_L_ME(      d_numMatls );
    std::vector< CCVariable<Vector> > vel_CC(        d_numMatls );
    std::vector< CCVariable<double> > int_eng_L_ME(  d_numMatls );
    std::vector< CCVariable<double> > Tdot(          d_numMatls );
    std::vector< CCVariable<Vector> > vel_T_CC(      d_numMatls );
    std::vector< CCVariable<Vector> > delta_vel_exch(d_numMatls );


    //__________________________________
    //  retreive data from the data warehouse
    delt_vartype delT;
    old_dw->get(delT, Ilb->delTLabel, level);

    Ghost::GhostType  gn = Ghost::None;
    new_dw->get( isSurfaceCell, d_isSurfaceCellLabel, 0, patch, gn, 0);

    for (int m = 0; m < d_numMatls; m++) {
      Material* matl = d_matlManager->getMaterial( m );
      ICEMaterial* ice_matl = dynamic_cast<ICEMaterial*>(matl);
      MPMMaterial* mpm_matl = dynamic_cast<MPMMaterial*>(matl);

      int indx = matl->getDWIndex();
      new_dw->allocateTemporary(cv[m], patch);

      if(mpm_matl){                 // M P M
        new_dw->get( surfaceNorm[m],     d_surfaceNormLabel, indx, patch, gn, 0);

        CCVariable<double> oldTempMPM;
        new_dw->getCopy(       oldTempMPM, Ilb->temp_CCLabel,indx, patch, gn,0 );
        new_dw->getModifiable( vel_CC[m],  Ilb->vel_CCLabel, indx, patch, gn,0 );
        new_dw->getModifiable( Temp_CC[m], Ilb->temp_CCLabel,indx, patch, gn,0 );

        old_temp[m] = oldTempMPM;
        cv[m].initialize(mpm_matl->getSpecificHeat());
      }

      if(ice_matl){                 // I C E
        constCCVariable<double> cv_ice;
        old_dw->get( old_temp[m],     Ilb->temp_CCLabel,      indx, patch, gn, 0 );
        new_dw->get( cv_ice,          Ilb->specific_heatLabel,indx, patch, gn, 0 );
        new_dw->get( gamma[m],        Ilb->gammaLabel,        indx, patch, gn, 0 );
        new_dw->get( meanFreePath[m], d_meanFreePathLabel,    indx, patch, gn, 0 );

        new_dw->allocateTemporary( vel_CC[m],  patch );
        new_dw->allocateTemporary( Temp_CC[m], patch );
        cv[m].copyData(cv_ice);
      }
                                 // A L L  M A T L S

      new_dw->get( mass_L[m],        Ilb->mass_L_CCLabel,   indx, patch, gn, 0 );
      new_dw->get( sp_vol_CC[m],     Ilb->sp_vol_CCLabel,   indx, patch, gn, 0 );
      new_dw->get( mom_L[m],         Ilb->mom_L_CCLabel,    indx, patch, gn, 0 );
      new_dw->get( int_eng_L[m],     Ilb->int_eng_L_CCLabel,indx, patch, gn, 0 );
      new_dw->get( vol_frac_CC[m],   Ilb->vol_frac_CCLabel, indx, patch, gn, 0 );

      new_dw->allocateAndPut( Tdot[m],         Ilb->Tdot_CCLabel,    indx, patch );
      new_dw->allocateAndPut( mom_L_ME[m],     Ilb->mom_L_ME_CCLabel,indx, patch );
      new_dw->allocateAndPut( int_eng_L_ME[m], Ilb->eng_L_ME_CCLabel,indx, patch );
      new_dw->allocateAndPut( vel_T_CC[m],     d_vel_CCTransLabel,   indx, patch );
      vel_T_CC[m].initialize( Vector(0,0,0) );     // diagnostic variable

      new_dw->allocateTemporary( delta_vel_exch[m], patch );
      delta_vel_exch[m].initialize( Vector(0,0,0) );
    }

    //__________________________________
    // Convert momenta to velocities and internal energy to Temp
    for (int m = 0; m < d_numMatls; m++) {
      for(CellIterator iter = patch->getExtraCellIterator(); !iter.done();iter++){
        IntVector c = *iter;

        Temp_CC[m][c] = int_eng_L[m][c]/(mass_L[m][c]*cv[m][c]);
        vel_CC[m][c]  = mom_L[m][c]/mass_L[m][c];
      }
      const_vel_CC[m] = vel_CC[m];
    }

    //__________________________________
    // declare local variables
    double b[MAX_MATLS];
    FastMatrix beta(d_numMatls, d_numMatls);
    FastMatrix Q(3,3);
    FastMatrix a(d_numMatls, d_numMatls);
    FastMatrix K(d_numMatls, d_numMatls);
    FastMatrix h(d_numMatls, d_numMatls);
    FastMatrix H(d_numMatls, d_numMatls);

    // Initialize
    K.zero();
    h.zero();
    H.zero();

    d_exchCoeff->getConstantExchangeCoeff( K,h );

    //__________________________________
    //  compute the change in CC velocity due to exchange
    CellIterator cell_iterator = patch->getCellIterator();

    vel_CC_exchange( cell_iterator,  patch, K, delT,
                    isSurfaceCell, surfaceNorm,  vol_frac_CC, sp_vol_CC, /*mass_L,*/
                    meanFreePath,  const_vel_CC, vel_T_CC,  delta_vel_exch);

// update the velocity

    for (int m = 0; m < d_numMatls; m++) {
      for(CellIterator iter = patch->getExtraCellIterator(); !iter.done();iter++){
        IntVector c = *iter;
        vel_CC[m][c] += delta_vel_exch[m][c];
      }
    }

    //__________________________________
    //  E N E R G Y   E X C H A N G E

    int gm = d_fluidMatlIndx;
    int sm = d_solidMatlIndx;

    Vector dx = patch->dCell();

    for(CellIterator iter = patch->getCellIterator(); !iter.done();iter++){
      IntVector c = *iter;

      //
      H.copy(h);
      Q.identity();

      //__________________________________
      //  If cell is a surface cell modify exchange coefficients
      if ( isSurfaceCell[c] && d_useSlipCoeffs){

        // This should work for more that one solid.  It's hard wired for 2 matls-- Todd

        computeSurfaceRotationMatrix(Q, surfaceNorm[sm][c]); // Makes Q at each cell c

                       // For the temperature do you need to compute Q??  --Todd

        double A_V = 1.0/( dx.x()*fabs(Q(1,0)) +
                           dx.y()*fabs(Q(1,1)) +
                           dx.z()*fabs(Q(1,2)) );

        // thermal
        double at     = d_thermal_accommodation_coeff;
        double Beta_t = ((2 - at)/at) * (2/(1 + gamma[gm][c])) * (1/cv[gm][c]);

        H(gm,sm) = A_V / (Beta_t * meanFreePath[gm][c] * vol_frac_CC[sm][c]);  // The viscosity does not appear here because it's taken out of lambda

	 if(H(gm,sm) > h(gm,sm)) {
	   H(gm,sm) = h(gm,sm);
	 }
	 H(sm,gm) = H(gm,sm);
      }  // if a surface cell

      //__________________________________
      //  Perform exchange
      a.zero();

      for(int m = 0; m < d_numMatls; m++) {
        double adiag = 1.0;
        b[m] = 0.0;

        for(int n = 0; n < d_numMatls; n++) {
          a(m,n) = - delT * vol_frac_CC[n][c] * sp_vol_CC[m][c] * H(m,n) / cv[m][c];  // double check equation --Todd
          adiag -= a(m,n);
          b[m]  -= a(m,n) * (Temp_CC[n][c] - Temp_CC[m][c]);
        }
        a(m,m) = adiag;
      }

      a.destructiveSolve(b);

      for(int m = 0; m < d_numMatls; m++) {
        Temp_CC[m][c] += b[m];
      }
    }  //end CellIterator loop

    //__________________________________
    //  Boundary Condition Code
    if( BC_globalVars->usingLodi || BC_globalVars->usingMicroSlipBCs){

      std::vector<CCVariable<double> > temp_CC_Xchange(d_numMatls);
      std::vector<CCVariable<Vector> > vel_CC_Xchange(d_numMatls);

      for (int m = 0; m < d_numMatls; m++) {
        Material* matl = d_matlManager->getMaterial(m);
        int indx = matl->getDWIndex();

        new_dw->allocateAndPut(temp_CC_Xchange[m], Ilb->temp_CC_XchangeLabel, indx, patch);
        new_dw->allocateAndPut(vel_CC_Xchange[m],  Ilb->vel_CC_XchangeLabel,  indx, patch);
        vel_CC_Xchange[m].copy(vel_CC[m]);
        temp_CC_Xchange[m].copy(Temp_CC[m]);
      }
    }

    //__________________________________
    //  Set boundary conditions
    for (int m = 0; m < d_numMatls; m++)  {
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
      IntVector c = *iter;                                              // shouldn't the loop over matls be outside the cell iterator for speed. --Todd
      for (int m = 0; m < d_numMatls; m++) {
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

  const MaterialSet* ice_matls = d_matlManager->allMaterials( "ICE" );
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

    int numICEMatls = d_matlManager->getNumMatls( "ICE" );
    Ghost::GhostType  gn = Ghost::None;

    for (int m = 0; m < numICEMatls; m++) {
      ICEMaterial* ice_matl = (ICEMaterial*) d_matlManager->getMaterial( "ICE", m);
      int indx = ice_matl->getDWIndex();
      constCCVariable<double> temp;
      constCCVariable<double> sp_vol;
      constCCVariable<double> gamma;
      constCCVariable<double> cv;
      CCVariable<double>      meanFreePath;

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
void SlipExch::computeSurfaceRotationMatrix(FastMatrix   & Q,
                                            const Vector & surfaceNorm )
{
  Q.zero();

  if(fabs(surfaceNorm[1]) > 0.9999) {
    Q.identity();
  }else {                               //if coord.'s are not already aligned with surface
    Q(1,0) = surfaceNorm[0];
    Q(1,1) = surfaceNorm[1];
    Q(1,2) = surfaceNorm[2];

    double sqrtTerm = sqrt(1 - Q(1,1) * Q(1,1));
    double invTerm  = 1.0/sqrtTerm;

    Q(0,0) = Q(1,1);
    Q(0,1) =-Q(1,0);
    Q(0,2) = Q(1,1) * Q(1,2) * invTerm;
    Q(2,0) =-Q(1,2) * invTerm;
    Q(2,1) = 0.0;
    Q(2,2) = Q(1,0) * invTerm;
  }
}
