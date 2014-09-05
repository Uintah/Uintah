/// RMCRTRadiationModel.cc-------------------------------------------------------
/// Reverse Monte Carlo Ray Tracing Radiation Model interface
/// 
/// @author Xiaojing Sun ( Paula ) and Jeremy Thornock
/// @date Feb 20, 2009.
///
/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
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
//#include <CCA/Components/Arches/MCRT/ArchesRMCRT/RMCRTFactory.h>
//#include <CCA/Components/Arches/MCRT/ArchesRMCRT/RMCRTnoInterpolation.h>
//#include <CCA/Components/Arches/MCRT/ArchesRMCRT/RMCRTRRSD.h>
#include <CCA/Components/Arches/MCRT/ArchesRMCRT/RMCRTRRSDStratified.h>
#include <CCA/Components/Arches/BoundaryCondition.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/SimulationState.h>
#include <CCA/Components/Arches/CellInformation.h>
#include <CCA/Components/Arches/CellInformationP.h>

#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/MCRT/ArchesRMCRT/RMCRTRadiationModel.h>
#include <CCA/Components/Arches/TimeIntegratorLabel.h>

//class RMCRTFactory;

using namespace Uintah; 
using namespace std;

//---------------------------------------------------------------------------
//  Constructor
//---------------------------------------------------------------------------
RMCRTRadiationModel::RMCRTRadiationModel( const ArchesLabel* label, BoundaryCondition* bc ) : 
d_lab(label) 
{
  d_mmWallID = bc->getMMWallId();  //kumar's multimaterial (mpm) wall 
  d_wallID   = bc->wallCellType(); //regular old wall
  d_flowID   = bc->flowCellType(); //flow cell 
}

//---------------------------------------------------------------------------
//  Destructor
//---------------------------------------------------------------------------
RMCRTRadiationModel::~RMCRTRadiationModel()
{
}

//---------------------------------------------------------------------------
//  ProblemSetup
//---------------------------------------------------------------------------
void 
RMCRTRadiationModel::problemSetup( const ProblemSpecP& params )
{
  // This will look for a block in the input file called <RMCRTRadiationModel>
  ProblemSpecP db_rad = params->findBlock("RMCRTRadiationModel");

  string prop_model; 
  
  // ask for number of rays
  db_rad->require("const_num_rays", d_constNumRays);

  // if using russian roulette
  db_rad->getWithDefault("rr_threshold", d_StopLowerBound, 1.e-4);
  
//   if ( db_rad->findBlock("russian_roulette")){
//     d_rr = true; 
//     db_rad->getWithDefault("rr_threshold", d_StopLowerBound, 1.e-4);
//   }
  
  // property model set up
  // radcoef requires opl 
  db_rad->require("opl",d_opl);
  db_rad->require("property_model", prop_model);
  
  db_rad->getWithDefault("sampling_scheme", sample_sche, "simple_sample");

  if ( sample_sche == "simple_sample" || sample_sche == "RRSD"){
    db_rad->getWithDefault("x_n", i_n, 1);
    db_rad->getWithDefault("y_n", j_n, 1);
    db_rad->getWithDefault("z_n", k_n, 1);
    db_rad->getWithDefault("theta_n", theta_n, 1);
    db_rad->getWithDefault("phi_n", phi_n, 1);
  }
  else if ( sample_sche == "RRSDstratified"){
  
    ProblemSpecP db_stratified = db_rad->findBlock("RRSDstratified");
    db_stratified->require("x_n", i_n);
    db_stratified->require("y_n", j_n);
    db_stratified->require("z_n", k_n);
    db_stratified->require("theta_n", theta_n);
    db_stratified->require("phi_n", phi_n);
    
  }
    
    
  if (prop_model == "radcoef"){  // have this in c++
    d_radcal     = false;
    d_wsgg       = false;
    d_ambda      = 1;
    d_planckmean = false;
    d_patchmean  = false;
    d_fssk       = false;
    d_fsck       = false;
  }

  if (prop_model == "patchmean"){ 
    cout << "WARNING! Serial and parallel results may deviate for this model" << endl;
    d_radcal     = true;
    d_wsgg       = false;
    d_ambda      = 6;
    d_planckmean = false;
    d_patchmean  = true;
    d_fssk       = false;
    d_fsck       = false;    
  }

  if (prop_model == "wsggm"){ // have this in c++
    throw InternalError("WSGG radiation model does not run in parallel and has been disabled", __FILE__, __LINE__);
    d_radcal       = false;
    d_wsgg         = true;
    d_ambda        = 4;
    d_planckmean   = false;
    d_patchmean    = false;
    d_fssk         = false;
    d_fsck         = false;    
  }


  // fssk and fsck needs the stretching factor a for different reference states
  // however 'a' is calculated every update
  // the reference state is also updated every radiation time
  // but we need an initial reference state
  if (prop_model == "fssk"){
    db_rad->require("reference_temperature", d_refTemp);
    // ref_comp is a vector of all gas mixtures composition
    // how to decide the size of ref_comp?
    db_rad->require("reference_composition", d_refComp);
    db_rad->require("scaling_function",d_uScale);    
    // db_rad->require("a_fssk", d_afssk);

    
    d_radcal       = false;
    d_wsgg         = false;
    //   d_ambda        = 4;
    d_planckmean   = false;
    d_patchmean    = false;
    d_fssk         = false;
    d_fsck         = false;    
  }

  if (prop_model == "fsck"){
    
    db_rad->require("reference_temperature", d_refTemp);
    // ref_comp is a vector of all gas mixtures composition
    // how to decide the size of ref_comp?
    db_rad->require("reference_composition", d_refComp);
    //  db_rad->require("a_fsck", d_afsck);

    d_radcal       = false;
    d_wsgg         = false;
    //   d_ambda        = 4;
    d_planckmean   = false;
    d_patchmean    = false;
    d_fssk         = false;
    d_fsck         = false;        

  }
  
  
  
}

//---------------------------------------------------------------------------
//  Schedule the solve of the RTE using RMCRT
//---------------------------------------------------------------------------

void 
RMCRTRadiationModel::sched_solve( const LevelP& level, SchedulerP& sched, const TimeIntegratorLabel* timeLabels )
{
  const string taskname = "RMCRTRadiationModel::solve"; 
  Task* tsk = scinew Task(taskname, this, &RMCRTRadiationModel::solve, timeLabels); 

  //Variables needed from DW
  Ghost::GhostType  gac = Ghost::AroundCells;
  //Ghost::GhostType  gaf = Ghost::AroundFaces;
  //Ghost::GhostType  gn = Ghost::None;
  
  tsk->requires(Task::OldDW, d_lab->d_tempINLabel, gac, 1); // getting temperature w/1 ghost to use only (not to modify it)
  tsk->requires(Task::OldDW, d_lab->d_absorpINLabel, gac, 1); // getting absorption coef w/1 ghost
  tsk->requires(Task::OldDW, d_lab->d_cellTypeLabel, gac, 1);

  if (timeLabels->integrator_step_number == TimeIntegratorStepNumber::First){
    tsk->computes(d_lab->d_tempFxLabel);
    tsk->computes(d_lab->d_tempFyLabel);
    tsk->computes(d_lab->d_tempFzLabel);
  } else {
    tsk->modifies(d_lab->d_tempFxLabel);
    tsk->modifies(d_lab->d_tempFyLabel);
    tsk->modifies(d_lab->d_tempFzLabel);
  } 

  sched->addTask(tsk, level->eachPatch(), d_lab->d_sharedState->allArchesMaterials()); 

}



//---------------------------------------------------------------------------
//  Actually solve the RTE using RMCRT
//---------------------------------------------------------------------------
void 
RMCRTRadiationModel::solve(  const ProcessorGroup* pc, 
                             const PatchSubset* patches,
                             const MaterialSubset*, 
                             DataWarehouse* old_dw, 
                             DataWarehouse* new_dw,
                             const TimeIntegratorLabel* timeLabels )

{
  Ghost::GhostType  gac = Ghost::AroundCells;
  //Ghost::GhostType  gaf = Ghost::AroundFaces;
  //Ghost::GhostType  gn = Ghost::None;

  // need to pass in property, RR, stratified sampling para.

  // generate a root class RMCRTScheme
  // then RMCRTScheme *obRMCRT in factory
  // return new derivedClass

  // RMCRTFactory *obRMCRT;
  // obRMCRT = RMCRTFactory::RMCRTModel(sample_sche);
  
  RMCRTRRSDStratified *obRMCRT;
  obRMCRT=0;

  // patch loop 
  for ( int p = 0; p < patches->size(); p++ ) {
    const Patch* patch = patches->get(p); 
    int archIndex = 0; // only one arches material for now
    //int matlIndex = 0; 

    // get temperature and absorption coefficient 
    constCCVariable<double> T; 
    constCCVariable<double> absorpCoef; 
    constCCVariable<double> scatterCoeff;
    constCCVariable<int> cellType; 

    SFCXVariable<double> Tx;
    SFCYVariable<double> Ty; 
    SFCZVariable<double> Tz; 

    // have to reset my CCVariables
    // so that i can initialize them.
    // wondering if they have got the same structure as obtained from DW
    CCVariable<double> Ttest;
    CCVariable<double> absorpCoeftest;

    new_dw->allocateTemporary(Ttest, patch);
    new_dw->allocateTemporary(absorpCoeftest, patch);
    
    // what about absorb_surface, rs_surface, and rd_surface?
    // work the same way as T ?
    // but there are no rs, rd for media at all.
    // temperature variables on physical boundaries.
    double Tleft, Tright, Ttop, Tbottom, Tfront, Tback;
    Tleft = 0;
    Tright = 0;
    Ttop = 0;
    Tbottom = 0;
    Tfront = 0;
    Tback = 0;
    
    int currI=0, currJ=0, currK=0;
    // emission coefficient on boundaries are CCVariable? or SFXVariable?
    // absorption coefficient on boundaries? define as what type?
    // rs, rd?
    
    // get emission coeff on boundaries
    // get X Y Z
    // get cellType wall or flow
    // X node, Y node, Z node numbers    
    //  CellInformation* cellinfo, cellinfo->x, cellinfo->y..
    //  ArchesVariables* vars, vars->temperature
    //  ArchesConstVariables* constvars) constvars->cellType
    
    old_dw->get( T, d_lab->d_tempINLabel, 0, patch, gac, 1 ); 
    old_dw->get( absorpCoef, d_lab->d_absorpINLabel, 0, patch, gac, 1 );
    old_dw->get( cellType, d_lab->d_cellTypeLabel, 0, patch, gac, 1 ); 

    if (timeLabels->integrator_step_number == TimeIntegratorStepNumber::First){
      new_dw->allocateAndPut( Tx, d_lab->d_tempFxLabel, archIndex, patch );
      new_dw->allocateAndPut( Ty, d_lab->d_tempFyLabel, archIndex, patch ); 
      new_dw->allocateAndPut( Tz, d_lab->d_tempFzLabel, archIndex, patch ); 
    } else {
      new_dw->getModifiable( Tx, d_lab->d_tempFxLabel, archIndex, patch ); 
      new_dw->getModifiable( Ty, d_lab->d_tempFyLabel, archIndex, patch ); 
      new_dw->getModifiable( Tz, d_lab->d_tempFzLabel, archIndex, patch ); 
    }

    IntVector currCell(currI, currJ, currK);

    // this is the interior's boundary indices
    // are these returning an array of indices???
    IntVector idxLo = patch->getCellLowIndex();
    IntVector idxHi = patch->getCellHighIndex();

    // this is the SFCY's boundary indices
    // what values are idxLos and idxHis??
    // an array of indices????
    IntVector idxLos = patch->getSFCYLowIndex();
    IntVector idxHis = patch->getSFCYHighIndex();

    IntVector idxLoe = patch->getExtraCellLowIndex();
    IntVector idxHie = patch->getExtraCellHighIndex();
    
    cout << "Cell idxLo = " << idxLo << endl;
    cout << "cell idxHi = " << idxHi << endl;
    cout << "SFCY idxLos = " << idxLos << endl;
    cout << "SFCY idxHis = " << idxHis << endl;
    cout << "Extra idxLoe = " << idxLoe << endl;
    cout << "Extra idxHie = " << idxHie << endl;
    
    /*
      for (CellIterator iter=patch->getCellIterator();!iter.done(); iter++){
      Point p = patch->cellPosition(*iter);
      Point p_xp = patch->cellPosition(*iter + IntVector(1,0,0));
      Point p_xm = patch->cellPosition(*iter - IntVector(1,0,0));
      Point p_yp = patch->cellPosition(*iter + IntVector(0,1,0));
      Point p_ym = patch->cellPosition(*iter - IntVector(0,1,0));
      Point p_ymm = patch->cellPosition(*iter - IntVector(0,2,0));
      Point p_zp = patch->cellPosition(*iter + IntVector(0,0,1));
      Point p_zm = patch->cellPosition(*iter - IntVector(0,0,1));
      Point p_zmm = patch->cellPosition(*iter - IntVector(0,0,2));
    */
    
    // this is only iteratioing over interior cells.
    // to be able to follow currCell,
    // i have to change all of the variables???
    // cuz my indexes are different from Arches.

      // error in here, seems cannot get to Ttest[currCell]
	// because i just claim it myself, and it doesnot have the dimension
	// it is supposed to have from datawarehouse???
	
    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){
      IntVector currCell = *iter;
       cout << "i am in the patch" << endl;
       cout << "currCell = " << currCell << endl;
       
       if ( currCell.x() == idxLo.x() )  {
	 // left side boundary
	 cout << "currcellx() o the left face" << endl;
	 cout << "Ttest[currCell] = " << Ttest[currCell] << endl;
	  cout << "Ttest[currCell - IntVector(1,0,0)] = " <<
 	   Ttest[currCell - IntVector(1,0,0)] << endl;	 
	Ttest[currCell - IntVector(1, 0, 0)] = Tleft;
	cout << " i am on the left surface " << endl;
       }
      else if ( currCell.x() == idxHi.x() ) // right side 
	Ttest[currCell + IntVector(1, 0, 0)] = Tright;
      
      if ( currCell.y() == idxLo.y() ) // front side boundary
	Ttest[currCell - IntVector(0, 1, 0)] = Tfront;
      else if ( currCell.y() == idxHi.y() ) // back side boundary
	Ttest[currCell + IntVector(0, 1, 0)] = Tback;
      
      if ( currCell.z() == idxLo.z() ) //  bottom side boundary
	Ttest[currCell - IntVector(0, 0, 1)] = Tbottom;
      else if ( currCell.y() == idxHi.z() ) // top side boundary
	Ttest[currCell + IntVector(0, 0, 1)] = Ttop;
      
      
      Ttest[currCell] = 64.80721904;
      
      cout << "i am before point p " << endl;
      Point p = patch->cellPosition(*iter);
      cout << "i am after point p " << endl;
      // the p.x(), p.y(), p.z() are the cell centered x, y, z.
      absorpCoeftest[currCell] = 0.9 * ( 1 - 2 * abs ( p.x() ) )
	* ( 1 - 2 * abs ( p.y() ) )
	* ( 1 - 2 * abs ( p.z() ) ) + 0.1;
      // cout << "i am in the patch" << endl;
				     
    }
    
    // concluseion set my own Ttest, wont work.
    // seems have to get it from DW
    
      
    // interpolate temperatures to FC
    // changed T to Ttest
    IntVector dir(1,0,0);
    IntVector highIdx = patch->getCellHighIndex(); 
    interpCCTemperatureToFC( cellType, Tx, dir, highIdx, Ttest, patch ); 
    dir += IntVector(-1,1,0); 
    interpCCTemperatureToFC( cellType, Ty, dir, highIdx, Ttest, patch ); 
    dir += IntVector(0,-1,1); 
    interpCCTemperatureToFC( cellType, Tz, dir, highIdx, Ttest, patch );
 
    
    // where does the currCell starts? for b.c. cells and ghost cells
    
    cout << "GOING TO CALL STAND ALONE SOLVER!\n"; 
    

    /*
      obRMCRT->RMCRTsolver(constCCVariable<int>& cellType,
      constCCVariable<double> &T,
      SFCXVariable<double> &Tx,
      SFCYVariable<double> &Ty, 
      SFCZVariable<double> &Tz )
    */

    // Tx, Ty, Tz??
    // what the index should be when i use Tx for left and right bc?
    obRMCRT->RMCRTsolver(i_n, j_n, k_n, theta_n, phi_n);
  } // end patch loop 

}

 

