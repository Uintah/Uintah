#include <iostream>
#include <iomanip>
#include <cmath>
#include <limits>
#include <CCA/Components/Arches/PressureSolverV2.h>
#include <CCA/Components/Arches/linSolver.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/ArchesMaterial.h>
#include <CCA/Components/Arches/BoundaryCondition.h>
#include <CCA/Components/Arches/Discretization.h>

#include <CCA/Components/Arches/PhysicalConstants.h>

#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Grid/DbgOutput.h>
#include <Core/Grid/MaterialManager.h>
#include <Core/Grid/MaterialManagerP.h>
#include <Core/Grid/Variables/PerPatch.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/Timers/Timers.hpp>


#include <Core/Datatypes/DenseMatrix.h>
#include <Core/Datatypes/Matrix.h>
#include <Core/Datatypes/MatrixOperations.h>
#include <Core/Datatypes/SparseRowMatrix.h>
#include <Core/Grid/AMR.h>
#include <iostream>
using namespace std;
 namespace Uintah{


void
linSolver::sched_PreconditionerConstruction(SchedulerP& sched, const MaterialSet* matls,const LevelP& level){  

    cout << level->getID() << " LEVEL derekx \n";
//if (level->hasFinerLevel()){
//return;
//}
  const PatchSet * patches=level->eachPatch();
  d_blockSize=1; d_stencilWidth=(d_blockSize-1)+d_blockSize  ; // NOT TRUE FOR EVEN BLOCK SIZE 
  //d_custom_relax_type=jacobi_relax;
  d_custom_relax_type=redBlack;

  if (d_custom_relax_type==redBlack){
    cg_ghost=1;   // number of rb iterations
  }else{
    cg_ghost=d_blockSize-1;   // Not true for even block size
  }

  if (level->hasFinerLevel() == false){  // only create labels first time through ( hhighest level) 
  const TypeDescription* CC_double = CCVariable<double>::getTypeDescription();
    if (d_custom_relax_type==jacobi_relax){
      for (int kx=0;kx<d_stencilWidth;kx++){
        for (int jx=0;jx<d_stencilWidth;jx++){
          for (int ix=0;ix<d_stencilWidth;ix++){
            ostringstream my_stringstream_object;
            my_stringstream_object << "precM_" << setfill('0') << setw(2)<<  ix  << "_"<< setw(2) << jx <<"_" << setw(2)<< kx ;
            d_precMLabel.push_back(VarLabel::create(my_stringstream_object.str(),  CC_double));
          }
        }
      }
    }else if(d_custom_relax_type==redBlack){
      ////////////// ASSSUME SYMMETRIC SYSTEM!!!!!!!!!!!! ///////////////
      d_precMLabel.push_back(VarLabel::create("mg_A_p",  CC_double)); // mutigrid A -cetnral
      d_precMLabel.push_back(VarLabel::create("mg_A_w",  CC_double)); // mutigrid A - west 
      d_precMLabel.push_back(VarLabel::create("mg_A_s",  CC_double)); // mutigrid A - north
      d_precMLabel.push_back(VarLabel::create("mg_A_b",  CC_double)); // mutigrid A - south
      ////////////// For non-symmetric ////////////////////
      //d_precMLabel.push_back(VarLabel::create("mg_A_e",  CC_double)); // mutigrid A - west 
      //d_precMLabel.push_back(VarLabel::create("mg_A_n",  CC_double)); // mutigrid A - north
    d_precMLabel.push_back(VarLabel::create("mg_A_t",  CC_double)); // mutigrid A - south

    }
    d_residualLabel = VarLabel::create("cg_residual",  CC_double);
    d_littleQLabel = VarLabel::create("littleQ",  CC_double);
    d_bigZLabel    = VarLabel::create("bigZ",  CC_double);
    d_smallPLabel  = VarLabel::create("smallP",  CC_double);


  }

    cg_n_iter=30;
    for (int i=0 ; i < cg_n_iter ; i++){
      d_convMaxLabel.push_back(  VarLabel::create("convergence_check"+ std::to_string(i),   max_vartype::getTypeDescription()));
      d_corrSumLabel.push_back(  VarLabel::create("correctionSum"+ std::to_string(i),   sum_vartype::getTypeDescription()));
      d_resSumLabel.push_back(  VarLabel::create("residualSum"+ std::to_string(i),   sum_vartype::getTypeDescription()));
    }
      d_resSumLabel.push_back(  VarLabel::create("residualSum_"+ std::to_string(cg_n_iter),   sum_vartype::getTypeDescription()));



  proc0cout << " SCHEDULING CUSTOM SOLVE derekx\n";
  std::string taskname("linSolver::initialize_preconditioner_" + std::to_string(level->getID()));
  if (level->hasFinerLevel() == false){  // only create labels first time through ( hhighest level) 
  sched_buildAMatrix( sched, patches, matls);
  const VarLabel* ALabel = d_presCoefPBLMLabel;
  Task* tsk = scinew Task(taskname, this, &linSolver::customSolve,ALabel);

  tsk->requires(Task::NewDW, ALabel, Ghost::AroundCells, 1);   // ghosts needed fro jjacobiblock, not jacobi

  for (unsigned int i=0;i<d_precMLabel.size();i++){
    tsk->computes(d_precMLabel[i]);
  }

  sched->addTask(tsk, patches, matls);



} else {
  Task* tskc = scinew Task("linSolver::coarsenA_"+std::to_string(level->getID()), this, &linSolver::coarsen_A);
  for (unsigned int i=0;i<d_precMLabel.size();i++){
    //tsk->requires( fineLevel_Q_dw, variable, 0, Task::FineLevel, 0, Task::NormalDomain, d_gn, 0 );
    tskc->requires( Task::NewDW ,d_precMLabel[i], 0, Task::FineLevel, 0, Task::NormalDomain, Ghost::None, 0 );
    tskc->computes(d_precMLabel[i]);
  }

  proc0cout << " SCHEDULING CUSTOM SOLVE  on COARSE LEVEL derekx\n";
  sched->addTask(tskc, patches, matls);
}


 //This allows us to populate ghosts into the "static" a_matrix, and is otherwise unnecessary, this could have been donw ith more complex upstream logic, but this is much more simple.
  Task* tsk_hack = scinew Task("linSolver::fillGhosts", this, &linSolver::fillGhosts);
  for (unsigned int i=0;i<d_precMLabel.size();i++){
    tsk_hack->requires( Task::NewDW ,d_precMLabel[i], Ghost::AroundCells, cg_ghost+1 ); // +1 not needed for i=0;
    tsk_hack->modifies( d_precMLabel[i] ); // +1 not needed for i=0;
  }
  sched->addTask(tsk_hack, patches, matls);




}

/// THIS IS A MULTILEVEL TASK, its purpose is only to populate ghost cells, once at begining of simulation.
void 
linSolver::fillGhosts( const ProcessorGroup* pg,
             const PatchSubset* patches,
             const MaterialSubset* matls,
             DataWarehouse* old_dw,
             DataWarehouse* new_dw
             ){
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    for (unsigned int i=0;i<d_precMLabel.size();i++){
      
     constCCVariable<double>  x; 
     new_dw->get(x,d_precMLabel[i],d_indx,patch,Ghost::AroundCells,cg_ghost+1);  
     //cout << " fillGhosts Derex " << *d_precMLabel[i] << " "  << x.getDataSize() << " " << x.getWindow()->getOffset() <<"\n";
      }
  }
};


void
linSolver::sched_buildAMatrix(SchedulerP& sched,
                                        const PatchSet* patches,
                                        const MaterialSet* matls)
{
  //  build pressure equation coefficients and source
  string taskname =  "linSolver::buildAMatrix_" ;

  Task* tsk = scinew Task(taskname, this,
                          &linSolver::buildAMatrix
                          );


  Ghost::GhostType  gac = Ghost::AroundCells;
  //Ghost::GhostType  gn  = Ghost::None;
  //Ghost::GhostType  gaf = Ghost::AroundFaces;

  tsk->requires(Task::NewDW, d_cellTypeLabel,       gac, 1);
  // get drhodt that goes in the rhs of the pressure equation
  if (d_MAlab) {
    tsk->requires(Task::NewDW, d_mmgasVolFracLabel, gac, 1);
  }

  tsk->computes(d_presCoefPBLMLabel);

  sched->addTask(tsk, patches, matls);
}

void
linSolver::buildAMatrix(const ProcessorGroup* pc,
                                  const PatchSubset* patches,
                                  const MaterialSubset* /* matls */,
                                  DataWarehouse* old_dw,
                                  DataWarehouse* new_dw
                                  )
{


  Discretization* discrete = scinew Discretization(d_physicalConsts);

  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);

    ArchesConstVariables constVars;
    ArchesVariables vars;

    //Ghost::GhostType  gn = Ghost::None;
    Ghost::GhostType  gac = Ghost::AroundCells;
    //Ghost::GhostType  gaf = Ghost::AroundFaces;

    new_dw->get(constVars.cellType,     d_cellTypeLabel,    d_indx, patch, gac, 1);

    new_dw->allocateAndPut(vars.pressCoeff,        d_presCoefPBLMLabel,      d_indx, patch);

    const IntVector idxLo = patch->getCellLowIndex();
    const IntVector idxHi = patch->getCellHighIndex();

    vars.pressNonlinearSrc.allocate(idxLo,idxHi);
    vars.pressLinearSrc.allocate(idxLo,idxHi);

    vars.pressNonlinearSrc.initialize(0.0); // b matrix dummy variable
    vars.pressLinearSrc.initialize(0.0); // b matrix dummy variable

  for ( int k=idxLo.z(); k< idxHi.z(); k++ ){
    for ( int j=idxLo.y(); j< idxHi.y(); j++ ){
      for ( int i=idxLo.x(); i< idxHi.x(); i++ ){ // annoying stencil7, better way to initialize?
       vars.pressCoeff(i,j,k).initialize(0.0);
      }}}

    d_pressureSolver->calculatePressureCoeff(patch, &vars, &constVars);
    // Modify pressure coefficients for multimaterial formulation
    if (d_MAlab) {
      new_dw->get(constVars.voidFraction, d_mmgasVolFracLabel, d_indx, patch,gac, 1);
    }

    //Vector Dx = patch->dCell();
    //double volume = Dx.x()*Dx.y()*Dx.z();
    
    d_boundaryCondition->mmpressureBC(new_dw, patch,
                                      &vars, &constVars);
    // Calculate Pressure Diagonal
    discrete->calculatePressDiagonal(patch, &vars);

    d_boundaryCondition->pressureBC(patch, d_indx, &vars, &constVars);

  }
  delete discrete;
}

void
linSolver::printError( const ProcessorGroup* pg,
                             const PatchSubset* patches,
                             const MaterialSubset* matls,
                             DataWarehouse* old_dw,
                             DataWarehouse* new_dw,
                             const VarLabel* ALabel,
                             const VarLabel* xLabel,
                             const VarLabel* bLabel){
  
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    int matl = d_indx ;



  const IntVector idxLo = patch->getCellLowIndex();
  const IntVector idxHi = patch->getCellHighIndex();
  constCCVariable<Stencil7> A_m;  // matrix, central element 5 5 5
  constCCVariable<double> x_v; // x-vector
  constCCVariable<double> b_v; // b-vector

  new_dw->get(A_m, ALabel,      matl , patch, Ghost::None, 0);
  new_dw->get(x_v, xLabel,      matl , patch, Ghost::AroundCells, 1);
  new_dw->get(b_v, bLabel,      matl , patch, Ghost::None, 0);

       double maxError=0.;
       for ( int k=idxLo.z(); k< idxHi.z(); k++ ){
         for ( int j=idxLo.y(); j< idxHi.y(); j++ ){
           for ( int i=idxLo.x(); i< idxHi.x(); i++ ){ //compute correction 
               maxError=max(abs(b_v(i,j,k)-(A_m(i,j,k)[6]*x_v(i,j,k)+
                                           A_m(i,j,k)[0]*x_v(i-1,j,k)+
                                           A_m(i,j,k)[1]*x_v(i+1,j,k)+
                                           A_m(i,j,k)[2]*x_v(i,j-1,k)+
                                           A_m(i,j,k)[3]*x_v(i,j+1,k)+
                                           A_m(i,j,k)[4]*x_v(i,j,k-1)+
                                           A_m(i,j,k)[5]*x_v(i,j,k+1))),maxError);  

}
}
}
              //cout <<std::setprecision(16)  <<"tol=" <<maxError  << "\n";
              cout <<"tol=" <<maxError  << "\n";
}

}

void
linSolver::Update_preconditioner( const ProcessorGroup* pg,
             const PatchSubset* patches,
             const MaterialSubset* matls,
             DataWarehouse* old_dw,
             DataWarehouse* new_dw
             ){
  for (unsigned int i=0;i<d_precMLabel.size();i++){

//constCCVariable<double > x;
    //new_dw->get(A_m, ALabel,      matl , patch, Ghost::None, 0);
     //old_dw->get(x,d_precMLabel[i],matl,patches->get(0),Ghost::AroundCells,1);  
     //cout << "1rst TRANSFER Derex " << *d_precMLabel[i] << " " <<  x.getDataSize() << "  " <<x.getLowIndex() << "\n";
     new_dw->transferFrom(old_dw,d_precMLabel[i],patches,matls);
     //new_dw->get(x,d_precMLabel[i],matl,patches->get(0),Ghost::None,0);  
     //cout <<  x.getDataSize() << "\n";


      }


}

void linSolver::sched_customSolve(SchedulerP& sched, const MaterialSet* matls,
                                   const PatchSet* patches,
               const VarLabel*  ALabel,     Task::WhichDW A_dw,
               const VarLabel*  xLabel,     Task::WhichDW modifies_x,
               const VarLabel*  bLabel,     Task::WhichDW b_dw,
               const VarLabel*  guess, Task::WhichDW guess_dw, int rk_step, LevelP fineLevel ){

//sched_PreconditionerConstruction( sched,  matls, patches,nullptr);  

  int  maxLevels=fineLevel->getGrid()->numLevels();
  GridP grid = fineLevel->getGrid();

  if(rk_step==0){
    int maxLevels = grid->numLevels();
    for (int l = 0; l < maxLevels; ++l) {
      const LevelP& coarse_level = grid->getLevel(l);
      const PatchSet * level_patches=coarse_level->eachPatch();

      string taskname_update =  "linSolver::update_preconditioner_";
      Task* task_u = scinew Task(taskname_update, this, &linSolver::Update_preconditioner);
      for (unsigned int i=0;i<d_precMLabel.size();i++){
        task_u->requires(Task::OldDW,d_precMLabel[i],Ghost::AroundCells,cg_ghost+1);
        task_u->computes(d_precMLabel[i]);// the right way, but may impose unnecessary ghost cell communication
      }
      sched->addTask(task_u, level_patches, matls);
    }
  }


////------------------ set up CG solver---------------------//
  string taskname_i1 =  "linSolver::cg_init1";
  Task* task_i1 = scinew Task(taskname_i1, this, &linSolver::cg_init1,ALabel,xLabel,bLabel,guess);


  task_i1->requires(Task::NewDW, guess, Ghost::AroundCells, 1);
  task_i1->requires(Task::NewDW, bLabel, Ghost::None, 0);
  task_i1->requires(Task::NewDW, ALabel, Ghost::AroundCells, 1);
  task_i1->computes(xLabel);
  if (rk_step==0){
  task_i1->computes(d_residualLabel);
  }else{
  task_i1->modifies(d_residualLabel);
  }

  sched->addTask(task_i1, fineLevel->eachPatch(), matls);

  int  total_rb_switch=30;
  int final_iter=total_rb_switch-1;
for (int rb_iter=0; rb_iter < total_rb_switch; rb_iter++){
  string taskname_i2 =  "linSolver::cg_init2";
  Task* task_i2 = scinew Task(taskname_i2, this, &linSolver::cg_init2, rb_iter);

  if (rk_step==0 && rb_iter==0){
  task_i2->computes(d_smallPLabel);
}else{
  task_i2->modifies(d_smallPLabel);
}

if (rb_iter == final_iter){
  if (rk_step==0 ){
    task_i2->computesWithScratchGhost
      (d_bigZLabel, nullptr, Uintah::Task::NormalDomain,Ghost::AroundCells,cg_ghost);
    task_i2->computes(d_littleQLabel);
    task_i2->computes(d_resSumLabel[0]);
  }else{
    task_i2->modifies(d_smallPLabel);
    task_i2->modifies(d_bigZLabel);
    task_i2->modifies(d_littleQLabel);
    //task_i2->computes(d_resSumLabel[0]);   // NEED FOR RK2
  }
}


  task_i2->requires(Task::NewDW, d_residualLabel, Ghost::AroundCells, cg_ghost); 

    for (unsigned int i=0;i<d_precMLabel.size();i++){
      task_i2->requires(Task::NewDW,d_precMLabel[i],Ghost::AroundCells,cg_ghost+1); //  possibly not needed...
  }

  sched->addTask(task_i2,fineLevel->eachPatch(), matls);
}


  for (int cg_iter=0 ; cg_iter < cg_n_iter ; cg_iter++){
  Task* task1 = scinew Task("linSolver::cg_task1" ,
                           this, &linSolver::cg_task1,ALabel, cg_iter);
  task1->requires(Task::NewDW,ALabel,Ghost::None, 0 );
  task1->requires(Task::NewDW,d_smallPLabel, Ghost::AroundCells, 1);
  task1->modifies(d_littleQLabel);
  task1->computes(d_corrSumLabel[cg_iter]);

  sched->addTask(task1, fineLevel->eachPatch(),matls);

  Task* task2 = scinew Task("linSolver::cg_task2",
                           this, &linSolver::cg_task2,xLabel, cg_iter);
  task2->requires(Task::NewDW,d_corrSumLabel[cg_iter]);
  task2->requires(Task::NewDW,d_resSumLabel[cg_iter]);
  task2->requires(Task::NewDW,d_littleQLabel, Ghost::None, 0);
  task2->requires(Task::NewDW,d_smallPLabel, Ghost::None, 0);
  task2->modifies(xLabel);
  task2->modifies(d_residualLabel);


  sched->addTask(task2, fineLevel->eachPatch(),matls);



  for (int l = maxLevels-2; l > -1; l--) { // Coarsen fine to coarse
    const LevelP& coarse_level = grid->getLevel(l);

    const PatchSet * level_patches=coarse_level->eachPatch();
    Task* task_multigrid_up = scinew Task("linSolver::cg_coarsenResidual",
        this, &linSolver::cg_moveResUp, cg_iter);

    task_multigrid_up->requires( Task::NewDW ,d_residualLabel, 0, Task::FineLevel, 0, Task::NormalDomain, Ghost::None, 0 );
    if ( cg_iter==0){
      task_multigrid_up->computes(d_residualLabel);
    }else{
      task_multigrid_up->modifies(d_residualLabel);

    }
    sched->addTask(task_multigrid_up, level_patches,matls);
  }



  for (int l = 0; l < maxLevels; ++l) {
    const LevelP& coarse_level = grid->getLevel(l);
    const PatchSet * level_patches=coarse_level->eachPatch();

    Task* task_multigrid_down = scinew Task("linSolver::cg_multigridDown",
        this, &linSolver::cg_multigrid_down, cg_iter);

    if (l<maxLevels-1 && cg_iter==0){
      task_multigrid_down->computes(d_bigZLabel);
    }else{
      task_multigrid_down->modifies(d_bigZLabel);
    }

    if (l>0){
        //int offset = 1;
        task_multigrid_down->requires( Task::NewDW ,d_bigZLabel, 0, Task::CoarseLevel, 0, Task::NormalDomain, Ghost::None, 0 );
    }
    
    sched->addTask(task_multigrid_down, level_patches,matls);

    int smoother_iter=3;// In case user didn't set up levels

    for (int red_black_switch=smoother_iter-1; red_black_switch > -1; red_black_switch--){
      Task* task_multigrid_smooth = scinew Task("linSolver::cg_multigridSmooth_" +  std::to_string(coarse_level->getID()),
          this, &linSolver::cg_multigrid_smooth,red_black_switch, cg_iter);

      task_multigrid_smooth->requires(Task::NewDW,d_residualLabel, Ghost::AroundCells, cg_ghost);
      task_multigrid_smooth->requires(Task::NewDW,d_bigZLabel, Ghost::AroundCells,cg_ghost);
      task_multigrid_smooth->modifies(d_bigZLabel);

      for (unsigned int i=0;i<d_precMLabel.size();i++){
        task_multigrid_smooth->requires(Task::NewDW,d_precMLabel[i], Ghost::AroundCells,cg_ghost+1); // NOT REALLY 0 ghost cells, we are taking shortcuts upstream
      }


      if (l==maxLevels-1 && red_black_switch==0){
        task_multigrid_smooth->computes(d_resSumLabel [cg_iter+1]);
        task_multigrid_smooth->computes(d_convMaxLabel[cg_iter]);
      }
      sched->addTask(task_multigrid_smooth, level_patches,matls);
    }
    

  }


  Task* task4 = scinew Task("linSolver::cg_task4",
                           this, &linSolver::cg_task4, cg_iter);

  task4->requires(Task::NewDW,d_convMaxLabel[cg_iter]);
  task4->requires(Task::NewDW,d_resSumLabel[cg_iter]);
  task4->requires(Task::NewDW,d_resSumLabel[cg_iter+1]);
  task4->modifies(d_smallPLabel);
  task4->requires(Task::NewDW,d_bigZLabel,Ghost::None, 0);

  sched->addTask(task4, fineLevel->eachPatch(),matls);
  } // END CG_iter

};


void
linSolver::testIter( const ProcessorGroup* pg,
             const PatchSubset* patches,
             const MaterialSubset* matls,
             DataWarehouse* old_dw,
             DataWarehouse* new_dw
  ){
//cout << "dummy task output::  LEVEL = " <<patches->get(0)->getLevel()->getID()  << "\n";


}



void
linSolver::cg_init1( const ProcessorGroup* pg,
                       const PatchSubset* patches,
                       const MaterialSubset* matls,
                       DataWarehouse* old_dw,
                       DataWarehouse* new_dw, const VarLabel * ALabel,const VarLabel * xLabel, const VarLabel * bLabel, const VarLabel * guessLabel){
/////////////////////////////////// TASK 1 //////////////////////////////////////
//          Compute the correction factor requires ghosts on "p"        /////////
//          correction factor requires a reduction                     //////////
/////////////////////////////////////////////////////////////////////////////////
//
   bool rk1=true;
  if(new_dw->exists(d_resSumLabel[0])){
      rk1=false;
  }
//
    int matl = d_indx ;
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    Uintah::BlockRange range(patch->getCellLowIndex(),patch->getCellHighIndex());

    CCVariable<double> x_v;      //littleq
    constCCVariable<Stencil7> A_m ;
    constCCVariable<double> g_v;  
    constCCVariable<double> b_v;  
    CCVariable<double> residual;  

    new_dw->get(A_m, ALabel,      matl , patch, Ghost::AroundCells, 1);
    new_dw->get(g_v, guessLabel,      matl , patch, Ghost::AroundCells, 1);
    new_dw->get( b_v,bLabel,      matl , patch, Ghost::None, 0);
    new_dw->allocateAndPut(x_v,xLabel, matl, patch,Ghost::AroundCells,1); // not supported long term , but padds data(avoids copy)


    if (rk1==true){
      new_dw->allocateAndPut(residual,d_residualLabel , matl, patch,Ghost::AroundCells,max(cg_ghost,1)); // need at least 1 ghost cell padding
      residual.initialize(0.0);  
    }else{
      new_dw->getModifiable(residual,d_residualLabel , matl, patch); 
    }



    Uintah::parallel_for( range,   [&](int i, int j, int k){  //compute correction, GHOST CELLS REQUIRED

               residual(i,j,k)=b_v(i,j,k)-(A_m(i,j,k)[6]*g_v(i,j,k)+
                                           A_m(i,j,k)[0]*g_v(i-1,j,k)+
                                           A_m(i,j,k)[1]*g_v(i+1,j,k)+
                                           A_m(i,j,k)[2]*g_v(i,j-1,k)+
                                           A_m(i,j,k)[3]*g_v(i,j+1,k)+
                                           A_m(i,j,k)[4]*g_v(i,j,k-1)+
                                           A_m(i,j,k)[5]*g_v(i,j,k+1));

               x_v(i,j,k)=g_v(i,j,k);
  });


    //Uintah::parallel_for( range,   [&](int i, int j, int k){  //compute correction, GHOST CELLS REQUIRED
 //cout << b_v(i,j,k) << " \n";
//});

} // end patch loop
}




void
linSolver::cg_init2( const ProcessorGroup* pg,
                       const PatchSubset* patches,
                       const MaterialSubset* matls,
                       DataWarehouse* old_dw,
                       DataWarehouse* new_dw, int iter){
/////////////////////////////////// TASK 1 //////////////////////////////////////
//          Compute the correction factor requires ghosts on "p"        /////////
//          correction factor requires a reduction                     //////////
/////////////////////////////////////////////////////////////////////////////////
//
   bool rk1=true;
  sum_vartype adjustReductionValue(0.0); 
  if(new_dw->exists(d_resSumLabel[0])){
      rk1=false;
     //new_dw->get(adjustReductionValue,d_resSumLabel);
  }
  double R_squared=0.0;
//
    int matl = d_indx ;
  for (int p = 0; p < patches->size(); p++) {
   const Patch* patch = patches->get(p);

   const IntVector idxLo = patch->getCellLowIndex();
   const IntVector idxHi = patch->getCellHighIndex();

    Uintah::BlockRange range(patch->getCellLowIndex(),patch->getCellHighIndex());

    constCCVariable<double> residual;  
    std::vector<constCCVariable<double> > BJmat( d_precMLabel.size()); 

    CCVariable<double> smallP; 


  int  total_rb_switch=30;
  int final_iter=total_rb_switch-1;
    new_dw->get(residual,d_residualLabel,matl,patch,Ghost::AroundCells,cg_ghost);

  if (rk1==true && iter ==0){
        new_dw->allocateAndPut(smallP,d_smallPLabel , matl, patch,Ghost::AroundCells,cg_ghost); // not supported long term , but padds data(avoids copy)
    smallP.initialize(0.0); 
  }else{
    new_dw->getModifiable(smallP,d_smallPLabel , matl, patch); 
    if (iter==0){
      smallP.initialize(0.0); 
    }
  }
  
    if (iter == final_iter){
    CCVariable<double> littleQ;      //littleq
    CCVariable<double> bigZ;      //littleq
      if (rk1==true ){
        new_dw->allocateAndPut(littleQ,d_littleQLabel , matl, patch,Ghost::None,0); 
        new_dw->allocateAndPut(bigZ,d_bigZLabel , matl, patch,Ghost::AroundCells,cg_ghost); // not supported long term , but padds data(avoids copy)
      }else{
        new_dw->getModifiable(littleQ,d_littleQLabel , matl, patch); 
        new_dw->getModifiable(bigZ,d_bigZLabel , matl, patch); 
      }
    bigZ.initialize(0.0);      
    littleQ.initialize(0.0);      
    }


    for (unsigned int i=0;i<d_precMLabel.size();i++){
         new_dw->get(BJmat[i],d_precMLabel[i],matl,patch,Ghost::AroundCells, cg_ghost+1); // not supported long term , but padds data(avoids copy)
      }
       
     //               A      b       x0
      red_black_relax(BJmat,residual,smallP,idxLo, idxHi,iter, patch);

if (iter==final_iter){
      Uintah::parallel_for( range,   [&](int i, int j, int k){  //compute correction, GHOST CELLS REQUIRED
         R_squared+=residual(i,j,k)*smallP(i,j,k);                                                                                                                        
        });

  

if ( rk1==true ){
  new_dw->put(sum_vartype(R_squared),d_resSumLabel[0]);
} else{
  //new_dw->put(sum_vartype(R_squared),d_resSumLabel2[0]); // need for RK
}

} // final_iter if
} // patch loop

}



void
linSolver::cg_task1( const ProcessorGroup* pg,
                       const PatchSubset* patches,
                       const MaterialSubset* matls,
                       DataWarehouse* old_dw,
                       DataWarehouse* new_dw, const VarLabel * ALabel,int iter){
/////////////////////////////////// TASK 1 //////////////////////////////////////
//          Compute the correction factor requires ghosts on "p"        /////////
//          correction factor requires a reduction                     //////////
/////////////////////////////////////////////////////////////////////////////////
  double correction_sum=0.0;
    int matl = d_indx ;
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    Uintah::BlockRange range(patch->getCellLowIndex(),patch->getCellHighIndex());
 
    constCCVariable<Stencil7> A_m ;
    constCCVariable<double> smallP;  
    CCVariable<double> littleQ;  
    new_dw->get(A_m, ALabel,      matl , patch, Ghost::None, 0);
    new_dw->get(smallP, d_smallPLabel,      matl , patch, Ghost::AroundCells, 1);
    new_dw->getModifiable(littleQ, d_littleQLabel,      matl , patch);

    Uintah::parallel_for( range,   [&](int i, int j, int k){  //compute correction, GHOST CELLS REQUIRED

         //cout << "  " << littleQ(i,j,k) <<  "  " << smallP(i,j,k) << "  " << smallP(i,j,k) << " \n";
        // NON-symmetric 
        littleQ(i,j,k)=A_m(i,j,k)[6]*smallP(i,j,k)+
        A_m(i,j,k)[0]*smallP(i-1,j,k)+
        A_m(i,j,k)[1]*smallP(i+1,j,k)+
        A_m(i,j,k)[2]*smallP(i,j-1,k)+
        A_m(i,j,k)[3]*smallP(i,j+1,k)+
        A_m(i,j,k)[4]*smallP(i,j,k-1)+
        A_m(i,j,k)[5]*smallP(i,j,k+1);


        // ASSUME SYMMETRIC
        //littleQ(i,j,k)=A_m(i,j,k)[6]*smallP(i,j,k)+  // THIS MAKES NANS because A_m is not defined in extra cells, which is required for symmetric
        //A_m(i,j,k)[0]  *smallP(i-1,j,k)+
        //A_m(i+1,j,k)[0]*smallP(i+1,j,k)+ 
        //A_m(i,j,k)[2]  *smallP(i,j-1,k)+
        //A_m(i,j+1,k)[2]*smallP(i,j+1,k)+
        //A_m(i,j,k)[4]  *smallP(i,j,k-1)+
        //A_m(i,j,k+1)[4]*smallP(i,j,k+1);
        //cout << littleQ(i,j,k) << " \n";
        correction_sum+=littleQ(i,j,k)*smallP(i,j,k);   // REDUCTION
        });
  }
  new_dw->put(sum_vartype(correction_sum),d_corrSumLabel[iter]);
  
}



void
linSolver::cg_task2( const ProcessorGroup* pg,
                       const PatchSubset* patches,
                       const MaterialSubset* matls,
                       DataWarehouse* old_dw,
                       DataWarehouse* new_dw, const VarLabel* xLabel, int iter){
/////////////////////////////////// TASK 2 ///////////////////////////////
///////   apply correction to x_v as well as the residual vector"   //////
//////////////////////////////////////////////////////////////////////////

    sum_vartype R_squared_old;
    new_dw->get(R_squared_old,d_resSumLabel[iter]); 
    sum_vartype correction_sum;
    new_dw->get(correction_sum,d_corrSumLabel[iter]);

    double correction_factor= (abs(correction_sum) < 1e-100) ? 0.0 :R_squared_old/correction_sum; // ternary may not be needed when we switch to do-while loop
    int matl = d_indx ;
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    Uintah::BlockRange range(patch->getCellLowIndex(),patch->getCellHighIndex());

    CCVariable<double> residual;  
    new_dw->getModifiable(residual, d_residualLabel,     matl , patch);
    constCCVariable<double> smallP;  
    new_dw->get(smallP, d_smallPLabel,      matl , patch, Ghost::None, 0);
    CCVariable<double> x_v;  
    new_dw->getModifiable(x_v, xLabel,      matl , patch);
    constCCVariable<double> littleQ;  
    new_dw->get(littleQ, d_littleQLabel,      matl , patch, Ghost::None, 0);

    Uintah::parallel_for( range,   [&](int i, int j, int k){  //compute correction, GHOST CELLS REQUIRED
                             x_v(i,j,k)=x_v(i,j,k)+correction_factor*smallP(i,j,k);
                             residual(i,j,k)=residual(i,j,k)-correction_factor*littleQ(i,j,k);
   //cout << littleQ(i,j,k) <<"  " <<  residual(i,j,k) << "  " <<correction_factor  << " \n";
   //cout <<residual(i,j,k) <<   "  " << littleQ(i,j,k) <<  " " << correction_factor << " unicorn\n";
                                });
}
}

// switch DW

void
linSolver::cg_task3( const ProcessorGroup* pg, // APPLY PRECONDITIONER
                       const PatchSubset* patches,
                       const MaterialSubset* matls,
                       DataWarehouse* old_dw,
                       DataWarehouse* new_dw, int iter){
  int matl = d_indx;  
  double  R_squared=0.0;
  double max_residual=0.0;
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    IntVector idxLo(patch->getCellLowIndex());
    IntVector idxHi(patch->getCellHighIndex());

  std::vector<constCCVariable<double> > BJmat(d_precMLabel.size()); 
    for (unsigned int i=0;i<d_precMLabel.size();i++){
          new_dw->get(BJmat[i], d_precMLabel[i],      matl , patch, Ghost::AroundCells,cg_ghost+1);
  }


  CCVariable<double> bigZ;
  new_dw->getModifiable(bigZ,d_bigZLabel,    matl, patch);
  bigZ.initialize(0.0);
  constCCVariable<double> residual; 
  new_dw->get(residual, d_residualLabel,      matl , patch, Ghost::AroundCells,cg_ghost);

  red_black_relax(BJmat,residual,bigZ,idxLo, idxHi,-1,patch);

         
Uintah::parallel_for( Uintah::BlockRange(idxLo , idxHi ),   [&](int i, int j, int k){  //compute correction, GHOST CELLS REQUIRED
                       R_squared+=residual(i,j,k)*bigZ(i,j,k); // reduction
                       max_residual=max(abs(residual(i,j,k)),max_residual);  // presumably most efficcient to comptue here.......could be computed earlier
                   });

}
  new_dw->put(sum_vartype(R_squared),d_resSumLabel[iter]);
  new_dw->put(max_vartype(max_residual),d_convMaxLabel[iter]);


}

void
linSolver::cg_task4( const ProcessorGroup* pg,
                       const PatchSubset* patches,
                       const MaterialSubset* matls,
                       DataWarehouse* old_dw,
                       DataWarehouse* new_dw, int iter){

/////////////////////////////////// TASK 4 ////////////////////////////////
//          apply preconditioner to residual vector           ////////////
//          requires no ghost cells, but a reduction          ///////////
//////////////////////////////////////////////////////////////////////

    sum_vartype R_squared;
    new_dw->get(R_squared,d_resSumLabel[iter+1]);
    sum_vartype R_squared_old;
    new_dw->get(R_squared_old,d_resSumLabel[iter]);

    max_vartype convergence;
    new_dw->get(convergence,d_convMaxLabel[iter]);
    proc0cout << "MAX RESIDUAL VALUE:::: " << convergence << " \n"; 

    const double  beta=R_squared/R_squared_old;  
  int matl = d_indx ;  
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    Uintah::BlockRange range(patch->getCellLowIndex(),patch->getCellHighIndex());


    CCVariable<double> smallP;  
    new_dw->getModifiable(smallP, d_smallPLabel,      matl , patch);
    constCCVariable<double> bigZ;
    new_dw->get(bigZ,d_bigZLabel,  matl, patch, Ghost::None, 0);


       Uintah::parallel_for( range,   [&](int i, int j, int k){
                             smallP(i,j,k)=bigZ(i,j,k)+beta*smallP(i,j,k);
                            });

}
}

void
linSolver::cg_moveResUp( const ProcessorGroup* pg,
                       const PatchSubset* patches,
                       const MaterialSubset* matls,
                       DataWarehouse* old_dw,
                       DataWarehouse* new_dw,int  iter){

  const Level* coarseLevel = getLevel(patches);
  const Level* fineLevel   = coarseLevel->getFinerLevel().get_rep();
  IntVector r_Ratio = fineLevel->getRefinementRatio();


  for(int p=0;p<patches->size();p++){
    const Patch* cPatch = patches->get(p);

      CCVariable<double>  coarse_res;        
      CCVariable<double>  bigZ;        
      if (iter == 0 ){
        new_dw->allocateAndPut(coarse_res,d_residualLabel , d_indx, cPatch,Ghost::AroundCells,cg_ghost); // not supported long term , but padds data(avoids copy)
        coarse_res.initialize(0.0);
      } else{
        new_dw->getModifiable(coarse_res,d_residualLabel,d_indx,cPatch,Ghost::AroundCells, cg_ghost); // not supported long term , but padds data(avoids copy)
        //new_dw->getModifiable(bigZ,d_bigZLabel,d_indx,fPatch); // not supported long term , but padds data(avoids copy)
        coarse_res.initialize(0.0);

      }

      Level::selectType finePatches;
      cPatch->getFineLevelPatches(finePatches);
                          
      for(size_t ip=0;ip<finePatches.size();ip++){  // very robust, but expensive, rewrite to optimize, but may break muultibox
        const Patch* fPatch = finePatches[ip];



        IntVector cl, ch, fl, fh;
        getFineLevelRange(cPatch, fPatch, cl, ch, fl, fh);

        if (fh.x() <= fl.x() || fh.y() <= fl.y() || fh.z() <= fl.z()) {  // WHAT IS THIS DOING AND WHY
          continue;
        }

          constCCVariable<double> fine_res;
          new_dw->getRegion(fine_res,d_residualLabel , d_indx, fineLevel, fl, fh,false);

        Uintah::parallel_for(Uintah::BlockRange(fl,fh) ,   [&](int i, int j, int k){  //compute correction, GHOST CELLS REQUIRED
            int ir=i%r_Ratio[0];
            int jr=j%r_Ratio[1];
            int kr=k%r_Ratio[2];

            int ic = (i - ir) /r_Ratio[0];
            int jc = (j - jr) /r_Ratio[1];
            int kc = (k - kr) /r_Ratio[2];

            coarse_res(ic,jc,kc)+=fine_res(i,j,k);
            });

      }
              //cout << "LEVEL: " <<coarseLevel->getID() << " \n";
if(false){   // for debugging
  const IntVector idxLo = cPatch->getCellLowIndex();
  const IntVector idxHi = cPatch->getCellHighIndex();
  for ( int k=idxLo.z(); k< idxHi.z(); k++ ){
    for ( int j=idxLo.y(); j< idxHi.y(); j++ ){
      for ( int i=idxLo.x(); i< idxHi.x(); i++ ){  // move unpadded A matrix to padded space, we do this because it simplifies downstream logic
              //int dz = idxHi.z()-idxLo.z();
              //int dy = idxHi.y()-idxLo.y();
              //int dx = idxHi.x()-idxLo.x();
             //for (int ii=1; ii<4; ii++){
               //if (A_m[ii](i,j,k) >-1e-85){
                 //A_m[ii](i,j,k) =-A_m[ii](i,j,k) ; // for clarity of output
               //}
             //}

              //for ( int k2=idxLo.z(); k2< idxHi.z(); k2++ ){
                //for ( int j2=idxLo.y(); j2< idxHi.y(); j2++ ){
                  //for ( int i2=idxLo.x(); i2< idxHi.x(); i2++ ){
                    //if ((k2*dy*dx+j2*dy+i2) ==  (k*dy*dx+j*dy+i)){
                    //cout <<  A_m[0](i,j,k) << " ";
                    //}else if ((k2*dy*dx+j2*dy+i2+1) ==  (k*dy*dx+j*dy+i)){// print iterator is 1 behind (this is the minus direction)
                      //cout << A_m[1](i,j,k)<< " " ;
                    //}else if ((k2*dy*dx+j2*dy+i2-1) ==  (k*dy*dx+j*dy+i)){ // print iterator is 1 ahead ( this is the plus direction)
                      //cout << A_m[1](i+1,j,k)<< " " ;
                    //}else if ((k2*dy*dx+(j2+1)*dy+i2) ==  (k*dy*dx+j*dy+i)){ // print iterator is 1 behind ( this is the minusdirection)
                      //cout << A_m[2](i,j,k)<< " " ;
                    //}else if ((k2*dy*dx+(j2-1)*dy+i2) ==  (k*dy*dx+j*dy+i)){ // print iterator is 1 ahead ( this is the plus direction)
                      //cout << A_m[2](i,j+1,k)<< " " ;
                    //}else if (((k2+1)*dy*dx+j2*dy+i2) ==  (k*dy*dx+j*dy+i)){ // print iterator is 1 behind ( this is the minusdirection)
                      //cout << A_m[3](i,j,k)<< " " ;
                    //}else if (((k2-1)*dy*dx+j2*dy+i2) ==  (k*dy*dx+j*dy+i)){ // print iterator is 1 ahead ( this is the plus direction)
                      //cout << A_m[3](i,j,k+1)<< " " ;
                    //}else{
                      //cout <<" " <<"0" << " "; 
                    //}
                  //}
                //}
              //}
              cout << coarse_res(i,j,k) << " ";
              cout << "\n";


          } // end i loop
        } // end j loop
      } // end k loop
              cout << "\n";
}

} // coarse patch loop
}

void
linSolver::cg_multigrid_smooth( const ProcessorGroup* pg,
                       const PatchSubset* patches,
                       const MaterialSubset* matls,
                       DataWarehouse* old_dw,
                       DataWarehouse* new_dw, int param,int iter){

  const Level* fineLevel = getLevel(patches);
  int  maxLevels=fineLevel->getGrid()->numLevels();
  double  R_squared=0.0;
  double  max_residual=0.0;
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    IntVector idxLo(patch->getCellLowIndex());
    IntVector idxHi(patch->getCellHighIndex());

    constArchVector  A_m(d_precMLabel.size());
    for (unsigned int ix=0; ix< d_precMLabel.size(); ix++) {
      new_dw->get(A_m[ix],d_precMLabel[ix] , d_indx, patch,Ghost::AroundCells,cg_ghost+1); // NOT REALLY 0 ghost cells, we are taking short cuts upstream
    }

    constCCVariable<double> b_res;
    new_dw->get(b_res,d_residualLabel , d_indx, patch,Ghost::AroundCells,cg_ghost); // not supported long term , but padds data(avoids copy)
    CCVariable<double> bigZ;
    new_dw->getModifiable(bigZ,d_bigZLabel , d_indx, patch,Ghost::AroundCells, cg_ghost );

    //int  level_iters=cg_ghost; 
    red_black_relax(A_m,b_res,bigZ,idxLo, idxHi, param , patch); 


    //int delme;
    //cin >> delme; 
    if (fineLevel->getID()==maxLevels-1 && param==0){ // only do reduction on finest level of last smoother iteration

      Uintah::parallel_for( Uintah::BlockRange(idxLo , idxHi ),   [&](int i, int j, int k){ 
          R_squared+=b_res(i,j,k)*bigZ(i,j,k); // reduction
          max_residual=max(abs(b_res(i,j,k)),max_residual);  


          });
    }
  }

    if (fineLevel->getID()==maxLevels-1 && param==0){
    new_dw->put(sum_vartype(R_squared),d_resSumLabel[iter+1]);
    new_dw->put(max_vartype(max_residual),d_convMaxLabel[iter]);
  }

 //cout << " done smoothing on level " <<  fineLevel->getID() <<  "\n";


};




void
linSolver::red_black_relax( constArchVector & precMatrix , constCCVariable<double> & residual, CCVariable<double>& bigZ,const IntVector &idxLo,const IntVector &idxHi, int rb_switch,const Patch* patch ){
       // precMatrix is the inverted precondition matrix or the A matrix.  Depending on the relaxation type. 
       //
       //
  if (d_custom_relax_type==jacobi_relax){
    int offset=cg_ghost;
    int inside_buffer=max(cg_ghost-1,0);
    for (int kk=0;kk<d_stencilWidth;kk++){
      int kr=kk-offset;
      for (int jj=0;jj<d_stencilWidth;jj++){
        int jr=jj-offset;
        for (int ii=0;ii<d_stencilWidth;ii++){
          int ir=ii-offset;

          //residual may only has 1 layer of wall cells, so we have to check for walls, but not for ghosts...........
          int wallCheckzp= kk==4? idxHi.z()-inside_buffer: idxHi.z();
          int wallCheckzm= kk==0? idxLo.z()+inside_buffer: idxLo.z();
          int wallCheckyp= jj==4? idxHi.y()-inside_buffer: idxHi.y();
          int wallCheckym= jj==0? idxLo.y()+inside_buffer: idxLo.y();
          int wallCheckxp= ii==4? idxHi.x()-inside_buffer: idxHi.x(); 
          int wallCheckxm= ii==0? idxLo.x()+inside_buffer: idxLo.x(); 

          Uintah::BlockRange rangetemp(IntVector(wallCheckxm,wallCheckym,wallCheckzm) , IntVector(wallCheckxp,wallCheckyp,wallCheckzp) );

          Uintah::parallel_for( rangetemp,   [&](int i, int j, int k){  //compute correction, GHOST CELLS REQUIRED
              bigZ(i,j,k)+= precMatrix[d_stencilWidth*d_stencilWidth*(kk)+d_stencilWidth*(jj)+ii](i,j,k)*residual(i+ir,j+jr,k+kr); /// JACOBI 3x3 BLOCK
              });
        }
      }
    }
  } else { //redBlack
              //cout << "  REDBLACK\n";
  //for ( int k=idxLo.z(); k< idxHi.z(); k++ ){
    //for ( int j=idxLo.y(); j< idxHi.y(); j++ ){
      //for ( int i=idxLo.x(); i< idxHi.x(); i++ ){  // move unpadded A matrix to padded space, we do this because it simplifies downstream logic
              //cout << bigZ(i,j,k) << " " << residual(i,j,k)  << "  " << precMatrix[0](i,j,k) << " ";
              //cout << "\n";
          //} // end i loop
        //} // end j loop
      //} // end k loop
      
        
    //if (rb_i < max_iter-1){
      //iter_on_ghosts=0; //use patch neighbor logic to set for multi-box 
    //}
    //


    int niter = cg_ghost; 
    if(rb_switch < 0 ){ // HACK DEREKX DEBUG FIX
      rb_switch=0;
        niter=3;
    }
    if ((cg_ghost%2)==0){
      rb_switch=0;
}


     //int niter=cg_ghost;

    for (int rb_i = 0 ; rb_i <niter ; rb_i++) {
  IntVector  iter_on_ghosts(max(cg_ghost-1-rb_i,0),max(cg_ghost-1-rb_i,0),max(cg_ghost-1-rb_i,0));
    Uintah::BlockRange rangedynamic(idxLo-iter_on_ghosts*patch->neighborsLow(),idxHi +iter_on_ghosts*patch->neighborsHigh() ); // assumes 1 ghost cell
      //cout << idxLo-iter_on_ghosts*patch->neighborsLow()  << "  " << idxHi+iter_on_ghosts*patch->neighborsHigh() << " \n";
      Uintah::parallel_for( rangedynamic,   [&](int i, int j, int k){  //compute correction, GHOST CELLS REQUIRED
          if ( (i + j +k + rb_i +rb_switch )% 2 ==0){    
          bigZ(i,j,k)= (residual(i,j,k) - precMatrix[1](i,j,k)*bigZ(i-1,j,k)-precMatrix[1](i+1,j,k)*bigZ(i+1,j,k)  // SYMMTRIC APPROXIMATION
                                        - precMatrix[2](i,j,k)*bigZ(i,j-1,k)-precMatrix[2](i,j+1,k)*bigZ(i,j+1,k) 
                                        - precMatrix[3](i,j,k)*bigZ(i,j,k-1)-precMatrix[3](i,j,k+1)*bigZ(i,j,k+1) ) / precMatrix[0](i,j,k) ; //red_black


          //bigZ(i,j,k)= (residual(i,j,k) - precMatrix[1](i,j,k)*bigZ(i-1,j,k)  // SYMMTRIC APPROXIMATION
                                        //- precMatrix[2](i,j,k)*bigZ(i,j-1,k) 
                                        //- precMatrix[3](i,j,k)*bigZ(i,j,k-1) ) / precMatrix[0](i,j,k) ; //red_black


          //bigZ(i,j,k)= residual(i,j,k)  ; //  no preconditioner

          //bigZ(i,j,k)= residual(i,j,k)/precMatrix[0](i,j,k) ; //  no preconditioner


          } 

          });


      //cout << "bigZ iter" << rb_i << " \n";
    //Uintah::parallel_for( BlockRange(idxLo, idxHi),   [&](int i, int j, int k){  //compute correction, GHOST CELLS REQUIRED
       //cout <<   bigZ(i,j,k)   << "  " << precMatrix[0](i,j,k) << " \n";
    //});

      //cout << "\n";
    }
if(false){   // for debugging
  //const IntVector idxLo = cPatch->getCellLowIndex();
  //const IntVector idxHi = cPatch->getCellHighIndex();
  for ( int k=idxLo.z(); k< idxHi.z(); k++ ){
    for ( int j=idxLo.y(); j< idxHi.y(); j++ ){
      for ( int i=idxLo.x(); i< idxHi.x(); i++ ){  // move unpadded A matrix to padded space, we do this because it simplifies downstream logic
              //int dz = idxHi.z()-idxLo.z();
              int dy = idxHi.y()-idxLo.y();
              int dx = idxHi.x()-idxLo.x();
             for (int ii=1; ii<4; ii++){
               //if (A_m[ii](i,j,k) >-1e-85){
                 //A_m[ii](i,j,k) =-A_m[ii](i,j,k) ; // for clarity of output
               //}
             }

              for ( int k2=idxLo.z(); k2< idxHi.z(); k2++ ){
                for ( int j2=idxLo.y(); j2< idxHi.y(); j2++ ){
                  for ( int i2=idxLo.x(); i2< idxHi.x(); i2++ ){
                    if ((k2*dy*dx+j2*dy+i2) ==  (k*dy*dx+j*dy+i)){
                    cout <<  precMatrix[0](i,j,k) << " ";
                    }else if ((k2*dy*dx+j2*dy+i2+1) ==  (k*dy*dx+j*dy+i)){// print iterator is 1 behind (this is the minus direction)
                      cout << precMatrix[1](i,j,k)<< " " ;
                    }else if ((k2*dy*dx+j2*dy+i2-1) ==  (k*dy*dx+j*dy+i)){ // print iterator is 1 ahead ( this is the plus direction)
                      cout << precMatrix[1](i+1,j,k)<< " " ;
                    }else if ((k2*dy*dx+(j2+1)*dy+i2) ==  (k*dy*dx+j*dy+i)){ // print iterator is 1 behind ( this is the minusdirection)
                      cout << precMatrix[2](i,j,k)<< " " ;
                    }else if ((k2*dy*dx+(j2-1)*dy+i2) ==  (k*dy*dx+j*dy+i)){ // print iterator is 1 ahead ( this is the plus direction)
                      cout << precMatrix[2](i,j+1,k)<< " " ;
                    }else if (((k2+1)*dy*dx+j2*dy+i2) ==  (k*dy*dx+j*dy+i)){ // print iterator is 1 behind ( this is the minusdirection)
                      cout << precMatrix[3](i,j,k)<< " " ;
                    }else if (((k2-1)*dy*dx+j2*dy+i2) ==  (k*dy*dx+j*dy+i)){ // print iterator is 1 ahead ( this is the plus direction)
                      cout << precMatrix[3](i,j,k+1)<< " " ;
                    }else{
                      cout <<" " <<"0" << " "; 
                    }
                  }
                }
              }
              cout << "\n";


          } // end i loop
        } // end j loop
      } // end k loop
}





  }
};


void
linSolver::cg_iterate( const ProcessorGroup* pg,
                             const PatchSubset* patches,
                             const MaterialSubset* matls,
                             DataWarehouse* old_dw,
                             DataWarehouse* new_dw,  
                             const VarLabel* ALabel, const VarLabel* xLabel,
                             const VarLabel* bLabel, 
                             LevelP level,  Scheduler* sched){





}










void
linSolver::coarsen_A( const ProcessorGroup* pg,
           const PatchSubset* patches,
           const MaterialSubset* matls,
           DataWarehouse* old_dw,
           DataWarehouse* new_dw
           ){
  const Level* coarseLevel = getLevel(patches);
  const Level* fineLevel   = coarseLevel->getFinerLevel().get_rep();
  IntVector r_Ratio = fineLevel->getRefinementRatio();

//cout << coarseLevel->getID() << "    <----          " << fineLevel->getID() << " \n";
  for(int p=0;p<patches->size();p++){
    const Patch* cPatch = patches->get(p);
          
      archVector  A_m(d_precMLabel.size());
    for (unsigned int ix=0; ix< d_precMLabel.size(); ix++) {
      new_dw->allocateAndPut(A_m[ix],d_precMLabel[ix] , d_indx, cPatch,Ghost::AroundCells,cg_ghost+1); // not supported long term , but padds data(avoids copy)

      if (ix ==0){
        A_m[ix].initialize(1.);
      } else {
        A_m[ix].initialize(0.0);
      }
    }

      Level::selectType finePatches;
      cPatch->getFineLevelPatches(finePatches);
                          
         int nc =0;
      for(size_t i=0;i<finePatches.size();i++){  // very robust, but expensive, its ok since we do this once durint initialization
        const Patch* fPatch = finePatches[i];

        IntVector cl, ch, fl, fh;
        getFineLevelRange(cPatch, fPatch, cl, ch, fl, fh);

        if (fh.x() <= fl.x() || fh.y() <= fl.y() || fh.z() <= fl.z()) {  // WHAT IS THIS DOING AND WHY
          continue;
        }

          constArchVector A_m_fine(d_precMLabel.size());
        for (unsigned int ix=0; ix< d_precMLabel.size(); ix++) {
          new_dw->getRegion(A_m_fine[ix],  d_precMLabel[ix], d_indx, fineLevel, fl, fh,false);
        }


        Uintah::parallel_for(Uintah::BlockRange(fl,fh) ,   [&](int i, int j, int k){  //compute correction, GHOST CELLS REQUIRED
            int ir=i%r_Ratio[0];
            int jr=j%r_Ratio[1];
            int kr=k%r_Ratio[2];

            int ic = (i - ir) /r_Ratio[0];
            int jc = (j - jr) /r_Ratio[1];
            int kc = (k - kr) /r_Ratio[2];
            A_m[0](ic,jc,kc)=0; // set all interior cells to zero for downstream sum.  they were initialized to 1 for ghost regions.
            });

        Uintah::parallel_for(Uintah::BlockRange(fl,fh) ,   [&](int i, int j, int k){  //compute correction, GHOST CELLS REQUIRED
            int ir=i%r_Ratio[0];
            int jr=j%r_Ratio[1];
            int kr=k%r_Ratio[2];

            int ic = (i - ir) /r_Ratio[0];
            int jc = (j - jr) /r_Ratio[1];
            int kc = (k - kr) /r_Ratio[2];

            A_m[0](ic,jc,kc)+=A_m_fine[0](i,j,k);

            if (ir ==0){
            A_m[1](ic,jc,kc)+=A_m_fine[1](i,j,k);
            }else{
            A_m[0](ic,jc,kc)+=A_m_fine[1](i,j,k);
             nc+=A_m_fine[1](i,j,k);
            }
           if (jr ==0){
            A_m[2](ic,jc,kc)+=A_m_fine[2](i,j,k);
            } else{
            A_m[0](ic,jc,kc)+=A_m_fine[2](i,j,k);
            }
           if (kr ==0){
            A_m[3](ic,jc,kc)+=A_m_fine[3](i,j,k);
            } else{
            A_m[0](ic,jc,kc)+=A_m_fine[3](i,j,k);
            }


            if (ir <r_Ratio[0]-1){
            A_m[0](ic,jc,kc)+=A_m_fine[1](i+1,j,k); // assume symeetry
            }
            if (jr <r_Ratio[1]-1){
            A_m[0](ic,jc,kc)+=A_m_fine[2](i,j+1,k);
            }
            if (kr <r_Ratio[2]-1){
            A_m[0](ic,jc,kc)+=A_m_fine[3](i,j,k+1); // no ghost cells needed 
            }

            });

      }
    cout << nc << " COUNTER derekx \n";


if(false){   // for debugging
  const IntVector idxLo = cPatch->getCellLowIndex();
  const IntVector idxHi = cPatch->getCellHighIndex();
  for ( int k=idxLo.z(); k< idxHi.z(); k++ ){
    for ( int j=idxLo.y(); j< idxHi.y(); j++ ){
      for ( int i=idxLo.x(); i< idxHi.x(); i++ ){  // move unpadded A matrix to padded space, we do this because it simplifies downstream logic
              //int dz = idxHi.z()-idxLo.z();
              int dy = idxHi.y()-idxLo.y();
              int dx = idxHi.x()-idxLo.x();
             for (int ii=1; ii<4; ii++){
               if (A_m[ii](i,j,k) >-1e-85){
                 A_m[ii](i,j,k) =-A_m[ii](i,j,k) ; // for clarity of output
               }
             }

              for ( int k2=idxLo.z(); k2< idxHi.z(); k2++ ){
                for ( int j2=idxLo.y(); j2< idxHi.y(); j2++ ){
                  for ( int i2=idxLo.x(); i2< idxHi.x(); i2++ ){
                    if ((k2*dy*dx+j2*dy+i2) ==  (k*dy*dx+j*dy+i)){
                    cout <<  A_m[0](i,j,k) << " ";
                    }else if ((k2*dy*dx+j2*dy+i2+1) ==  (k*dy*dx+j*dy+i)){// print iterator is 1 behind (this is the minus direction)
                      cout << A_m[1](i,j,k)<< " " ;
                    }else if ((k2*dy*dx+j2*dy+i2-1) ==  (k*dy*dx+j*dy+i)){ // print iterator is 1 ahead ( this is the plus direction)
                      cout << A_m[1](i+1,j,k)<< " " ;
                    }else if ((k2*dy*dx+(j2+1)*dy+i2) ==  (k*dy*dx+j*dy+i)){ // print iterator is 1 behind ( this is the minusdirection)
                      cout << A_m[2](i,j,k)<< " " ;
                    }else if ((k2*dy*dx+(j2-1)*dy+i2) ==  (k*dy*dx+j*dy+i)){ // print iterator is 1 ahead ( this is the plus direction)
                      cout << A_m[2](i,j+1,k)<< " " ;
                    }else if (((k2+1)*dy*dx+j2*dy+i2) ==  (k*dy*dx+j*dy+i)){ // print iterator is 1 behind ( this is the minusdirection)
                      cout << A_m[3](i,j,k)<< " " ;
                    }else if (((k2-1)*dy*dx+j2*dy+i2) ==  (k*dy*dx+j*dy+i)){ // print iterator is 1 ahead ( this is the plus direction)
                      cout << A_m[3](i,j,k+1)<< " " ;
                    }else{
                      cout <<" " <<"0" << " "; 
                    }
                  }
                }
              }
              cout << "\n";


          } // end i loop
        } // end j loop
      } // end k loop



}

} // end coarse patch loop

};




void
linSolver::cg_multigrid_down( const ProcessorGroup* pg,
                       const PatchSubset* patches,
                       const MaterialSubset* matls,
                       DataWarehouse* old_dw,
                       DataWarehouse* new_dw, int iter){
  const Level* fineLevel = getLevel(patches);
  IntVector r_Ratio = fineLevel->getRefinementRatio();
  int  maxLevels=fineLevel->getGrid()->numLevels();
   //double  R_squared=0.0;
   //double  max_residual=0.0;
  for(int p=0;p<patches->size();p++){
    const Patch* fPatch = patches->get(p);
    IntVector idxLo(fPatch->getCellLowIndex());
    IntVector idxHi(fPatch->getCellHighIndex());
          
    //constArchVector  A_m(d_precMLabel.size());
    //for (unsigned int ix=0; ix< d_precMLabel.size(); ix++) {
      //new_dw->get(A_m[ix],d_precMLabel[ix] , d_indx, fPatch,Ghost::None,0); // NOT REALLY 0 ghost cells, we are taking short cuts upstream
    //}

      //constCCVariable<double> b_res;
      //new_dw->get(b_res,d_residualLabel , d_indx, fPatch,Ghost::AroundCells,cg_ghost); // not supported long term , but padds data(avoids copy)

      CCVariable<double> bigZ;
      if (fineLevel->getID()<maxLevels-1  && iter==0 ){
        new_dw->allocateAndPut(bigZ,d_bigZLabel,  d_indx, fPatch,Ghost::AroundCells,cg_ghost+1 ); // not supported long term , but padds data(avoids copy)
      } else{
        new_dw->getModifiable(bigZ,d_bigZLabel,d_indx,fPatch); // not supported long term , but padds data(avoids copy)
      }
      bigZ.initialize(0.0);

      
      if (fineLevel->getID()>0){
        const Level* coarseLevel   = fineLevel->getCoarserLevel().get_rep();
        Level::selectType coarsePatches;
        fPatch->getCoarseLevelPatches(coarsePatches);


        for(size_t i=0;i<coarsePatches.size();i++){  // very robust, but expensive, its ok since we do this once durint initialization
          //const Patch* cPatch = coarsePatches[i];

          IntVector cl, ch, fl, fh;
          getCoarseLevelRange(fPatch,coarseLevel,cl,ch,fl,fh,IntVector(0,0,0),0,true);

          constCCVariable<double>   bigZ_coarse ;

          new_dw->getRegion(bigZ_coarse,  d_bigZLabel, d_indx, coarseLevel, cl, ch,false);

          Uintah::parallel_for(Uintah::BlockRange(fl,fh) ,   [&](int i, int j, int k){  //compute correction, GHOST CELLS REQUIRED
              //Uintah::parallel_for(Uintah::BlockRange(idxLo-cg_ghost,idxHi+cg_ghost) ,   [&](int i, int j, int k){  //compute correction, GHOST CELLS REQUIRED
              int ir=i%r_Ratio[0];
              int jr=j%r_Ratio[1];
              int kr=k%r_Ratio[2];

              int ic = (i - ir) /r_Ratio[0];
              int jc = (j - jr) /r_Ratio[1];
              int kc = (k - kr) /r_Ratio[2];


              bigZ(i,j,k)=bigZ_coarse(ic,jc,kc);

              //cout << bigZ(i,j,k) << "  " <<R_squared  << " \n";
              });

              }

              }





























          //constCCVariable<double>   bigZ_coarse;
          ////new_dw->getRegion(bigZ_coarse,  d_bigZLabel, d_indx, coarseLevel,fineLevel->mapCellToCoarser(idxLo)-1 ,fineLevel->mapCellToCoarser(idxHi-1)+1+1,false); // GET REGION AND GHOSTS not just patch overlap
          //new_dw->getRegion(bigZ_coarse,  d_bigZLabel, d_indx, coarseLevel,fineLevel->mapCellToCoarser(idxLo) ,fineLevel->mapCellToCoarser(idxHi-1)+1,false); 


          //Uintah::parallel_for(Uintah::BlockRange(idxLo,idxHi) ,   [&](int i, int j, int k){  //compute correction, GHOST CELLS REQUIRED
              //int ir=i%r_Ratio[0];
              //int jr=j%r_Ratio[1];
              //int kr=k%r_Ratio[2];

              //ir = (i <0) ? -ir : ir;
              //jr = (j <0) ? -jr : jr;
              //kr = (k <0) ? -kr : kr;

              //int ic = (i - ir) /r_Ratio[0];
              //int jc = (j - jr) /r_Ratio[1];
              //int kc = (k - kr) /r_Ratio[2];

              //bigZ(i,j,k)=bigZ_coarse(ic,jc,kc);


          //});

      //}






     //int  level_iters=cg_ghost; 
      //red_black_relax(A_m,b_res,bigZ,idxLo, idxHi, level_iters ); // Wish i could do this here, but i'm having problems using ghost cells as a work space


//if(true){   // for debugging
  //const IntVector idxLo = cPatch->getCellLowIndex();
  //const IntVector idxHi = cPatch->getCellHighIndex();
  //for ( int k=idxLo.z(); k< idxHi.z(); k++ ){
    //for ( int j=idxLo.y(); j< idxHi.y(); j++ ){
      //for ( int i=idxLo.x(); i< idxHi.x(); i++ ){  // move unpadded A matrix to padded space, we do this because it simplifies downstream logic
              //int dz = idxHi.z()-idxLo.z();
              //int dy = idxHi.y()-idxLo.y();
              //int dx = idxHi.x()-idxLo.x();
             //for (int ii=1; ii<4; ii++){
               ////if (A_m[ii](i,j,k) >-1e-85){
                 ////A_m[ii](i,j,k) =-A_m[ii](i,j,k) ; // for clarity of output
               ////}
             //}

              //for ( int k2=idxLo.z(); k2< idxHi.z(); k2++ ){
                //for ( int j2=idxLo.y(); j2< idxHi.y(); j2++ ){
                  //for ( int i2=idxLo.x(); i2< idxHi.x(); i2++ ){
                    //if ((k2*dy*dx+j2*dy+i2) ==  (k*dy*dx+j*dy+i)){
                    //cout <<  A_m[0](i,j,k) << " ";
                    //}else if ((k2*dy*dx+j2*dy+i2+1) ==  (k*dy*dx+j*dy+i)){// print iterator is 1 behind (this is the minus direction)
                      //cout << A_m[1](i,j,k)<< " " ;
                    //}else if ((k2*dy*dx+j2*dy+i2-1) ==  (k*dy*dx+j*dy+i)){ // print iterator is 1 ahead ( this is the plus direction)
                      //cout << A_m[1](i+1,j,k)<< " " ;
                    //}else if ((k2*dy*dx+(j2+1)*dy+i2) ==  (k*dy*dx+j*dy+i)){ // print iterator is 1 behind ( this is the minusdirection)
                      //cout << A_m[2](i,j,k)<< " " ;
                    //}else if ((k2*dy*dx+(j2-1)*dy+i2) ==  (k*dy*dx+j*dy+i)){ // print iterator is 1 ahead ( this is the plus direction)
                      //cout << A_m[2](i,j+1,k)<< " " ;
                    //}else if (((k2+1)*dy*dx+j2*dy+i2) ==  (k*dy*dx+j*dy+i)){ // print iterator is 1 behind ( this is the minusdirection)
                      //cout << A_m[3](i,j,k)<< " " ;
                    //}else if (((k2-1)*dy*dx+j2*dy+i2) ==  (k*dy*dx+j*dy+i)){ // print iterator is 1 ahead ( this is the plus direction)
                      //cout << A_m[3](i,j,k+1)<< " " ;
                    //}else{
                      //cout <<" " <<"0" << " "; 
                    //}
                  //}
                //}
              //}
              //cout << "\n";


          //} // end i loop
        //} // end j loop
      //} // end k loop
//}








      //Level::selectType finePatches;
      //cPatch->getFineLevelPatches(finePatches);
                          
      //for(size_t i=0;i<finePatches.size();i++){  // very robust, but expensive, its ok since we do this once durint initialization
        //const Patch* fPatch = finePatches[i];

        //IntVector cl, ch, fl, fh;
        //getFineLevelRange(cPatch, fPatch, cl, ch, fl, fh);

        //if (fh.x() <= fl.x() || fh.y() <= fl.y() || fh.z() <= fl.z()) {  // WHAT IS THIS DOING AND WHY
          //continue;
        //}

          //constArchVector A_m_fine(d_precMLabel.size());
        //for (unsigned int ix=0; ix< d_precMLabel.size(); ix++) {
          //new_dw->getRegion(A_m_fine[ix],  d_precMLabel[ix], d_indx, fineLevel, fl, fh,false);
        //}

        //Uintah::parallel_for(Uintah::BlockRange(fl,fh) ,   [&](int i, int j, int k){  //compute correction, GHOST CELLS REQUIRED
            //int ir=i%r_Ratio[0];
            //int jr=j%r_Ratio[1];
            //int kr=k%r_Ratio[2];

            //int ic = (i - ir) /r_Ratio[0];
            //int jc = (j - jr) /r_Ratio[1];
            //int kc = (k - kr) /r_Ratio[2];

            //A_m[0](ic,jc,kc)+=A_m_fine[0](i,j,k);

            //if (ir ==0){
            //A_m[1](ic,jc,kc)+=A_m_fine[1](i,j,k);
            //} else{
            //A_m[0](ic,jc,kc)+=A_m_fine[1](i,j,k);
            //}
           //if (jr ==0){
            //A_m[2](ic,jc,kc)+=A_m_fine[2](i,j,k);
            //} else{
            //A_m[0](ic,jc,kc)+=A_m_fine[2](i,j,k);
            //}
           //if (kr ==0){
            //A_m[3](ic,jc,kc)+=A_m_fine[3](i,j,k);
            //} else{
            //A_m[0](ic,jc,kc)+=A_m_fine[3](i,j,k);
            //}


            //if (ir <r_Ratio[0]-1){
            //A_m[0](ic,jc,kc)+=A_m_fine[1](i+1,j,k); // assume symeetry
            //}
            //if (jr <r_Ratio[1]-1){
            //A_m[0](ic,jc,kc)+=A_m_fine[2](i,j+1,k);
            //}
            //if (kr <r_Ratio[2]-1){
            //A_m[0](ic,jc,kc)+=A_m_fine[3](i,j,k+1); // no ghost cells needed 
            //}

            //});

      //}


      //if (fineLevel->getID()==maxLevels-1){

        //Uintah::parallel_for( Uintah::BlockRange(idxLo , idxHi ),   [&](int i, int j, int k){  //compute correction, GHOST CELLS REQUIRED
            //R_squared+=b_res(i,j,k)*bigZ(i,j,k); // reduction
            //max_residual=max(abs(b_res(i,j,k)),max_residual);  // presumably most efficcient to comptue here.......could be computed earlier
            //});
      //}

}

  //new_dw->put(sum_vartype(R_squared),d_resSumLabel);
  //new_dw->put(max_vartype(max_residual),d_convMaxLabel);


  //cout << "exiting MULTIGRID_DOWN =   " << fineLevel->getID() << " \n";

}








void
linSolver::customSolve( const ProcessorGroup* pg,
                             const PatchSubset* patches,
                             const MaterialSubset* matls,
                             DataWarehouse* old_dw,
                             DataWarehouse* new_dw,
                             const VarLabel* ALabel){ // pas level

    int matl = d_indx ;

  //double R_squared=0.0;

  sum_vartype adjustReductionValue(0.0); 

  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    Uintah::BlockRange range(patch->getCellLowIndex(),patch->getCellHighIndex());

  constCCVariable<Stencil7> A_m_unpadded;  // A-matrix
  CCVariable<Stencil7> A_m;  

  new_dw->get(A_m_unpadded, ALabel,   matl , patch,Ghost::AroundCells,1);

  const IntVector idxLo = patch->getCellLowIndex();
  const IntVector idxHi = patch->getCellHighIndex();

  const IntVector idxLo_extra = patch->getExtraCellLowIndex();
  const IntVector idxHi_extra = patch->getExtraCellHighIndex();

  archVector BJmat(d_precMLabel.size()); 
    for (unsigned int i=0;i<d_precMLabel.size();i++){
         new_dw->allocateAndPut(BJmat[i],d_precMLabel[i] , matl, patch,Ghost::AroundCells,cg_ghost+1); // +1 because of assumptions
      }



  CCVariable<double> BJm;      //block jacobi minus
  CCVariable<double> BJmm;      //block jacobi minus
  CCVariable<double> BJc;      //block jacobi central
  CCVariable<double> BJp;      //block jacobi plus
  CCVariable<double> BJpp;      //block jacobi plus


  CCVariable<double> BJm2;      //block jacobi minus
  CCVariable<double> BJmm2;      //block jacobi minus
  CCVariable<double> BJp2;      //block jacobi plus
  CCVariable<double> BJpp2;      //block jacobi plus

  CCVariable<double> BJm3;      //block jacobi minus
  CCVariable<double> BJmm3;      //block jacobi minus
  CCVariable<double> BJp3;      //block jacobi plus
  CCVariable<double> BJpp3;      //block jacobi plus

  BJc.allocate(idxLo_extra,idxHi_extra);
  BJm.allocate(idxLo_extra,idxHi_extra);
  BJp.allocate(idxLo_extra,idxHi_extra);
  BJmm.allocate(idxLo_extra,idxHi_extra);
  BJpp.allocate(idxLo_extra,idxHi_extra);

  BJm2.allocate(idxLo_extra,idxHi_extra);
  BJp2.allocate(idxLo_extra,idxHi_extra);
  BJmm2.allocate(idxLo_extra,idxHi_extra);
  BJpp2.allocate(idxLo_extra,idxHi_extra);

  BJm3.allocate(idxLo_extra,idxHi_extra);
  BJp3.allocate(idxLo_extra,idxHi_extra);
  BJmm3.allocate(idxLo_extra,idxHi_extra);
  BJpp3.allocate(idxLo_extra,idxHi_extra);


  BJc.initialize(0.); 
  BJm.initialize(0.); 
  BJp.initialize(0.); 
  BJmm.initialize(0.); 
  BJpp.initialize(0.); 

  BJm2.initialize(0.00); 
  BJp2.initialize(0.00); 
  BJmm2.initialize(0.00); 
  BJpp2.initialize(.00); 

  BJm3.initialize(0.); 
  BJp3.initialize(0.); 
  BJmm3.initialize(0.); 
  BJpp3.initialize(0.); 

  BJmat[0].initialize(1.0);  // Useful for boundary conditions in RB abstraction
   for (unsigned int i=1;i<d_precMLabel.size();i++){
     BJmat[i].initialize(0.);
   }

  A_m.allocate(idxLo_extra,idxHi_extra);

  for ( int k=idxLo_extra.z(); k< idxHi_extra.z(); k++ ){
    for ( int j=idxLo_extra.y(); j< idxHi_extra.y(); j++ ){
      for ( int i=idxLo_extra.x(); i< idxHi_extra.x(); i++ ){ // annoying stencil7, better way to initialize?
        A_m(i,j,k).initialize(0.0);
      }}}


  for ( int k=idxLo.z(); k< idxHi.z(); k++ ){
    for ( int j=idxLo.y(); j< idxHi.y(); j++ ){
      for ( int i=idxLo.x(); i< idxHi.x(); i++ ){  // move unpadded A matrix to padded space, we do this because it simplifies downstream logic

             for (int ii=0; ii<7; ii++){
        A_m(i,j,k)[ii]=A_m_unpadded(i,j,k)[ii];//no preconditione
             }
      }
    }
  }


if(false){
  for ( int k=idxLo.z(); k< idxHi.z(); k++ ){
    for ( int j=idxLo.y(); j< idxHi.y(); j++ ){
      for ( int i=idxLo.x(); i< idxHi.x(); i++ ){  // move unpadded A matrix to padded space, we do this because it simplifies downstream logic
              //int dz = idxHi.z()-idxLo.z();
              int dy = idxHi.y()-idxLo.y();
              int dx = idxHi.x()-idxLo.x();
             for (int ii=0; ii<6; ii++){
               if (A_m(i,j,k)[ii] >-1e-85){
                 A_m(i,j,k)[ii] =-A_m(i,j,k)[ii] ; // for clarity of output
               }
             }

              for ( int k2=idxLo.z(); k2< idxHi.z(); k2++ ){
                for ( int j2=idxLo.y(); j2< idxHi.y(); j2++ ){
                  for ( int i2=idxLo.x(); i2< idxHi.x(); i2++ ){
                    if ((k2*dy*dx+j2*dy+i2) ==  (k*dy*dx+j*dy+i)){
                    cout <<  A_m(i,j,k)[6] << " ";
                    }else if ((k2*dy*dx+j2*dy+i2+1) ==  (k*dy*dx+j*dy+i)){// print iterator is 1 behind (this is the minus direction)
                      cout << A_m(i,j,k)[0]<< " " ;
                    }else if ((k2*dy*dx+j2*dy+i2-1) ==  (k*dy*dx+j*dy+i)){ // print iterator is 1 ahead ( this is the plus direction)
                      cout << A_m(i,j,k)[1]<< " " ;
                    }else if ((k2*dy*dx+(j2+1)*dy+i2) ==  (k*dy*dx+j*dy+i)){ // print iterator is 1 behind ( this is the minusdirection)
                      cout << A_m(i,j,k)[2]<< " " ;
                    }else if ((k2*dy*dx+(j2-1)*dy+i2) ==  (k*dy*dx+j*dy+i)){ // print iterator is 1 ahead ( this is the plus direction)
                      cout << A_m(i,j,k)[3]<< " " ;
                    }else if (((k2+1)*dy*dx+j2*dy+i2) ==  (k*dy*dx+j*dy+i)){ // print iterator is 1 behind ( this is the minusdirection)
                      cout << A_m(i,j,k)[4]<< " " ;
                    }else if (((k2-1)*dy*dx+j2*dy+i2) ==  (k*dy*dx+j*dy+i)){ // print iterator is 1 ahead ( this is the plus direction)
                      cout << A_m(i,j,k)[5]<< " " ;
                    }else{
                      cout <<" " <<"0" << " "; 
                    }
                  }
                }
              }
              cout << "\n";


          } // end i loop
        } // end j loop
      } // end k loop
}










// XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX  X X X X X X
  //for ( int k=idxLo.z(); k< idxHi.z(); k++ ){
    //for ( int j=idxLo.y(); j< idxHi.y(); j++ ){
      //for ( int i=idxLo.x(); i< idxHi.x(); i++ ){  // move unpadded A matrix to padded space, we do this because it simplifies downstream logic

            //if( i >idxLo.x() &&  i <idxHi.x()-1)
            //{
               //double ips,jps,kps;
               //double ims,jms,kms;

               //if(i>0) {                       // NEEDS EDGE OF DOMAIN LOGIC TO WORK FOR L-domains
                 //ips=1;  jps=0;  kps=0;
               //}else{
                 //if(j>0) {                       
                   //ips=0;  jps=1;  kps=0;
                 //}else{
                   //ips=0;  jps=0;  kps=1;
                 //}
               //}

               //if(i<(idxHi.x()-1)) {                       // NEEDS EDGE OF DOMAIN LOGIC TO WORK FOR L-domains
                 //ims=1;  jms=0;  kms=0;
               //}else{
                 //if(j>(idxHi.y()-1)) {                       
                   //ims=0;  jms=1;  kms=0;
                 //}else{
                   //ims=0;  jms=0;  kms=1;
                 //}
               //}

          //double abde =   A_m(i-ims,j-jms,k-kms)[6]*A_m(i    ,j    ,k    )[6]-A_m(i  ,j,k)[0]*A_m(i-ims,j-jms,k-kms)[1]; // 3 3
          //double abgh =   A_m(i-ims,j-jms,k-kms)[6]*A_m(i+ips,j+jps,k+kps)[0];                               // 2 3
          //double degh =   A_m(i    ,j    ,k    )[0]*A_m(i+ips,j+jps,k+kps)[0];                               // 1 3
         
          //double acdf =   A_m(i-ims,j-jms,k-kms)[6]*A_m(i    ,j    ,k    )[1];                                 // 3 2
          //double acgi =   A_m(i-ims,j-jms,k-kms)[6]*A_m(i+ips,j+jps,k+kps)[6];                                 // 2 2
          //double dfgi =   A_m(i    ,j   ,k    )[0]*A_m(i+ips,j+jps,k+kps)[6];                                 // 1 2
         
          //double bcef =   A_m(i-ims,j-jms,k-kms)[1]*A_m(i  ,j,k)[1];                                 // 3 1
          //double bchi =   A_m(i-ims,j-jms,k-kms)[1]*A_m(i+ips,j+jps,k+kps)[6];                                 // 2 1
          //double efhi =   A_m(i    ,j    ,k    )[6]*A_m(i+ips,j+jps,k+kps)[6]-A_m(i+ips,j+jps,k+kps)[0]*A_m(i    ,j    ,k    )[1]; // 1 1
   
          //double detA=  A_m(i-ims,j-jms,k-kms)[6] *efhi  -  A_m(i-ims,j-jms,k-kms)[1]*dfgi;

          //BJc(i-ims,j-jms,k-kms)+=efhi/detA;  BJp(i-ims,j-jms,k-kms)+=-dfgi/detA;   BJpp(i-ims,j-jms,k-kms)+=degh/detA;
          //BJm(i,j,k)+=-bchi/detA;             BJc(i,j,k)+=acgi/detA;               BJp(i,j,k)+=-abgh/detA;
          //BJmm(i+ips,j+jps,k+kps)+=bcef/detA; BJm(i+ips,j+jps,k+kps)+=-acdf/detA; BJc(i+ips,j+jps,k+kps)+=abde/detA;
//}
//}}}
 ////// Y     YYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYYY
  //for ( int k=idxLo.z(); k< idxHi.z(); k++ ){
    //for ( int j=idxLo.y(); j< idxHi.y(); j++ ){
      //for ( int i=idxLo.x(); i< idxHi.x(); i++ ){  // move unpadded A matrix to padded space, we do this because it simplifies downstream logic
            //if( j >idxLo.y() &&  j <idxHi.y()-1)
            //{

               //double ips,jps,kps;
               //double ims,jms,kms;

               //if(j>0) {                       // NEEDS EDGE OF DOMAIN LOGIC TO WORK FOR L-domains
                   //ips=0;  jps=1;  kps=0;
               //}else{
                 //if(i>0) {                       
                 //ips=1;  jps=0;  kps=0;
                 //}else{
                   //ips=0;  jps=0;  kps=1;
                 //}
               //}

               //if(j<(idxHi.y()-1)) {                       // NEEDS EDGE OF DOMAIN LOGIC TO WORK FOR L-domains
                   //ims=0;  jms=1;  kms=0;
               //}else{
                 //if(i>(idxHi.x()-1)) {                       
                 //ims=1;  jms=0;  kms=0;
                 //}else{
                   //ims=0;  jms=0;  kms=1;
                 //}
               //}
              

          //double abde =   A_m(i-ims,j-jms,k-kms)[6]*A_m(i    ,j    ,k    )[6]-A_m(i  ,j,k)[2]*A_m(i-ims,j-jms,k-kms)[3]; // 3 3
          //double abgh =   A_m(i-ims,j-jms,k-kms)[6]*A_m(i+ips,j+jps,k+kps)[2];                               // 2 3
          //double degh =   A_m(i    ,j    ,k    )[2]*A_m(i+ips,j+jps,k+kps)[2];                               // 1 3
         
          //double acdf =   A_m(i-ims,j-jms,k-kms)[6]*A_m(i    ,j    ,k    )[3];                                 // 3 2
          //double acgi =   A_m(i-ims,j-jms,k-kms)[6]*A_m(i+ips,j+jps,k+kps)[6];                                 // 2 2
          //double dfgi =   A_m(i    ,j   ,k    )[2]*A_m(i+ips,j+jps,k+kps)[6];                                 // 1 2
         
         
         
          //double bcef =   A_m(i-ims,j-jms,k-kms)[3]*A_m(i  ,j,k)[3];                                 // 3 1
          //double bchi =   A_m(i-ims,j-jms,k-kms)[3]*A_m(i+ips,j+jps,k+kps)[6];                                 // 2 1
          //double efhi =   A_m(i    ,j    ,k    )[6]*A_m(i+ips,j+jps,k+kps)[6]-A_m(i+ips,j+jps,k+kps)[2]*A_m(i    ,j    ,k    )[3]; // 1 1
   
          //double detA=  A_m(i-ims,j-jms,k-kms)[6] *efhi  -  A_m(i-ims,j-jms,k-kms)[3]*dfgi;

          //BJc(i-ims,j-jms,k-kms)+=efhi/detA;  BJp2(i-ims,j-jms,k-kms)+=-dfgi/detA; BJpp2(i-ims,j-jms,k-kms)+=degh/detA;
          //BJm2(i,j,k)+=-bchi/detA;            BJc(i,j,k)+=acgi/detA;              BJp2(i,j,k)+=-abgh/detA;
          //BJmm2(i+ips,j+jps,k+kps)+=bcef/detA; BJm2(i+ips,j+jps,k+kps)+=-acdf/detA; BJc(i+ips,j+jps,k+kps)+=abde/detA;
 
//}
//}}}




      if(d_custom_relax_type==redBlack){
            Uintah::parallel_for( range,   [&](int i, int j, int k){  //compute correction, GHOST CELLS REQUIRED
                BJmat[0](i,j,k)=A_m(i,j,k)[6];//jacobi preconditioner
                BJmat[1](i,j,k)=A_m(i,j,k)[0];//jacobi preconditioner
                BJmat[2](i,j,k)=A_m(i,j,k)[2];//jacobi preconditioner
                BJmat[3](i,j,k)=A_m(i,j,k)[4];//jacobi preconditioner
                    /////////// For nonsymetric /////////////
                //BJmat[4](i,j,k)=1.0/A_m(i,j,k)[1];//jacobi preconditioner
                //BJmat[5](i,j,k)=1.0/A_m(i,j,k)[3];//jacobi preconditioner
                //BJmat[6](i,j,k)=1.0/A_m(i,j,k)[5];//jacobi preconditioner
                });
     }else{
//////////////////compute 125 point stencil
      enum Preconditioners{ jacobi, jacobiBlock};
     
     int precondType;
     if (d_blockSize > 1){
       precondType=jacobiBlock;
     }else{
       precondType=jacobi;
     }

        int offset =cg_ghost;
          if(precondType==jacobi){
            Uintah::parallel_for( range,   [&](int i, int j, int k){  //compute correction, GHOST CELLS REQUIRED
                BJmat[d_stencilWidth*d_stencilWidth*offset+d_stencilWidth*offset+offset](i,j,k)=1.0/A_m(i,j,k)[6];//jacobi preconditioner
                });

          }else if(precondType==jacobiBlock){
            Uintah::parallel_for( range,   [&](int i, int j, int k){  //compute correction, GHOST CELLS REQUIRED

            //if((i*dy*dz+j*dz+k) % d_blockSize==1 && (k*dz*dy+i*dy+j) % d_blockSize==1   && (k*dy*dx+j*dx+i) % d_blockSize==1  )
            //ghost_ d_stencilWidth 
              

              // This assumes that there is always 1Layer of ghost cells or extra cells for patch boundaries!
            int inner_buffer=cg_ghost-1;  
            //if( k >idxLo.z()  && j>idxLo.y()  && i >idxLo.x()  &&  k <idxHi.z()-1 && j<idxHi.y()-1  &&  i<idxHi.x()-1)
            if( k >=idxLo.z()+inner_buffer  && j>=idxLo.y()+inner_buffer  && i >=idxLo.x()+inner_buffer  &&  k <idxHi.z()-inner_buffer && j<idxHi.y()-inner_buffer  &&  i<idxHi.x()-inner_buffer)
            {

            DenseMatrix* dfdrh = scinew DenseMatrix(d_blockSize*d_blockSize*d_blockSize,d_blockSize*d_blockSize*d_blockSize);
            dfdrh->zero();// [-]

            for (int kc=0;kc<d_blockSize;kc++){ // construct A-matrix
            int km=kc-1;
            int kp=kc+1;
            int kk=kc-offset/2;
            for (int jc=0;jc<d_blockSize;jc++){
            int jm=jc-1;
            int jp=jc+1;
            int jj=jc-offset/2;
            for (int ic=0;ic<d_blockSize;ic++){
            int im=ic-1;
            int ip=ic+1;
            int ii=ic-offset/2;
            (*dfdrh)[kc*d_blockSize*d_blockSize+jc*d_blockSize+ic][kc*d_blockSize*d_blockSize+jc*d_blockSize+ic]   =A_m(i+ii,j+jj,k+kk)[6] ;
            if (ic<(d_blockSize-1)) {
              (*dfdrh)[kc*d_blockSize*d_blockSize+jc*d_blockSize+ic][kc*d_blockSize*d_blockSize+jc*d_blockSize+ip] =A_m(i+ii,j+jj,k+kk)[1] ;
            }
            if (jc<(d_blockSize-1)) {
              (*dfdrh)[kc*d_blockSize*d_blockSize+jc*d_blockSize+ic][kc*d_blockSize*d_blockSize+jp*d_blockSize+ic] =A_m(i+ii,j+jj,k+kk)[3] ;
            }
            if (kc<(d_blockSize-1)) {
              (*dfdrh)[kc*d_blockSize*d_blockSize+jc*d_blockSize+ic][kp*d_blockSize*d_blockSize+jc*d_blockSize+ic] =A_m(i+ii,j+jj,k+kk)[5] ;
            }
            if (ic>0) {
              (*dfdrh)[kc*d_blockSize*d_blockSize+jc*d_blockSize+ic][kc*d_blockSize*d_blockSize+jc*d_blockSize+im] =A_m(i+ii,j+jj,k+kk)[0] ;
            }if (jc>0) {
              (*dfdrh)[kc*d_blockSize*d_blockSize+jc*d_blockSize+ic][kc*d_blockSize*d_blockSize+jm*d_blockSize+ic] =A_m(i+ii,j+jj,k+kk)[2] ;
            }if (kc>0) {
              (*dfdrh)[kc*d_blockSize*d_blockSize+jc*d_blockSize+ic][km*d_blockSize*d_blockSize+jc*d_blockSize+ic] =A_m(i+ii,j+jj,k+kk)[4] ;
            }
            }}}



            dfdrh->invert(); // simple matrix inversion for a dense matrix.
            for (int kc=0;kc<d_blockSize;kc++){
              for (int jc=0;jc<d_blockSize;jc++){
                for (int ic=0;ic<d_blockSize;ic++){

                  for (int kk=0;kk<d_blockSize;kk++){
                    int kr=kk-kc;
                    for (int jj=0;jj<d_blockSize;jj++){
                      int jr=jj-jc;
                      for (int ii=0;ii<d_blockSize;ii++){
                       int ir=ii-ic;
                        //BJmat[d_stencilWidth*d_stencilWidth*(kr+offset)+d_stencilWidth*(jr+offset)+ir+offset](i-1+ic,j-1+jc,k-1+kc)=(*dfdrh)[kc*d_blockSize*d_blockSize+jc*d_blockSize+ic][kk*d_blockSize*d_blockSize+jj*d_blockSize+ii]; // WRONG BUG CONVERGES FASTER?
                        BJmat[d_stencilWidth*d_stencilWidth*(kr+offset)+d_stencilWidth*(jr+offset)+ir+offset](i-1+ic,j-1+jc,k-1+kc)+=(*dfdrh)[kc*d_blockSize*d_blockSize+jc*d_blockSize+ic][kk*d_blockSize*d_blockSize+jj*d_blockSize+ii];
                        //if (kr==0 && jr==0 && ir==0){
                        //BJmat[d_stencilWidth*d_stencilWidth*(kr+offset)+d_stencilWidth*(jr+offset)+ir+offset](i,j,k)=b_v(i,j,k)/A_m(i,j,k)[6];
                        //}
                      } // matrix k
                    } // matrix j
                  } // matrix i
                } //steniil k
              } // stencil j
            } // stencil i
          } // End if (for boundaries)
      }); //parallel_for
   } //  jacobi preconditionerlogic
 } // all preconditioner logic
}// end pathc loop






}






}

