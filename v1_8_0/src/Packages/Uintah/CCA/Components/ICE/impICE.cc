#include "ICE.h"
#include <Packages/Uintah/CCA/Components/ICE/ICEMaterial.h>
#include <Packages/Uintah/CCA/Components/ICE/BoundaryCond.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/MPMMaterial.h>
#include <Packages/Uintah/CCA/Components/MPM/ConstitutiveModel/ConstitutiveModel.h>
#include <Packages/Uintah/Core/Grid/Task.h>
#include <Packages/Uintah/CCA/Ports/Scheduler.h>
#include <Core/Util/DebugStream.h>
#include <Packages/Uintah/Core/Grid/SimulationState.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/Exceptions/InvalidValue.h> 
#include <Packages/Uintah/Core/Grid/Stencil7.h>


using namespace SCIRun;
using namespace Uintah;

static DebugStream cout_norm("ICE_NORMAL_COUT", false);  
static DebugStream cout_doing("ICE_DOING_COUT", false);


/* ---------------------------------------------------------------------
 Function~  ICE::scheduleImplicitPressureSolve--
_____________________________________________________________________*/
void ICE::scheduleImplicitPressureSolve(  SchedulerP& sched,
                                          const LevelP& level,
                                          SolverInterface* solver, 
                                          const SolverParameters* solver_parameters,
                                          const PatchSet* patches,
                                          const MaterialSubset* press_matl,
                                          const MaterialSubset* ice_matls,
                                          const MaterialSubset* mpm_matls,
                                          const MaterialSet* all_matls)
{
  Task* t;

  Ghost::GhostType  gac = Ghost::AroundCells;  
  Ghost::GhostType  gn  = Ghost::None; 
  //__________________________________
  //  Form the matrix
  cout_doing << "ICE::scheduleSetupMatrix" << endl;
  t = scinew Task("setupMatrix", this, &ICE::setupMatrix);
  t->requires( Task::OldDW, lb->delTLabel);
  t->requires( Task::NewDW, lb->vol_frac_CCLabel,   gac,1);
  t->requires( Task::NewDW, lb->uvel_FCMELabel,     gac,2);
  t->requires( Task::NewDW, lb->vvel_FCMELabel,     gac,2);
  t->requires( Task::NewDW, lb->wvel_FCMELabel,     gac,2);
  t->requires( Task::NewDW, lb->rho_CCLabel,        gn);
  
  t->computes(lb->matrixLabel);
  t->computes(lb->rhsLabel);
  t->computes(lb->initialGuessLabel);
  sched->addTask(t, patches, all_matls);

  //__________________________________
  //  solve for delP
  cout_doing << "ICE::scheduleSolve" << endl;
  solver->scheduleSolve(level, sched, all_matls,
                        lb->matrixLabel, lb->imp_delPLabel, false,
                        lb->rhsLabel,    lb->initialGuessLabel,
                        Task::NewDW,     solver_parameters);
                        
  //__________________________________
  // update the pressure
  cout_doing << "ICE::scheduleUpdatePressure" << endl;
  t = scinew Task("updatePressure", this, &ICE::updatePressure);
  t->requires(Task::NewDW, lb->imp_delPLabel,        press_matl,  gac, 1); 
  t->requires(Task::NewDW, lb->press_equil_CCLabel,  press_matl,  gn);  
  t->requires(Task::NewDW, lb->sp_vol_CCLabel,                    gn);
  t->requires(Task::NewDW, lb->rho_CCLabel,                      gn);
    
  t->computes(lb->delP_DilatateLabel, press_matl);
  t->computes(lb->delP_MassXLabel,    press_matl);  
  t->computes(lb->press_CCLabel,      press_matl); 
  t->computes(lb->sum_rho_CCLabel,    press_matl);  // only one mat subset     
      
  sched->addTask(t, patches, all_matls);
                       
}

/* --------------------------------------------------------------------- 
 Function~  ICE::setupMatrix-- 
_____________________________________________________________________*/
void ICE::setupMatrix(const ProcessorGroup*,
                      const PatchSubset* patches,
                      const MaterialSubset* ,
                      DataWarehouse* old_dw,
                      DataWarehouse* new_dw)
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    cout_doing<<"Doing setupMatrix on patch "
              << patch->getID() <<"\t\t\t ICE" << endl;
    
    int numMatls  = d_sharedState->getNumMatls();
    delt_vartype delT;
    old_dw->get(delT, d_sharedState->get_delt_label());
    Vector dx     = patch->dCell();
    double vol    = dx.x()*dx.y()*dx.z();
    double invvol = 1./vol;

    Advector* advector = d_advector->clone(new_dw,patch);
    CCVariable<double> q_CC, q_advected;                     
    CCVariable<double> rhs, initialGuess;
    CCVariable<Stencil7> matrix; 
    
    const IntVector gc(1,1,1);
    Ghost::GhostType  gn  = Ghost::None;
    Ghost::GhostType  gac = Ghost::AroundCells;
    new_dw->allocateAndPut(matrix,       lb->matrixLabel,       0, patch);              
    new_dw->allocateAndPut(rhs,          lb->rhsLabel,          0, patch);   
    new_dw->allocateAndPut(initialGuess, lb->initialGuessLabel, 0, patch); 
        
    for(int m = 0; m < numMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      int indx = matl->getDWIndex();
      constSFCXVariable<double> uvel_FC;
      constSFCYVariable<double> vvel_FC;
      constSFCZVariable<double> wvel_FC;
      constCCVariable<double> vol_frac;
      constCCVariable<double> rho_CC;      

      new_dw->get(uvel_FC, lb->uvel_FCMELabel,         indx,patch,gac, 2);   
      new_dw->get(vvel_FC, lb->vvel_FCMELabel,         indx,patch,gac, 2);   
      new_dw->get(wvel_FC, lb->wvel_FCMELabel,         indx,patch,gac, 2);   
      new_dw->get(vol_frac,lb->vol_frac_CCLabel,       indx,patch,gac, 1);   
      new_dw->get(rho_CC,      lb->rho_CCLabel,        indx,patch,gn,0);  

#if 0          
      //__________________________________
      // Advection preprocessing
      advector->inFluxOutFluxVolume(uvel_FC,vvel_FC,wvel_FC,delT,patch,indx); 

      for(CellIterator iter = patch->getCellIterator(gc); !iter.done();iter++){
       IntVector c = *iter;
        q_CC[c] = vol_frac[c] * invvol;
      }
      //__________________________________
      //   First order advection of q_CC 
      advector->advectQ(q_CC, patch, q_advected);
#endif
    }  //matl loop
//    delete advector;  
    
    cout << "setup matrix: done with advection" <<endl;     
    //__________________________________
    //  setup up the matrix an rhs
    // DUMMY matrix for right now
    for(CellIterator iter(patch->getCellIterator()); !iter.done(); iter++){
      IntVector c = *iter;
      matrix[c].p = 1.0;
      matrix[c].n = 1.0;
      matrix[c].s = 1.0;
      matrix[c].e = 1.0;
      matrix[c].w = 1.0;
      matrix[c].t = 1.0;
      matrix[c].b = 1.0;
      rhs[c]      = 1.0;
     initialGuess[c] = 0.0;
    }
    cout << "setup Matrix: done setting up matrix"<<endl;          
  }
}

/* --------------------------------------------------------------------- 
 Function~  ICE::updatePressure-- 
_____________________________________________________________________*/
void ICE::updatePressure(const ProcessorGroup*,
                         const PatchSubset* patches,                    
                         const MaterialSubset* ,                        
                         DataWarehouse* old_dw,                         
                         DataWarehouse* new_dw)                         
{
  for(int p=0;p<patches->size();p++){
    const Patch* patch = patches->get(p);
    cout_doing<<"Doing updatePressure on patch "
              << patch->getID() <<"\t\t\t ICE" << endl;
   
    int numMatls  = d_sharedState->getNumMatls(); 
      
    CCVariable<double> press_CC;
    CCVariable<double> delP_Dilatate;
    CCVariable<double> delP_MassX;
    CCVariable<double> sum_rho_CC;        
    constCCVariable<double> imp_delP;
    constCCVariable<double> pressure;
    constCCVariable<double> rho_CC;
    StaticArray<constCCVariable<double> > sp_vol_CC(numMatls);
    
    Ghost::GhostType  gn = Ghost::None;
    new_dw->get(imp_delP,      lb->imp_delPLabel,        0,patch,gn,0);
    new_dw->get(pressure,      lb->press_equil_CCLabel,  0,patch,gn,0);
    
    new_dw->allocateAndPut(delP_Dilatate,lb->delP_DilatateLabel,0, patch);
    new_dw->allocateAndPut(delP_MassX,   lb->delP_MassXLabel,   0, patch);
    new_dw->allocateAndPut(press_CC,     lb->press_CCLabel,     0, patch);
    new_dw->allocateAndPut(sum_rho_CC,   lb->sum_rho_CCLabel,   0, patch);

    for(int m = 0; m < numMatls; m++) {
      Material* matl = d_sharedState->getMaterial( m );
      int indx = matl->getDWIndex();
      new_dw->get(sp_vol_CC[m],lb->sp_vol_CCLabel, indx,patch,gn,0);
      new_dw->get(rho_CC,      lb->rho_CCLabel,    indx,patch,gn,0);
      //__________________________________
      //  compute sum_rho_CC used by press_FC
      for(CellIterator iter=patch->getExtraCellIterator(); !iter.done();iter++) {
        IntVector c = *iter;
        sum_rho_CC[c] += rho_CC[c];
      } 
    }    
/*`==========TESTING==========*/
    delP_Dilatate.initialize(0.0);
    delP_MassX.initialize(0.0); 
/*==========TESTING==========`*/
    
    //__________________________________
    //  add delP to press
    for(CellIterator iter = patch->getCellIterator(); !iter.done(); iter++) { 
      IntVector c = *iter;
      press_CC[c] = pressure[c];
    }

    setBC(press_CC, sp_vol_CC[SURROUND_MAT],
          "sp_vol", "Pressure", patch ,d_sharedState, 0, new_dw);
  }
}





//______________________________________________________________________
//      P E T S C   A T T E M P T
#if 0
//______________________________________________________________________
//
void ICE::petscExample(const PatchSubset* patches)
{
  Vec         x, b, exact_sol;      /* approx solution, RHS, exact solution */
  Mat         A;                    /* linear system matrix */
  SLES        sles;                 /* linear solver context */
  PC          pc;                   /* preconditioner context */
  KSP         ksp;                  /* Krylov subspace method context */
  PetscReal   norm;                 /* norm of solution error */
  int         ierr,i,n = 10,col[3],its,size;
  PetscScalar neg_one = -1.0,one = 1.0,value[3];

  int argc = 1;
  char** argv;
  argv = new char*[argc];
  argv[0] = "petscExample";
  
  static int n_passes = 0; 

/*`==========TESTING==========*/
  int numlrows;
  int numlcolumns;
  int globalrows;
  int globalcolumns;
  petscMapping(patches, numlrows,  numlcolumns, globalrows,  globalcolumns);  
  numlrows      = 10;
  numlcolumns   = 10;
  globalrows    = 10;
  globalcolumns = 10;         
/*==========TESTING==========`*/
  //__________________________________
  //  Only do it once
  if (n_passes == 0 ) {
    PetscInitialize(&argc,&argv, PETSC_NULL, PETSC_NULL);
    cout << "I've run PetscInitialize"<< endl;
    n_passes ++;

    ierr = MPI_Comm_size(PETSC_COMM_WORLD, &size);               CHKERRQ(ierr);
    if (size != 1) {
      SETERRQ(1,"This is a uniprocessor example only!");
    }
  }
  ierr = PetscOptionsGetInt(PETSC_NULL,"-n",&n,PETSC_NULL);   CHKERRQ(ierr);
  cout << "I've run PetscOptionsGetInt"<< endl;
  

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
         Compute the matrix and right-hand-side vector that define
         the linear system, Ax = b.
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */

  /* 
     Create vectors.  Note that we form 1 vector from scratch and
     then duplicate as needed.
  */
/*`==========TESTING==========*/

  ierr = VecCreateMPI(PETSC_COMM_WORLD,numlrows, globalrows,&x); 

  //ierr = VecCreate(PETSC_COMM_WORLD,&x);CHKERRQ(ierr);
/*==========TESTING==========`*/
  ierr = VecSetSizes(x,PETSC_DECIDE,n); CHKERRQ(ierr);
  ierr = VecSetFromOptions(x);          CHKERRQ(ierr);
  ierr = VecDuplicate(x,&b);            CHKERRQ(ierr);
  ierr = VecDuplicate(x,&exact_sol);    CHKERRQ(ierr);

  /* 
     Create matrix.  When using MatCreate(), the matrix format can
     be specified at runtime.

     Performance tuning note:  For problems of substantial size,
     preallocation of matrix memory is crucial for attaining good 
     performance.  Since preallocation is not possible via the generic
     matrix creation routine MatCreate(), we recommend for practical 
     problems instead to use the creation routine for a particular matrix
     format, e.g.,
         MatCreateSeqAIJ() - sequential AIJ (compressed sparse row)
         MatCreateSeqBAIJ() - block AIJ
     See the matrix chapter of the users manual for details.
  */    
  
/*`==========TESTING==========*/
#if 0
    ierr = MatCreateMPIAIJ(PETSC_COMM_WORLD, 
                    numlrows, numlcolumns, globalrows, globalcolumns, 
                    PETSC_DEFAULT, PETSC_NULL, PETSC_DEFAULT, PETSC_NULL, &A);     CHKERRQ(ierr);
#endif
                     
    ierr = MatCreate(PETSC_COMM_WORLD,PETSC_DECIDE,PETSC_DECIDE,n,n,&A);    CHKERRQ(ierr);
/*==========TESTING==========`*/

  

  ierr = MatSetFromOptions(A);    CHKERRQ(ierr);
  cout << "I've created the matrices and vector"<< endl;
  /* 
     Assemble matrix
  */
  
  // petsc matrix
#if 0
  IntVector lowIndex   = patch->getCellLowIndex();
  IntVector highIndex  = patch->getCellHighIndex();

  Array3<int> l2g(lowIndex, highIndex);
  l2g.copy(d_petscLocalToGlobal[patch]);  
#endif
//__________________________________
// Form the matrix
/*`==========TESTING==========*/
#if 0
  PetscScalar stencil_mat[3][3];
  int  val = 10;
  for (int i=0; i<3; i++) {
    for (int j =0; j< 3; j++) {
      stencil_mat[i][j] = (PetscScalar)val++;
    }
  }
  int row = 1;
  int column = 1;
  ierr = MatSetValuesBlocked(A,3,&row,3,&column,&stencil_mat[0][0],INSERT_VALUES);   CHKERRQ(ierr); 
#endif
/*==========TESTING==========`*/
#if 1
  value[0] = -1.0; 
  value[1] =  2.0; 
  value[2] = -1.0;
  
  for (i=1; i<n-1; i++) {
    col[0] = i-1; 
    col[1] = i; 
    col[2] = i+1;
    ierr = MatSetValues(A,1,&i,3,col,value,INSERT_VALUES);    CHKERRQ(ierr);
  }
  i = n - 1; 
  col[0] = n - 2; 
  col[1] = n - 1;
  ierr = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);      CHKERRQ(ierr);
  i      = 0; 
  col[0] = 0; 
  col[1] = 1; 
  value[0] = 2.0; 
  value[1] = -1.0;
   
  ierr = MatSetValues(A,1,&i,2,col,value,INSERT_VALUES);      CHKERRQ(ierr);
  
 #endif
 
 //__________________________________
 //  Assemble the matrix
  ierr = MatAssemblyBegin(A,MAT_FINAL_ASSEMBLY);              CHKERRQ(ierr);
  ierr = MatAssemblyEnd(  A,MAT_FINAL_ASSEMBLY);              CHKERRQ(ierr);
  
  
  ierr = MatView(A,PETSC_VIEWER_STDOUT_SELF );               CHKERRQ(ierr);
  /* 
     Set exact solution; then compute right-hand-side vector.
  */
  ierr = VecSet(&one, exact_sol);      CHKERRQ(ierr);
  ierr = MatMult(A, exact_sol, b);     CHKERRQ(ierr);


  cout << "I've Assembled the matrix and vector"<< endl;
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                Create the linear solver and set various options
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /* 
     Create linear solver context
  */
  ierr = SLESCreate(PETSC_COMM_WORLD,&sles);                    CHKERRQ(ierr);

  /* 
     Set operators. Here the matrix that defines the linear system
     also serves as the preconditioning matrix.
  */
  ierr = SLESSetOperators(sles,A,A,DIFFERENT_NONZERO_PATTERN);  CHKERRQ(ierr);

  /* 
     Set linear solver defaults for this problem (optional).
     - By extracting the KSP and PC contexts from the SLES context,
       we can then directly call any KSP and PC routines to set
       various options.
     - The following four statements are optional; all of these
       parameters could alternatively be specified at runtime via
       SLESSetFromOptions();
  */
  ierr = SLESGetKSP(sles,&ksp);           CHKERRQ(ierr);
  ierr = SLESGetPC( sles,&pc);            CHKERRQ(ierr);
  ierr = PCSetType( pc,PCJACOBI);         CHKERRQ(ierr);
  ierr = KSPSetTolerances(ksp,1.e-7,PETSC_DEFAULT,PETSC_DEFAULT,PETSC_DEFAULT);CHKERRQ(ierr);

  /* 
    Set runtime options, e.g.,
        -ksp_type <type> -pc_type <type> -ksp_monitor -ksp_rtol <rtol>
    These options will override those specified above as long as
    SLESSetFromOptions() is called _after_ any other customization
    routines.
  */
  ierr = SLESSetFromOptions(sles);    CHKERRQ(ierr);
 
  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                      Solve the linear system
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  ierr = SLESSolve(sles, b, x, &its);    CHKERRQ(ierr); 

  /* 
     View solver info; we could instead use the option -sles_view to
     print this info to the screen at the conclusion of SLESSolve().
  */
  ierr = SLESView(sles,PETSC_VIEWER_STDOUT_WORLD);    CHKERRQ(ierr);

  /* - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
                      Check solution and clean up
     - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - */
  /* 
     Check the error
  */
  ierr = VecAXPY(&neg_one, exact_sol, x);     CHKERRQ(ierr);
  ierr = VecNorm(x, NORM_2, &norm);           CHKERRQ(ierr);
  ierr = PetscPrintf(PETSC_COMM_WORLD,"Norm of error %A, Iterations %d\n",norm,its);CHKERRQ(ierr);
 
   cout << "I've checked the solution "<< endl;
  /* 
     Free work space.  All PETSc objects should be destroyed when they
     are no longer needed.
  */
  ierr = VecDestroy(x);               CHKERRQ(ierr);
  ierr = VecDestroy(exact_sol);       CHKERRQ(ierr);
  ierr = VecDestroy(b);               CHKERRQ(ierr);
  ierr = MatDestroy(A);               CHKERRQ(ierr);
  ierr = SLESDestroy(sles);           CHKERRQ(ierr);
   cout << "I've destroyed all the machinery "<< endl;
  /*
     Always call PetscFinalize() before exiting a program.  This routine
       - finalizes the PETSc libraries as well as MPI
       - provides summary and diagnostic information if certain runtime
         options are chosen (e.g., -log_summary).
  */
#if 0  
  ierr = PetscFinalize();   CHKERRQ(ierr);
  cout << "I've finalized Petsc "<< endl;
#endif
} 



//______________________________________________________________________
//    Generate UCF -> Petsc mapping
void ICE::petscMapping( const PatchSubset* patches,
                        int numlrows,
                        int numlcolumns,
                        int globalrows,
                        int globalcolumns)

{
  int numProcessors = d_myworld->size();
  vector<int> numCells(numProcessors, 0);
  vector<int> startIndex(numProcessors);
  int totalCells = 0;

  //__________________________________
  //  Find total number of cells on a processor (CPU)
  //  - loop through all the patches that are on 
  //    a particular processor and count up the cells (numCells)
  //  - set the startIndex for each patch in petscGlobal index
  for (int CPU = 0; CPU < d_perproc_patches->size(); CPU++) {
    startIndex[CPU] = totalCells;
    int cpu_total_cells = 0;
    cout << " I'm now working on cpu " << CPU << endl;
    const PatchSubset* patchsub = d_perproc_patches->getSubset(CPU);
    for (int ps = 0; ps<patchsub->size(); ps++) {
      const Patch* patch   = patchsub->get(ps);
      IntVector lowIndex  = patch->getCellLowIndex();
      IntVector highIndex = patch->getCellHighIndex();

      long nCells = (highIndex.x()-lowIndex.x())*
                    (highIndex.z()-lowIndex.y())*
                    (highIndex.z()-lowIndex.z());

      d_petscGlobalStart[patch]=totalCells;
      totalCells      +=nCells;
      cpu_total_cells +=nCells;
    }
    numCells[CPU] = cpu_total_cells;
    cout << " numcells "<< numCells[CPU] << " CPU "<< CPU << endl;
  }
  //__________________________________
  //  figure out the local to global index mapping
  //  - find the low and high index for that patch with one extra layer of cells
  //  - for that index range find all the main patch and all it's neighboring patches
  //  - Loop over all the neighboring patches and find the 
  for(int p=0;p<patches->size();p++){
    const Patch* patch=patches->get(p);
    
    IntVector patch_lowIndex  = patch->getCellLowIndex();
    IntVector patch_highIndex = patch->getCellHighIndex();
    cout << "patch extents = " << patch_lowIndex << " " << patch_highIndex << endl;
    Array3<int> l2g(patch_lowIndex, patch_highIndex);
    l2g.initialize(-1234);
    long totalCells=0;
    
    //__________________________________
    // find the neighbors of this patch
    const Level* level = patch->getLevel();
    Level::selectType neighbors;
    level->selectPatches(patch_lowIndex, patch_highIndex, neighbors);
    
    //__________________________________
    //  loop over all the neighbors and ???
    for(int i=0;i<neighbors.size();i++){
      const Patch* neighbor = neighbors[i];
      
      IntVector neighbor_lowIndex  = neighbor->getCellLowIndex();
      IntVector neighbor_highIndex = neighbor->getCellHighIndex();
      cout << "neighbor extents = " << neighbor_lowIndex << " " << neighbor_highIndex << endl;
      IntVector low  = Max(patch_lowIndex,  neighbor_lowIndex);
      IntVector high = Min(patch_highIndex, neighbor_highIndex);
      
      if( (    high.x() < low.x() ) 
          || ( high.y() < low.y() ) 
          || ( high.z() < low.z() ) )
        throw InternalError("Patch doesn't overlap?");
      
      int petscglobalIndex = d_petscGlobalStart[neighbor];
      IntVector dcells = neighbor_highIndex - neighbor_lowIndex;
      IntVector start  = low - neighbor_lowIndex;
      petscglobalIndex +=  start.z()*dcells.x()*dcells.y()
                         + start.y()*dcells.x()
                         + start.x();
#if 0
      cout << "Looking at patch:    " << patch->getID() << '\n';
      cout << "Looking at neighbor: " << neighbor->getID() << '\n';
      cout << "low=          " << low << '\n';
      cout << "high=         " << high << '\n';
      cout << "dcells=       " << dcells << '\n';
      cout << "start at:     " << d_petscGlobalStart[neighbor] << '\n';
      cout << "globalIndex = " << petscglobalIndex << '\n';
#endif
      //__________________________________
      // loop over all cells in 
      for (int colZ = low.z(); colZ < high.z(); colZ ++) {
        int idx_slab = petscglobalIndex;
        cout << "idx_slab = " << idx_slab << "\n";
        petscglobalIndex += dcells.x()*dcells.y();
        cout << "petscglobalIndex = " << petscglobalIndex << "\n";
        
        for (int colY = low.y(); colY < high.y(); colY ++) {
          int idx = idx_slab;
          idx_slab += dcells.x();
          for (int colX = low.x(); colX < high.x(); colX ++) {
            l2g[IntVector(colX, colY, colZ)] = idx++;
          }  // colX loop
        }  // colY loop
      }  // colZ loop
      IntVector d = high-low;
      totalCells+=d.x()*d.y()*d.z();
    }  //neighbor loop
    
    d_petscLocalToGlobal[patch].copyPointer(l2g);
    
#if 0
      cout<<"\n\n"<<endl; 
      IntVector l = l2g.getWindow()->getLowIndex();
      IntVector h = l2g.getWindow()->getHighIndex();
      for(int z=l.z();z<h.z();z++){
        for(int y=l.y();y<h.y();y++){
          for(int x=l.x();x<h.x();x++){
            IntVector idx(x,y,z);
            cout << "l2g" << idx << " \t\t=" << l2g[idx] << '\n';
          }
        }
      }
#endif

  } // patches loop

  int me = d_myworld->myrank();
  numlrows = numCells[me];
  numlcolumns   = numlrows;
  globalrows    = (int)totalCells;
  globalcolumns = (int)totalCells;
#if 0
  cout << " numlrows " << numlrows << " numlcolumns " << numlcolumns << " globalrows " <<
          globalrows << " globalcolumns " << globalcolumns << endl;
#endif
}

#endif
