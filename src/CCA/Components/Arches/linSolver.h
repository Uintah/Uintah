#ifndef Uintah_Components_Arches_linSolver_h
#define Uintah_Components_Arches_linSolver_h

#include <Core/Grid/MaterialManagerP.h>
#include <CCA/Components/Arches/ArchesVariables.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
//#include <Core/Grid/LevelP.h>
#include <Core/Grid/Task.h>

namespace Uintah {

class ArchesLabel;  //Spaghetti anyone?
class MPMArchesLabel;
class ArchesLabel;
class PressureSolver;
//class ProcessorGroup;
//class ArchesVariables;
class PhysicalConstants;
//class Discretization;
//class Source;
class BoundaryCondition;


class linSolver {


typedef std::vector< CCVariable <double > >     archVector ;
typedef std::vector< constCCVariable <double > >     constArchVector ;




public:

  //______________________________________________________________________/
  // Construct an instance of the Pressure solver.
  linSolver(const MPMArchesLabel* mal,   PressureSolver* ps,  BoundaryCondition* bc , PhysicalConstants* pc,const int indx,const VarLabel*  VolFrac,
                                                                                                                           const VarLabel*  CellType,
                                                                                                                           const VarLabel*  A_m,
                                                                                                                           const MaterialSet* f_matls
                                                                                                                                                      ) : 
d_MAlab(mal),
d_pressureSolver(ps),
d_boundaryCondition(bc),
d_physicalConsts(pc),
d_indx(indx),
d_mmgasVolFracLabel(VolFrac),
d_cellTypeLabel(CellType),
d_presCoefPBLMLabel(A_m),
d_matls(f_matls) 
{
}

  //______________________________________________________________________/
  // Destructor
  ~linSolver(){
     // TO DO:
     // delete all  varLabels created by class here!!!!!!
}

 
  void 
  sched_PreconditionerConstruction(SchedulerP& sched, const MaterialSet* matls,const LevelP& level );
 
  void
  sched_buildAMatrix(SchedulerP& sched,
                  const PatchSet* patches,
                  const MaterialSet* matls);
  void 
  buildAMatrix(const ProcessorGroup* pc,
                    const PatchSubset* patches,
                    const MaterialSubset* matls,
                    DataWarehouse* new_dw,
                    DataWarehouse* matrix_dw
                    );






void sched_customSolve(SchedulerP& sched, const MaterialSet* matls,
                                   const PatchSet* patches,
               const VarLabel*     A,     Task::WhichDW  A_dw,
               const VarLabel*     x,     Task::WhichDW   modifies_x,
               const VarLabel*     b,     Task::WhichDW   b_dw  ,
               const VarLabel*     guess, Task::WhichDW    guess_dw, int rkstep, LevelP level );


void printError( const ProcessorGroup* pg,
                             const PatchSubset* patches,
                             const MaterialSubset* matls,
                             DataWarehouse* old_dw,
                             DataWarehouse* new_dw,
                             const VarLabel* ALabel,
                             const VarLabel* xLabel,
                             const VarLabel* bLabel);

           
void red_black_relax( constArchVector& A_matrix ,constCCVariable<double>& b_array,  CCVariable<double>& x_guess,const IntVector &idxLo ,const IntVector &idxHi,int  niter,const  Patch * patch);
                             


void
customSolve( const ProcessorGroup* pg,
             const PatchSubset* patches,
             const MaterialSubset* matls,
             DataWarehouse* old_dw,
             DataWarehouse* new_dw,
             const VarLabel* ALabel
             );


void
fillGhosts( const ProcessorGroup* pg,
             const PatchSubset* patches,
             const MaterialSubset* matls,
             DataWarehouse* old_dw,
             DataWarehouse* new_dw
             );


void
coarsen_A( const ProcessorGroup* pg,
           const PatchSubset* patches,
           const MaterialSubset* matls,
           DataWarehouse* old_dw,
           DataWarehouse* new_dw
           );



void
Update_preconditioner( const ProcessorGroup* pg,
             const PatchSubset* patches,
             const MaterialSubset* matls,
             DataWarehouse* old_dw,
             DataWarehouse* new_dw
             );

void
cg_multigrid_smooth( const ProcessorGroup* pg,
                       const PatchSubset* patches,
                       const MaterialSubset* matls,
                       DataWarehouse* old_dw,
                       DataWarehouse* new_dw, int param, int iter);



void
cg_iterate( const ProcessorGroup* pg,
                             const PatchSubset* patches,
                             const MaterialSubset* matls,
                             DataWarehouse* old_dw,
                             DataWarehouse* new_dw, const VarLabel* ALabel, const VarLabel* xLabel, const VarLabel* bLabel, LevelP level, Scheduler* sched );



void
testIter( const ProcessorGroup* pg,
             const PatchSubset* patches,
             const MaterialSubset* matls,
             DataWarehouse* old_dw,
             DataWarehouse* new_dw
             );








 /// conjugate gradient objects
 std::vector<const VarLabel *> d_precMLabel; 
 const VarLabel * d_residualLabel;                   /// residual of CG algorithm  (R^2)
 const VarLabel * d_bigZLabel;                       /// z varaiable in conjugate gradient algorithm
 const VarLabel * d_littleQLabel;                    /// q varaible in conjugate gradient algorithm
 const VarLabel * d_smallPLabel;                     /// p variable in conjugate gradient algorithm
 const VarLabel * d_paddedALabel;                    /// A-matrix with padding (in extra cells only)

std::vector< const VarLabel *> d_corrSumLabel;                    /// reduction computing correction factor
std::vector< const VarLabel *> d_convMaxLabel;                    /// reduction checking for convergence1
std::vector< const VarLabel *> d_resSumLabel ;                     /// reduction computing sum of residuals

 int d_blockSize{1};             // size of jacobi block
 int d_stencilWidth{1}  ;        // stencilWidth  of jacobi block
 int cg_ghost{0};        // number of ghosts required by the jacoian preconditioner
 enum uintah_linear_solve_relaxType{ redBlack, jacobi_relax};
 int d_custom_relax_type;
 int cg_n_iter;
 

 SchedulerP cg_subsched;                    // create cg_subscheduler n

void
cg_moveResUp( const ProcessorGroup* pg,
                       const PatchSubset* patches,
                       const MaterialSubset* matls,
                       DataWarehouse* old_dw,
                       DataWarehouse* new_dw, int iter);
                                          

void
cg_init1( const ProcessorGroup* pg,
          const PatchSubset* patches,
          const MaterialSubset* matls,
          DataWarehouse* old_dw,
          DataWarehouse* new_dw, const VarLabel * ALabel,  const VarLabel * xLabel,  const VarLabel * bLabel, const VarLabel * guessLabel);

void
cg_init2( const ProcessorGroup* pg,
          const PatchSubset* patches,
          const MaterialSubset* matls,
          DataWarehouse* old_dw,
          DataWarehouse* new_dw, int iter);



void
cg_task1( const ProcessorGroup* pg,
          const PatchSubset* patches,
          const MaterialSubset* matls,
          DataWarehouse* old_dw,
          DataWarehouse* new_dw, const VarLabel * ALabel, int iter);

void
cg_task2( const ProcessorGroup* pg,
          const PatchSubset* patches,
          const MaterialSubset* matls,
          DataWarehouse* old_dw,
          DataWarehouse* new_dw, const VarLabel* xLabel, int iter);
void
cg_multigrid_down( const ProcessorGroup* pg,
              const PatchSubset* patches,
              const MaterialSubset* matls,
              DataWarehouse* old_dw,
              DataWarehouse* new_dw, int iter);


void
cg_task3( const ProcessorGroup* pg,
          const PatchSubset* patches,
          const MaterialSubset* matls,
          DataWarehouse* old_dw,
          DataWarehouse* new_dw, int iter);


void
cg_task4( const ProcessorGroup* pg,
          const PatchSubset* patches,
          const MaterialSubset* matls,
          DataWarehouse* old_dw,
          DataWarehouse* new_dw, int iter);

  const MPMArchesLabel* d_MAlab;
  PressureSolver*     d_pressureSolver;
  BoundaryCondition*  d_boundaryCondition;
  PhysicalConstants*  d_physicalConsts;

  int d_indx;             // Arches matl index.
  const VarLabel* d_mmgasVolFracLabel;
  const VarLabel* d_cellTypeLabel;
  const VarLabel* d_presCoefPBLMLabel;
  const MaterialSet* d_matls;




};



}
#endif
