#include <Core/Grid/BoundaryConditions/BCDataArray.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Grid/SimulationState.h>
#include <CCA/Ports/SchedulerP.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/Task.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/BoundaryCond_new.h>

//===========================================================================

using namespace std;
using namespace Uintah;
using namespace SCIRun; 

BoundaryCondition_new::BoundaryCondition_new(const ArchesLabel* fieldLabels):
d_fieldLabels(fieldLabels)
{} 

BoundaryCondition_new::~BoundaryCondition_new()
{}
//---------------------------------------------------------------------------
// Method: Problem Setup 
//---------------------------------------------------------------------------
void BoundaryCondition_new::problemSetup()
{}
//---------------------------------------------------------------------------
// Method: Schedule the assignment of boundary conditions 
//---------------------------------------------------------------------------
int BoundaryCondition_new::scheduleSetBC( const LevelP& level,
                                  SchedulerP& sched )
{
  string taskname = "BoundaryCondition_new::setBC"; 
  Task* tsk = scinew Task(taskname, this, &BoundaryCondition_new::setBC);

  sched->addTask(tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allArchesMaterials());

}
//---------------------------------------------------------------------------
// Method: Actually set the boundary condition 
//---------------------------------------------------------------------------
void BoundaryCondition_new::setBC( const ProcessorGroup* pc,
                          const PatchSubset* patches,
                          const MaterialSubset*,
                          DataWarehouse* old_dw,
                          DataWarehouse* new_dw)
{
  //patch loop 
  for (int p = 0; p < patches->size(); p++) {

    const Patch* patch = patches->get(p);
    int matlIndex = 0; //d_fieldLabels->d_sharedState->getArchesMaterial(0)->getDWIndex(); 

    //string varname = "temperature";
    //setScalarValueBC( pc, patch, temperature, varname );
   
  }
}
//---------------------------------------------------------------------------
// Method: Set Scalar BC values 
//---------------------------------------------------------------------------
void BoundaryCondition_new::setScalarValueBC( const ProcessorGroup*,
                                     const Patch* patch,
                                     CCVariable<double>& scalar, 
                                     string varname )
{
  // This method sets the value of the scalar in the boundary cell
  // so that the boundary condition set in the input file is satisfied. 
  vector<Patch::FaceType>::const_iterator iter;
  vector<Patch::FaceType> bf;
  patch->getBoundaryFaces(bf);

  //hard coded to zero for now (Arches only has one) 
  int mat_id = 0; 

  for (iter = bf.begin(); iter !=bf.end(); iter++){
    Patch::FaceType face = *iter;

    //get the face direction
    IntVector insideCellDir = patch->faceDirection(face);
    //get the number of children
    int numChildren = patch->getBCDataArray(face)->getNumberChildren(mat_id); //assumed one material

    for (int child = 0; child < numChildren; child++){
      double bc_value = -9; 
      string bc_kind = "NotSet";
      Iterator bound_ptr;

      bool foundIterator = 
        getIteratorBCValueBCKind( patch, face, child, varname, mat_id, bc_value, bound_ptr, bc_kind); 

      if (foundIterator) {
        // --- notation --- 
        // bp1: boundary cell + 1 or the interior cell one in from the boundary
        if (bc_kind == "Dirichlet") {
          switch (face) {
          case Patch::xminus:
            for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
              IntVector bp1(*bound_ptr - insideCellDir); 
              scalar[*bound_ptr] = 2.0*bc_value - scalar[bp1];
            }
          break;
          case Patch::xplus:
            for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
              IntVector bp1(*bound_ptr - insideCellDir);
              scalar[*bound_ptr] = 2.0*bc_value - scalar[bp1];
            }
          break;
#ifdef YDIM
          case Patch::yminus:
            for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
              IntVector bp1(*bound_ptr - insideCellDir);
              scalar[*bound_ptr] = 2.0*bc_value - scalar[bp1];
            }
          break;
          case Patch::yplus:
            for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
              IntVector bp1(*bound_ptr - insideCellDir);
              scalar[*bound_ptr] = 2.0*bc_value - scalar[bp1];
            }
          break;
#endif
#ifdef ZDIM
          case Patch::zminus:
            for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
              IntVector bp1(*bound_ptr - insideCellDir);
              scalar[*bound_ptr] = 2.0*bc_value - scalar[bp1];
            }
          break;
          case Patch::zplus:
            for (bound_ptr.reset(); !bound_ptr.done(); bound_ptr++) {
              IntVector bp1(*bound_ptr - insideCellDir);
              scalar[*bound_ptr] = 2.0*bc_value - scalar[bp1];
            }
          break;
#endif
          }

        } else if (bc_kind == "Neumann") {

        }
      }
    }
  }
}

 

