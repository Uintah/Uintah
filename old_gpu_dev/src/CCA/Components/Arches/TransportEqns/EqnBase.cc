#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/SimulationState.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>

using namespace std;
using namespace Uintah;

EqnBase::EqnBase(ArchesLabel* fieldLabels, ExplicitTimeInt* timeIntegrator, string eqnName):
d_fieldLabels(fieldLabels), d_timeIntegrator(timeIntegrator), d_eqnName(eqnName),
d_doClipping(false), d_doLowClip(false), d_doHighClip(false), d_lowClip(-999999), d_highClip(-999999), d_smallClip(-999999),
b_stepUsesCellLocation(false), b_stepUsesPhysicalLocation(false),
d_constant_init(0.0), d_step_dir("x"), d_step_start(0.0), d_step_end(0.0), d_step_cellstart(0), d_step_cellend(0), d_step_value(0.0)
{
  d_boundaryCond = scinew BoundaryCondition_new( d_fieldLabels ); 
  d_disc = scinew Discretization_new(); 
  _using_new_intrusion = false; 
}

EqnBase::~EqnBase()
{
  delete(d_boundaryCond);
  delete(d_disc);
}

void 
EqnBase::sched_checkBCs( const LevelP& level, SchedulerP& sched )
{
  string taskname = "EqnBase::checkBCs"; 
  Task* tsk = scinew Task(taskname, this, &EqnBase::checkBCs); 

  sched->addTask( tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allArchesMaterials() ); 
}

void 
EqnBase::checkBCs( const ProcessorGroup* pc, 
                 const PatchSubset* patches, 
                 const MaterialSubset* matls, 
                 DataWarehouse* old_dw, 
                 DataWarehouse* new_dw )
{

  //patch loop
  for (int p=0; p < patches->size(); p++){

    const Patch* patch = patches->get(p);
    int archIndex = 0;
    int matlIndex = d_fieldLabels->d_sharedState->getArchesMaterial(archIndex)->getDWIndex(); 

    vector<Patch::FaceType> bf;
    vector<Patch::FaceType>::const_iterator bf_iter;
    patch->getBoundaryFaces(bf);
    // Loop over all boundary faces on this patch
    for (bf_iter = bf.begin(); bf_iter != bf.end(); bf_iter++){
      Patch::FaceType face = *bf_iter; 

      int numChildren = patch->getBCDataArray(face)->getNumberChildren(matlIndex);
      for (int child = 0; child < numChildren; child++){

        string bc_kind = "NotSet"; 
        Iterator bound_ptr; 
        Iterator nu; //not used...who knows why?
        const BoundCondBase* bc = patch->getArrayBCValues( face, matlIndex, 
                                                           d_eqnName, bound_ptr, 
                                                           nu, child ); 
        const BoundCond<double> *new_bcs = dynamic_cast<const BoundCond<double> *>(bc); 
        if (new_bcs != 0) 
          bc_kind = new_bcs->getBCType__NEW(); 
        else {
          string whichface; 
          if (face == 0)
            whichface = "x-";
          else if (face == 1)
            whichface = "x+"; 
          else if (face == 2) 
            whichface = "y-";
          else if (face == 3)
            whichface = "y+";
          else if (face == 4)
            whichface = "z-";
          else if (face == 5)
            whichface = "z+";

          cout << "ERROR!:  Missing boundary condition specification!" << endl;
          cout << "Here are the details:" << endl;
          cout << "Variable = " << d_eqnName << endl;
          cout << "Face = " << whichface << endl; 
          cout << "Child = " << child << endl;
          cout << "Material = " << matlIndex << endl;
          throw ProblemSetupException("Please correct your <BoundaryCondition> section in your input file for this variable", __FILE__,__LINE__); 
        }

        delete bc; 

      }
    }
  }
}
