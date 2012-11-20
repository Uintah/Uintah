#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/SimulationState.h>
#include <CCA/Ports/Scheduler.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Components/Arches/ChemMix/MixingRxnModel.h>

using namespace std;
using namespace Uintah;

EqnBase::EqnBase(ArchesLabel* fieldLabels, ExplicitTimeInt* timeIntegrator, string eqnName):
d_fieldLabels(fieldLabels), d_timeIntegrator(timeIntegrator), d_eqnName(eqnName),
d_doClipping(false), d_doLowClip(false), d_doHighClip(false), d_lowClip(-999999), d_highClip(-999999), d_smallClip(-999999),
b_stepUsesCellLocation(false), b_stepUsesPhysicalLocation(false),
d_constant_init(0.0), d_step_dir("x"), d_step_start(0.0), d_step_end(0.0), d_step_cellstart(0), d_step_cellend(0), d_step_value(0.0), 
d_use_constant_D(false)
{
  d_boundaryCond = scinew BoundaryCondition_new( d_fieldLabels ); 
  d_disc = scinew Discretization_new(); 
  _using_new_intrusion = false; 
	_table_init = false; 
}

EqnBase::~EqnBase()
{
  delete(d_boundaryCond);
  delete(d_disc);
}

void
EqnBase::extraProblemSetup( ProblemSpecP& db ){ 

  d_boundaryCond->setupTabulatedBC( db, d_eqnName, _table );

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
        const BoundCond<double> *new_bcs_d = dynamic_cast<const BoundCond<double> *>(bc); 
        const BoundCond<std::string> *new_bcs_st = dynamic_cast<const BoundCond<std::string> *>(bc);
        bool failed = false; 

        if ( new_bcs_d == 0 ){ 
          failed = true; 
          //check string type
          if ( new_bcs_st != 0 ){ 
            failed = false; 
          }
        }

        if (failed){
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

        // delete bc?; FIXME

      }
    }
  }
}

void
EqnBase::sched_tableInitialization( const LevelP& level, SchedulerP& sched )
{

	std::string taskname = "EqnBase::tableInitialization";
  Task* tsk = scinew Task(taskname, this, &EqnBase::tableInitialization); 

	MixingRxnModel::VarMap ivVarMap = _table->getIVVars();

  // independent variables :: these must have been computed previously 
  for ( MixingRxnModel::VarMap::iterator i = ivVarMap.begin(); i != ivVarMap.end(); ++i ) {

    tsk->requires( Task::NewDW, i->second, Ghost::None, 0 ); 

  }

	// for inert mixing
	MixingRxnModel::InertMasterMap inertMap = _table->getInertMap(); 
  for ( MixingRxnModel::InertMasterMap::iterator iter = inertMap.begin(); iter != inertMap.end(); iter++ ){ 
    const VarLabel* label = VarLabel::find( iter->first ); 
    tsk->requires( Task::NewDW, label, Ghost::None, 0 ); 
  } 

	tsk->modifies( d_transportVarLabel ); 

  sched->addTask( tsk, level->eachPatch(), d_fieldLabels->d_sharedState->allArchesMaterials() ); 


}

void 
EqnBase::tableInitialization(const ProcessorGroup* pc, 
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

    //independent variables:
    std::vector<constCCVariable<double> > indep_storage; 
		MixingRxnModel::VarMap ivVarMap = _table->getIVVars();
    std::vector<string> allIndepVarNames = _table->getAllIndepVars(); 

    for ( int i = 0; i < (int) allIndepVarNames.size(); i++ ){

      MixingRxnModel::VarMap::iterator ivar = ivVarMap.find( allIndepVarNames[i] ); 

      constCCVariable<double> the_var; 
      new_dw->get( the_var, ivar->second, matlIndex, patch, Ghost::None, 0 );
      indep_storage.push_back( the_var ); 

    }

		MixingRxnModel::InertMasterMap inertMap = _table->getInertMap(); 
		MixingRxnModel::StringToCCVar inert_mixture_fractions; 
		inert_mixture_fractions.clear(); 
    for ( MixingRxnModel::InertMasterMap::iterator iter = inertMap.begin(); iter != inertMap.end(); iter++ ){ 
      const VarLabel* label = VarLabel::find( iter->first ); 
      constCCVariable<double> variable; 
      new_dw->get( variable, label, matlIndex, patch, Ghost::None, 0 ); 
			MixingRxnModel::ConstVarContainer container; 
      container.var = variable; 

      inert_mixture_fractions.insert( std::make_pair( iter->first, container) ); 

    } 

		CCVariable<double> eqn_var; 
		new_dw->getModifiable( eqn_var, d_transportVarLabel, matlIndex, patch ); 

    for (CellIterator iter=patch->getCellIterator(); !iter.done(); iter++){

			IntVector c = *iter; 

			std::vector<double> iv;
			for (std::vector<constCCVariable<double> >::iterator iv_iter = indep_storage.begin(); 
					iv_iter != indep_storage.end(); iv_iter++ ){ 

				iv.push_back( (*iv_iter)[c] ); 

			}

			eqn_var[c] = _table->getTableValue( iv, d_init_dp_varname, inert_mixture_fractions, c ); 

		}

    //recompute the BCs
    computeBCsSpecial( patch, d_eqnName, eqn_var );

	}
}
