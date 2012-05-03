
#include <CCA/Components/Arches/OperatorSplitChem.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermFactory.h>
#include <CCA/Components/Arches/SourceTerms/SourceTermBase.h>
#include <CCA/Components/Arches/TransportEqns/EqnFactory.h>
#include <CCA/Components/Arches/TransportEqns/EqnBase.h>
#include <CCA/Ports/Scheduler.h>
#include <Core/Grid/Task.h>

using namespace Uintah; 

//--------------------------------------------------/
OperatorSplitChem::OperatorSplitChem( ArchesLabel* field_labels ) :
	_field_labels(field_labels)
{
	_do_split_chem = false; 
}

//--------------------------------------------------/
OperatorSplitChem::~OperatorSplitChem(){}

//--------------------------------------------------/
void
OperatorSplitChem::problemSetup( const ProblemSpecP& params ){

	ProblemSpecP db = params; 

	if ( db->findBlock("OperatorSplitChem") ){ 

		_do_split_chem = true; 

		ProblemSpecP db_split = db->findBlock("OperatorSplitChem"); 

		db_split->require("N", _N); 
		
		for ( ProblemSpecP db_eqn = db_split->findBlock("eqn"); db_eqn != 0; db_eqn = db_eqn->findNextBlock("eqn") ){ 

			std::string eqn; 
			std::string src; 

			db_eqn->getAttribute("eqn", eqn);
			db_eqn->getAttribute("src", src); 

			_eqns.insert(make_pair(eqn, src));

		} 
	} 


} 

//--------------------------------------------------/
void 
OperatorSplitChem::sched_integrate( const LevelP& level, SchedulerP& sched )
{

	if (_do_split_chem) { 

		sched_eval( level, sched ); 


	} 
}

//--------------------------------------------------/
void 
OperatorSplitChem::sched_eval( const LevelP& level, SchedulerP& sched )
{ 

  Task* tsk = scinew Task("OperatorSplitChem::eval", this, &OperatorSplitChem::eval);

  //SourceTermFactory& src_factory = SourceTermFactory::self();
  //EqnFactory&        eqn_factory = EqnFactory::self();

	sched->addTask( tsk, level->eachPatch(), _field_labels->d_sharedState->allArchesMaterials() ); 

} 

//--------------------------------------------------/
void 
OperatorSplitChem::eval( const ProcessorGroup* pc,
                         const PatchSubset* patches,
                				 const MaterialSubset* matls,
                				 DataWarehouse* old_dw,
                				 DataWarehouse* new_dw )
{



}
