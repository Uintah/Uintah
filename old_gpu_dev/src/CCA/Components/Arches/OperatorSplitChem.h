#ifndef OperatorSplitChem_h
#define OperatorSplitChem_h

#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/SimulationState.h>
#include <Core/Grid/Variables/VarTypes.h>
#include <Core/Grid/Level.h>
#include <CCA/Ports/SimulationInterface.h>
#include <CCA/Components/Arches/ArchesLabel.h>
#include <CCA/Ports/DataWarehouse.h>
#include <CCA/Components/Arches/Directives.h>

#include <map>
#include <string>

//==========================================================================

/**
* @class Operator Split Chemistry 
* @author Jeremy Thornock
* @date July 2011
*
* @brief Will solve grid resolved chemistry in an operator split mode. 
*
* Input file should contain the following information: 
*
* <OperatorSplitChem  spec="OPTIONAL NODATA">
* 	<N 								spec="REQUIRED INTEGER 'positive'"> <!-- dt_split = dt_cfd / N --> 
* 	<!-- equations are listed in order of integration --> 
* 	<eqn 							spec="MULTIPLE NODATA"
* 										attribute1="var REQUIRED STRING"
* 										attribute2="src REQUIRED STRING"/>
* </OperatorSplitChem>
*
*/

namespace Uintah{

	class OperatorSplitChem{ 

		public: 

			OperatorSplitChem( ArchesLabel* field_labels );
			~OperatorSplitChem(); 

			/** @brief Input file interface */
			void problemSetup( const ProblemSpecP& params ); 

			/** @brief Integrate from \phi^t to \phi^* -- this is the inteface*/
			void sched_integrate( const LevelP& level, SchedulerP& sched ); 

			/** @brief Schedules the integration */ 
			void sched_eval( const LevelP& level, SchedulerP& sched ); 											///< Schedules the integration. 

		private: 

			/** @brief Evaluation of the integration task */ 
       void eval( const ProcessorGroup* pc, 
                  const PatchSubset* patches,
                	const MaterialSubset* matls,
                	DataWarehouse* old_dw,
                	DataWarehouse* new_dw ); 

			bool _do_split_chem; 									///< Does the operator splitting? 

			int  _N; 														  ///< number of time substeps

			std::map<std::string, std::string> _eqns; 			  		///< which equations are used and the paired source terms
			std::map<std::string, const VarLabel*> _eqn_labels; 	///< which labels are associated with which equation

			ArchesLabel* _field_labels; 


	}; 

}

#endif

