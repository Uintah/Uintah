#include <CCA/Components/Arches/Transport/TransportFactory.h>
#include <CCA/Components/Arches/Transport/ScalarRHS.h>
#include <CCA/Components/Arches/Transport/FEUpdate.h>
#include <CCA/Components/Arches/Transport/SSPInt.h>
#include <CCA/Components/Arches/Task/TaskInterface.h>

using namespace Uintah; 

TransportFactory::TransportFactory()
{}

TransportFactory::~TransportFactory()
{}

void 
TransportFactory::register_all_tasks( ProblemSpecP& db )
{ 

  /*
   
     <ScalarTransport>

      <eqn label="my_eqn">
        <diffusion/>
        <convection type="super_bee"/>
        <stage value="1"/> ???
        <clip high="1.0" low="0.0"/>
        <initialization  type="constant">...</intialization>
      </eqn> 

      <eqn label=....>
      .... and so on....

     </ScalarTransport>

     <Momentum>
      <convection type="..."/>
     </Momentum>


  */

  if ( db->findBlock("ScalarTransport") ){

    ProblemSpecP db_st = db->findBlock("ScalarTransport"); 

    for (ProblemSpecP eqn_db = db_st->findBlock("eqn"); eqn_db != 0; eqn_db = eqn_db->findNextBlock("eqn")){

      std::string eqn_name = "null"; 
      eqn_db->getAttribute("label", eqn_name);
      TaskInterface::TaskBuilder* tsk = scinew ScalarRHS::Builder(eqn_name,0); 
      register_task( eqn_name, tsk ); 

      _active_tasks.push_back(eqn_name); 

    }

    typedef SpatialOps::SVolField SVol;
    std::string update_task_name = "scalar_fe_update"; 
    FEUpdate<SVol>::Builder* tsk = scinew FEUpdate<SVol>::Builder( update_task_name, 0, _active_tasks ); 
    register_task( update_task_name, tsk ); 

    typedef SpatialOps::SVolField SVol;
    std::string ssp_task_name = "scalar_ssp_update"; 
    SSPInt<SVol>::Builder* tsk2 = scinew SSPInt<SVol>::Builder( ssp_task_name, 0, _active_tasks ); 
    register_task( ssp_task_name, tsk2 ); 

    _active_tasks.push_back( update_task_name );  
    _active_tasks.push_back( ssp_task_name );  
  
  }

}

void 
TransportFactory::build_all_tasks( ProblemSpecP& db )
{ 

  if ( db->findBlock("ScalarTransport") ){

    ProblemSpecP db_st = db->findBlock("ScalarTransport"); 

    for (ProblemSpecP eqn_db = db_st->findBlock("eqn"); eqn_db != 0; eqn_db = eqn_db->findNextBlock("eqn")){

      std::string eqn_name = "null"; 
      eqn_db->getAttribute("label", eqn_name);
      TaskInterface* tsk = retrieve_task(eqn_name); 
      tsk->problemSetup( eqn_db );

    }

    TaskInterface* tsk = retrieve_task("scalar_fe_update"); 
    tsk->problemSetup( db ); 

    tsk = retrieve_task("scalar_ssp_update"); 
    tsk->problemSetup( db ); 
  
  }
}
