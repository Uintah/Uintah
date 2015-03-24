#include <CCA/Components/Arches/PropertyModelsV2/PropertyModelFactoryV2.h>
#include <CCA/Components/Arches/Task/TaskInterface.h>
//Specific models: 
#include <CCA/Components/Arches/PropertyModelsV2/WallHFVariable.h>
#include <CCA/Components/Arches/PropertyModelsV2/VariableStats.h>

using namespace Uintah; 

PropertyModelFactoryV2::PropertyModelFactoryV2( SimulationStateP& shared_state )
{

  _shared_state = shared_state; 

}

PropertyModelFactoryV2::~PropertyModelFactoryV2()
{}

void 
PropertyModelFactoryV2::register_all_tasks( ProblemSpecP& db )
{ 

  /*
   
  <PropertyModelsV2>
    <model name="A_MODEL" type="A_TYPE">
      <stuff....>
    </model>
  </PropertyModelsV2>


  */

  if ( db->findBlock("PropertyModelsV2")){ 

    ProblemSpecP db_m = db->findBlock("PropertyModelsV2"); 

    for ( ProblemSpecP db_model = db_m->findBlock("model"); db_model != 0; 
          db_model=db_model->findNextBlock("model")){ 

      std::string name;
      std::string type; 
      db_model->getAttribute("label", name);
      db_model->getAttribute("type", type);

      if ( type == "wall_heatflux_variable" ){ 

        TaskInterface::TaskBuilder* tsk = scinew WallHFVariable::Builder( name, 0, _shared_state ); 
        register_task( name, tsk ); 
        _pre_update_property_tasks.push_back( name ); 

      } else if ( type == "variable_stats" ){ 

        std::string var_type; 
        db_model->require("variable_type",var_type);

        if ( var_type == "svol" ){ 

          TaskInterface::TaskBuilder* tsk = scinew VariableStats<SpatialOps::SVolField>::Builder( name, 0, _shared_state ); 
          register_task( name, tsk ); 
          _finalize_property_tasks.push_back( name ); 

        } else if ( var_type == "xvol" ){ 

          TaskInterface::TaskBuilder* tsk = scinew VariableStats<SpatialOps::XVolField>::Builder( name, 0, _shared_state ); 
          register_task( name, tsk ); 
          _finalize_property_tasks.push_back( name ); 

        } else if ( var_type == "yvol" ){ 

          //TaskInterface::TaskBuilder* tsk = scinew TimeAve<SpatialOps::YVolField>::Builder( name, 0, _shared_state ); 
          //register_task( name, tsk ); 
          //_finalize_property_tasks.push_back( name ); 

        } else if ( var_type == "zvol" ){ 

          //TaskInterface::TaskBuilder* tsk = scinew TimeAve<SpatialOps::ZVolField>::Builder( name, 0, _shared_state ); 
          //register_task( name, tsk ); 
          //_finalize_property_tasks.push_back( name ); 

        } else { 

          throw InvalidValue("Error: variable type not supported for variable_stats property model",__FILE__,__LINE__);

        }

      } else { 

        throw InvalidValue("Error: Property model not recognized.",__FILE__,__LINE__); 

      }


    }
  }
}

void 
PropertyModelFactoryV2::build_all_tasks( ProblemSpecP& db )
{ 
  if ( db->findBlock("PropertyModelsV2")){ 

    ProblemSpecP db_m = db->findBlock("PropertyModelsV2"); 

    for ( ProblemSpecP db_model = db_m->findBlock("model"); db_model != 0; db_model=db_model->findNextBlock("model")){ 
      std::string name;
      db_model->getAttribute("label", name);
      TaskInterface* tsk = retrieve_task(name); 
      tsk->problemSetup(db_model); 
      tsk->create_local_labels(); 
    }
  }
}
