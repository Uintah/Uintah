#include <CCA/Components/Arches/PropertyModelsV2/PropertyModelFactoryV2.h>
#include <CCA/Components/Arches/PropertyModelsV2/CoalDensity.h>
#include <CCA/Components/Arches/PropertyModelsV2/ConstEnvProperty.h>
#include <CCA/Components/Arches/PropertyModelsV2/CoalTemperature.h>
#include <CCA/Components/Arches/PropertyModelsV2/CoalTemperatureNebo.h>
#include <CCA/Components/Arches/Task/TaskInterface.h>

using namespace Uintah; 

PropertyModelFactoryV2::PropertyModelFactoryV2()
{}

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

    for ( ProblemSpecP db_model = db_m->findBlock("model"); db_model != 0; db_model=db_model->findNextBlock("model")){ 

      std::string name;
      std::string type; 
      db_model->getAttribute("label", name);
      db_model->getAttribute("type", type);

      if ( type == "coal_density" ){ 

        TaskInterface::TaskBuilder* tsk = scinew CoalDensity::Builder(name,0); 
        register_task( name, tsk ); 

        _active_tasks.push_back(name); 
        _coal_models.push_back(name); 

      } else if ( type == "coal_temperature" ) { 

        TaskInterface::TaskBuilder* tsk = scinew CoalTemperature::Builder(name,0); 
        register_task( name, tsk ); 

        _active_tasks.push_back(name); 
        _coal_models.push_back(name); 

      } else if ( type == "coal_temperature_nebo" ) { 

        TaskInterface::TaskBuilder* tsk = scinew CoalTemperatureNebo::Builder(name,0); 
        register_task( name, tsk ); 

        _active_tasks.push_back(name); 
        _coal_models.push_back(name); 

      } else if ( type == "const_environment"){ 

        TaskInterface::TaskBuilder* tsk = scinew ConstEnvProperty::Builder(name, 0); 
        register_task(name, tsk); 

        _active_tasks.push_back(name); 
        _coal_models.push_back(name); 
       
      } else { 

        throw InvalidValue("Error: PropertyModel type not recognized.",__FILE__,__LINE__); 

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
