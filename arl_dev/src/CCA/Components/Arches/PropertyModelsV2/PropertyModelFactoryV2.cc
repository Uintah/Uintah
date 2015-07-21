#include <CCA/Components/Arches/PropertyModelsV2/PropertyModelFactoryV2.h>
#include <CCA/Components/Arches/Task/TaskInterface.h>
//Specific models: 
#include <CCA/Components/Arches/PropertyModelsV2/WallHFVariable.h>
#include <CCA/Components/Arches/PropertyModelsV2/VariableStats.h>
#include <CCA/Components/Arches/PropertyModelsV2/DensityPredictor.h>

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
        db_model->getWithDefault("variable_type",var_type,"NA");

        bool set_type = true; 
        TypeDescription::Type the_old_type; 
        TypeDescription::Type uintah_type;

        if ( var_type == "NA" ){ 

          for ( ProblemSpecP var_db = db_model->findBlock("single_variable"); var_db != 0; 
                var_db = var_db->findNextBlock("single_variable") ){ 

            std::string var_name; 
            var_db->getAttribute("label", var_name);

            const VarLabel* varlabel = VarLabel::find(var_name); 
            if ( varlabel != 0 ){ 
              uintah_type = varlabel->typeDescription()->getType();

              if ( set_type ){ 
                the_old_type = uintah_type; 
                set_type = false;
              }

              if ( uintah_type != the_old_type ){ 
                throw ProblemSetupException("Error: Two variables with different types in variable_stats model for model: "+name,__FILE__,__LINE__);
              }
            }
          }

          for ( ProblemSpecP var_db = db_model->findBlock("flux_variable"); var_db != 0; 
                var_db = var_db->findNextBlock("flux_variable") ){ 

            std::string phi_label = "NA";
            var_db->getAttribute("phi",phi_label); 

            if ( phi_label != "NA" ){ 

              const VarLabel* varlabel = VarLabel::find(phi_label); 
              if ( varlabel != 0 ){ 

                uintah_type = varlabel->typeDescription()->getType();

                if ( set_type ){ 
                  the_old_type = uintah_type; 
                }

                if ( uintah_type != the_old_type ){ 
                  throw ProblemSetupException("Error: Two variables with different types in variable_stats model named: "+name,__FILE__,__LINE__);
                }
                  
              }
            }
          }
        } else { 
          if ( var_type == "svol"){
            uintah_type = TypeDescription::CCVariable;
          } else if ( var_type == "xvol"){ 
            uintah_type = TypeDescription::SFCXVariable;
          } else if ( var_type == "yvol"){ 
            uintah_type = TypeDescription::SFCYVariable;
          } else if ( var_type == "zvol"){ 
            uintah_type = TypeDescription::SFCZVariable;
          } else { 
            throw ProblemSetupException("Error: Explicit variable type description in ups file not recognized for variable_stats model: "+name,__FILE__,__LINE__);
          }
        }

        if ( uintah_type == TypeDescription::CCVariable ){ 

          TaskInterface::TaskBuilder* tsk = scinew VariableStats<SpatialOps::SVolField>::Builder( name, 0, _shared_state ); 
          register_task( name, tsk ); 
          _finalize_property_tasks.push_back( name ); 

        } else if ( uintah_type == TypeDescription::SFCXVariable ){ 

          TaskInterface::TaskBuilder* tsk = scinew VariableStats<SpatialOps::XVolField>::Builder( name, 0, _shared_state ); 
          register_task( name, tsk ); 
          _finalize_property_tasks.push_back( name ); 

        } else if ( uintah_type == TypeDescription::SFCYVariable ){ 

          TaskInterface::TaskBuilder* tsk = scinew VariableStats<SpatialOps::YVolField>::Builder( name, 0, _shared_state ); 
          register_task( name, tsk ); 
          _finalize_property_tasks.push_back( name ); 

        } else if ( uintah_type == TypeDescription::SFCZVariable ){ 

          TaskInterface::TaskBuilder* tsk = scinew VariableStats<SpatialOps::ZVolField>::Builder( name, 0, _shared_state ); 
          register_task( name, tsk ); 
          _finalize_property_tasks.push_back( name ); 

        } else { 

          throw InvalidValue("Error: variable type not supported for variable_stats property model",__FILE__,__LINE__);

        }

      } else if ( type == "density_predictor" ) {

        TaskInterface::TaskBuilder* tsk = scinew DensityPredictor::Builder( name, 0 ); 
        register_task( name, tsk ); 


      } else { 

        throw InvalidValue("Error: Property model not recognized.",__FILE__,__LINE__); 

      }

      //put the tasks in a type->(task names) map
      TypeToTaskMap::iterator iter = _type_to_tasks.find(type);
      iter->second.push_back(name);

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
