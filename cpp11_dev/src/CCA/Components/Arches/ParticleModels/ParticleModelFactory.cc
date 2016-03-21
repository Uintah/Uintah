#include <CCA/Components/Arches/ParticleModels/ParticleModelFactory.h>
#include <CCA/Components/Arches/Task/TaskInterface.h>

//Specific model headers:
#include <CCA/Components/Arches/ParticleModels/ExampleParticleModel.h>
#include <CCA/Components/Arches/ParticleModels/DragModel.h>
#include <CCA/Components/Arches/ParticleModels/BodyForce.h>
#include <CCA/Components/Arches/ParticleModels/Constant.h>
#include <CCA/Components/Arches/ParticleModels/RateDeposition.h>
#include <CCA/Components/Arches/ParticleModels/CoalTemperature.h>
#include <CCA/Components/Arches/ParticleModels/CoalTemperatureNebo.h>
#include <CCA/Components/Arches/ParticleModels/DepositionVelocity.h>
#include <CCA/Components/Arches/ParticleModels/CoalDensity.h>
#include <CCA/Components/Arches/ParticleModels/CoalMassClip.h>
#include <CCA/Components/Arches/ParticleModels/TotNumDensity.h>
#include <CCA/Components/Arches/ParticleModels/FOWYDevol.h>
#include <CCA/Components/Arches/ParticleModels/ShaddixOxidation.h>
#include <CCA/Components/Arches/ParticleModels/ShaddixEnthalpy.h>

using namespace Uintah;

ParticleModelFactory::ParticleModelFactory()
{}

ParticleModelFactory::~ParticleModelFactory()
{}

void
ParticleModelFactory::register_all_tasks( ProblemSpecP& db )
{
  /*

    <ParticleModels>
      <model label="some unique name" type="defined type">

        <spec consistent with model type>

      </model>
    </ParticleModels>


   */
  int N;

  if (db->findBlock("CQMOM") ) {
    std::vector<int> N_i;
    db->findBlock("CQMOM")->require("QuadratureNodes",N_i);
    N = 1;
    for (unsigned int i = 0; i < N_i.size(); i++ ) {
      N *= N_i[i];
    }
  } else if (db->findBlock("DQMOM") ) {
    db->findBlock("DQMOM")->require("number_quad_nodes",N);
  } else if (db->findBlock("LagrangianParticles")){
    N = 1;
  }

  if ( db->findBlock("ParticleModels")){

    ProblemSpecP db_pm = db->findBlock("ParticleModels");

    for (ProblemSpecP db_model = db_pm->findBlock("model"); db_model != 0;
         db_model = db_model->findNextBlock("model")){

      std::string model_name;
      std::string type;
      db_model->getAttribute("label",model_name );
      db_model->getAttribute("type", type );

      std::string task_name = model_name;

      typedef SpatialOps::SVolField SVol;

      if ( type == "simple_rate"){

        std::string dependent_type;
        std::string independent_type;
        db_model->findBlock("grid")->getAttribute("dependent_type", dependent_type);
        db_model->findBlock("grid")->getAttribute("independent_type", independent_type);

        if ( dependent_type == "svol" ){

          if ( independent_type == "svol"){

            TaskInterface::TaskBuilder* tsk = new
              ExampleParticleModel<SVol,SVol>::Builder(task_name, 0, model_name, N);

            register_task( task_name, tsk );
            _pre_update_particle_tasks.push_back(task_name);

          } else {
            throw InvalidValue("Error: Independent grid type not recognized.",__FILE__,__LINE__);
          }

          //else lagrangian particle type...need to add
        } else {
          throw InvalidValue("Error: Dependent grid type not recognized.",__FILE__,__LINE__);
        }


      } else if  ( type == "drag" ) {

        std::string dependent_type;
        std::string independent_type;
        db_model->findBlock("grid")->getAttribute("dependent_type", dependent_type);
        db_model->findBlock("grid")->getAttribute("independent_type", independent_type);

        if ( dependent_type == "svol" ){

          if ( independent_type == "svol"){

            TaskInterface::TaskBuilder* tsk = new
            DragModel<SVol,SVol>::Builder(task_name, 0, model_name, N);

            register_task( task_name, tsk );
            _pre_update_particle_tasks.push_back(task_name);

          } else {
            throw InvalidValue("Error: Independent grid type not recognized.",__FILE__,__LINE__);
          }

          //else lagrangian particle type...need to add
        } else {
          throw InvalidValue("Error: Dependent grid type not recognized.",__FILE__,__LINE__);
        }

      } else if  ( type == "gravity" ) {

        std::string dependent_type;
        std::string independent_type;
        db_model->findBlock("grid")->getAttribute("dependent_type", dependent_type);
        db_model->findBlock("grid")->getAttribute("independent_type", independent_type);

        if ( dependent_type == "svol" ){

          if ( independent_type == "svol"){

            TaskInterface::TaskBuilder* tsk = new
            BodyForce<SVol,SVol>::Builder(task_name, 0, model_name, N);

            register_task( task_name, tsk );
            _pre_update_particle_tasks.push_back(task_name);

          } else {
            throw InvalidValue("Error: Independent grid type not recognized.",__FILE__,__LINE__);
          }

          //else lagrangian particle type...need to add
        } else {
          throw InvalidValue("Error: Dependent grid type not recognized.",__FILE__,__LINE__);
        }

      } else if  ( type == "constant" ) {

        std::string dependent_type;
        std::string independent_type;
        db_model->findBlock("grid")->getAttribute("dependent_type", dependent_type);
        db_model->findBlock("grid")->getAttribute("independent_type", independent_type);

        if ( dependent_type == "svol" ){

          if ( independent_type == "svol"){

            TaskInterface::TaskBuilder* tsk = new
            Constant<SVol>::Builder(task_name, 0, model_name, N);

            register_task( task_name, tsk );
            _pre_update_particle_tasks.push_back(task_name);

          } else {
            throw InvalidValue("Error: Independent grid type not recognized: "+independent_type,__FILE__,__LINE__);
          }

          //else lagrangian particle type...need to add
        } else {
          throw InvalidValue("Error: Dependent grid type not recognized.",__FILE__,__LINE__);
        }

      } else if ( type == "coal_density" ){

        TaskInterface::TaskBuilder* tsk = new CoalDensity::Builder(task_name,0,N);
        register_task( task_name, tsk );

        _coal_models.push_back(task_name);
        _pre_update_particle_tasks.push_back(task_name);

      } else if ( type == "coal_temperature" ) {

        TaskInterface::TaskBuilder* tsk = new CoalTemperature::Builder(task_name,0,N);
        register_task( task_name, tsk );

        _coal_models.push_back(task_name);
        _pre_update_particle_tasks.push_back(task_name);

      } else if ( type == "coal_temperature_nebo" ) {

        TaskInterface::TaskBuilder* tsk = new CoalTemperatureNebo::Builder(task_name,0);
        register_task( task_name, tsk );

        _coal_models.push_back(task_name);
        _pre_update_particle_tasks.push_back(task_name);

      } else if ( type == "deposition_velocity" ) {

        TaskInterface::TaskBuilder* tsk = new DepositionVelocity::Builder(task_name,0,N,_shared_state);
        register_task( task_name, tsk );

        _coal_models.push_back(task_name);
        _pre_update_particle_tasks.push_back(task_name);

      } else if ( type == "coal_mass_clip" ) {

        TaskInterface::TaskBuilder* tsk = new CoalMassClip::Builder(task_name, 0, N);
        register_task( task_name, tsk );

        _post_update_coal_tasks.push_back(task_name);
      } else if ( type == "rate_deposition" ) {

        TaskInterface::TaskBuilder* tsk = new RateDeposition::Builder(task_name,0,N);
        register_task( task_name, tsk );

        _coal_models.push_back(task_name);
        _pre_update_particle_tasks.push_back(task_name);
      } else if ( type == "total_number_density" ) {

        TaskInterface::TaskBuilder* tsk = new TotNumDensity::Builder(task_name, 0);
        register_task( task_name, tsk );

        _active_tasks.push_back(task_name);
        _pre_update_particle_tasks.push_back(task_name);

      } else if  ( type == "fowy_devolatilization" ) {

        std::string dependent_type;
        std::string independent_type;
        db_model->findBlock("grid")->getAttribute("dependent_type", dependent_type);
        db_model->findBlock("grid")->getAttribute("independent_type", independent_type);

        if ( dependent_type == "svol" ){

          if ( independent_type == "svol"){

            TaskInterface::TaskBuilder* tsk = new
            FOWYDevol<SVol,SVol>::Builder(task_name, 0, model_name, N);

            register_task( task_name, tsk );
            _active_tasks.push_back(task_name);
            _pre_update_particle_tasks.push_back(task_name);

          } else {
            throw InvalidValue("Error: Independent grid type not recognized: "+independent_type,__FILE__,__LINE__);
          }

        } else {
          throw InvalidValue("Error: Dependent grid type not recognized.",__FILE__,__LINE__);
        }

      } else if  ( type == "shaddix_oxidation" ) {

        std::string dependent_type;
        std::string independent_type;
        db_model->findBlock("grid")->getAttribute("dependent_type", dependent_type);
        db_model->findBlock("grid")->getAttribute("independent_type", independent_type);

        if ( dependent_type == "svol" ){

          if ( independent_type == "svol"){

            TaskInterface::TaskBuilder* tsk = new
            ShaddixOxidation<SVol,SVol>::Builder(task_name, 0, model_name, N);

            register_task( task_name, tsk );
            _active_tasks.push_back(task_name);
            _pre_update_particle_tasks.push_back(task_name);

          } else {
            throw InvalidValue("Error: Independent grid type not recognized: "+independent_type,__FILE__,__LINE__);
          }
        } else {
          throw InvalidValue("Error: Dependent grid type not recognized.",__FILE__,__LINE__);
        }

      } else if  ( type == "shaddix_enthalpy" ) {

        std::string dependent_type;
        std::string independent_type;
        db_model->findBlock("grid")->getAttribute("dependent_type", dependent_type);
        db_model->findBlock("grid")->getAttribute("independent_type", independent_type);

        if ( dependent_type == "svol" ){

          if ( independent_type == "svol"){

            TaskInterface::TaskBuilder* tsk = new
            ShaddixEnthalpy<SVol,SVol>::Builder(task_name, 0, model_name, N);

            register_task( task_name, tsk );
            _active_tasks.push_back(task_name);
            _pre_update_particle_tasks.push_back(task_name);

          } else {
            throw InvalidValue("Error: Independent grid type not recognized: "+independent_type,__FILE__,__LINE__);
          }
        } else {
          throw InvalidValue("Error: Dependent grid type not recognized.",__FILE__,__LINE__);
        }

      } else {

        throw InvalidValue("Error: Particle model not recognized.",__FILE__,__LINE__);

      }

    }
  }
}

void
ParticleModelFactory::build_all_tasks( ProblemSpecP& db )
{

  if ( db->findBlock("ParticleModels")){

    ProblemSpecP db_pm = db->findBlock("ParticleModels");

    for (ProblemSpecP db_model = db_pm->findBlock("model"); db_model != 0;
         db_model = db_model->findNextBlock("model")){

      std::string model_name;
      std::string type;
      db_model->getAttribute("label",model_name );
      db_model->getAttribute("type", type );

      TaskInterface* tsk = retrieve_task(model_name);

      tsk->problemSetup( db_model );

      tsk->create_local_labels();

    }

  }
}
