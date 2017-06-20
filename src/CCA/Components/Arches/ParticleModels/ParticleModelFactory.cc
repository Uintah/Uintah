/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <CCA/Components/Arches/ParticleModels/ParticleModelFactory.h>

#include <CCA/Components/Arches/Task/TaskInterface.h>

// Specific model headers:
#include <CCA/Components/Arches/ParticleModels/BodyForce.h>
#include <CCA/Components/Arches/ParticleModels/CoalDensity.h>
#include <CCA/Components/Arches/ParticleModels/CoalTemperature.h>
#include <CCA/Components/Arches/ParticleModels/Constant.h>
#include <CCA/Components/Arches/ParticleModels/DepositionVelocity.h>
#include <CCA/Components/Arches/ParticleModels/DragModel.h>
#include <CCA/Components/Arches/ParticleModels/ExampleParticleModel.h>
#include <CCA/Components/Arches/ParticleModels/FOWYDevol.h>
#include <CCA/Components/Arches/ParticleModels/RateDeposition.h>
#include <CCA/Components/Arches/ParticleModels/ShaddixEnthalpy.h>
#include <CCA/Components/Arches/ParticleModels/ShaddixOxidation.h>
#include <CCA/Components/Arches/ParticleModels/TotNumDensity.h>

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

  // This is a **HACK** to get the post_update_particle_models to execute in a set order.
  // We need a more holistic (auto-ordering?) approach that fits within the task interface.
  // Specifically,
  // 1) Coal density
  // 2) Total number density
  // 3) Coal temperature
  // 4) Rate deposition
  // 5) Rate velocity
  // 6) Everything else
  // Note that 1-3 can be arbitrarily ordered, but just need to occur at the top.
  int num_models = 0;
  if ( db->findBlock("ParticleModels")){

    ProblemSpecP db_pm = db->findBlock("ParticleModels");
    for ( ProblemSpecP db_model = db_pm->findBlock("model"); db_model != nullptr;
          db_model = db_model->findNextBlock("model") ){

      num_models += 1;

    }
  }

  std::vector<std::string> temp_model_list;
  bool has_rate_dep = false;
  bool has_rate_vel = false;
  std::string rate_dep_name;
  std::string rate_vel_name;

  // hack continues below and is notated with "order hack" comments

  if ( db->findBlock("ParticleModels")){

    ProblemSpecP db_pm = db->findBlock("ParticleModels");

    for ( ProblemSpecP db_model = db_pm->findBlock("model"); db_model != nullptr;
          db_model = db_model->findNextBlock("model") ){

      std::string model_name;
      std::string type;
      db_model->getAttribute("label",model_name );
      db_model->getAttribute("type", type );

      std::string task_name = model_name;

      if ( type == "simple_rate"){

        std::string dependent_type;
        std::string independent_type;
        db_model->findBlock("grid")->getAttribute("dependent_type", dependent_type);
        db_model->findBlock("grid")->getAttribute("independent_type", independent_type);

        if ( dependent_type == "CC" ){

          if ( independent_type == "CC"){

            TaskInterface::TaskBuilder* tsk = scinew
              ExampleParticleModel<CCVariable<double> ,CCVariable<double> >::Builder(task_name, 0, model_name, N);

            register_task( task_name, tsk );
            _post_update_particle_tasks.push_back(task_name);

          } else {
            throw InvalidValue("Error: Independent grid type not recognized.",__FILE__,__LINE__);
          }

          //else lagrangian particle type...need to add
        } else {
          throw InvalidValue("Error: Dependent grid type not recognized.",__FILE__,__LINE__);
        }

        temp_model_list.insert(temp_model_list.end(), task_name); // order hack

      } else if  ( type == "drag" ) {

        std::string dependent_type;
        std::string independent_type;
        db_model->findBlock("grid")->getAttribute("dependent_type", dependent_type);
        db_model->findBlock("grid")->getAttribute("independent_type", independent_type);

        if ( dependent_type == "CC" ){

          if ( independent_type == "CC"){

            TaskInterface::TaskBuilder* tsk = scinew
            DragModel<constCCVariable<double>, CCVariable<double> >::Builder(task_name, 0, model_name, N);

            register_task( task_name, tsk );
            _post_update_particle_tasks.push_back(task_name);

          } else {
            throw InvalidValue("Error: Independent grid type not recognized.",__FILE__,__LINE__);
          }

          //else lagrangian particle type...need to add
        } else {
          throw InvalidValue("Error: Dependent grid type not recognized.",__FILE__,__LINE__);
        }

        temp_model_list.insert(temp_model_list.end(), task_name); // order hack

      } else if  ( type == "gravity" ) {

        std::string dependent_type;
        std::string independent_type;
        db_model->findBlock("grid")->getAttribute("dependent_type", dependent_type);
        db_model->findBlock("grid")->getAttribute("independent_type", independent_type);

        if ( dependent_type == "CC" ){

          if ( independent_type == "CC"){

            TaskInterface::TaskBuilder* tsk = scinew
            BodyForce<CCVariable<double>, CCVariable<double> >::Builder(task_name, 0, model_name, N);

            register_task( task_name, tsk );
            _post_update_particle_tasks.push_back(task_name);

          } else {
            throw InvalidValue("Error: Independent grid type not recognized.",__FILE__,__LINE__);
          }

          //else lagrangian particle type...need to add
        } else {
          throw InvalidValue("Error: Dependent grid type not recognized.",__FILE__,__LINE__);
        }

        temp_model_list.insert(temp_model_list.end(), task_name); // order hack

      } else if  ( type == "constant" ) {

        std::string dependent_type;
        if ( db_model->findBlock("grid") != nullptr ){
          db_model->findBlock("grid")->getAttribute("dependent_type", dependent_type);
        } else {
          throw InvalidValue("Error: You must specify the <grid> for the constant model", __FILE__, __LINE__);
        }

        if ( dependent_type == "CC" ){

          TaskInterface::TaskBuilder* tsk = scinew
          Constant<CCVariable<double> >::Builder(task_name, 0, model_name, N);

          register_task( task_name, tsk );
          _post_update_particle_tasks.push_back(task_name);

          //else lagrangian particle type...need to add
        } else {
          throw InvalidValue("Error: Dependent grid type not recognized.",__FILE__,__LINE__);
        }

        temp_model_list.insert(temp_model_list.end(), task_name); // order hack

      } else if ( type == "coal_density" ){

        TaskInterface::TaskBuilder* tsk = scinew CoalDensity::Builder(task_name,0,N);
        register_task( task_name, tsk );

        _coal_models.push_back(task_name);
        _post_update_particle_tasks.push_back(task_name);

        temp_model_list.insert(temp_model_list.begin(), task_name); //order hack

      } else if ( type == "coal_temperature" ) {

        TaskInterface::TaskBuilder* tsk = scinew CoalTemperature::Builder(task_name,0,N);
        register_task( task_name, tsk );

        _coal_models.push_back(task_name);
        _post_update_particle_tasks.push_back(task_name);

        temp_model_list.insert(temp_model_list.begin(), task_name); // order hack

      } else if ( type == "deposition_velocity" ) {

        TaskInterface::TaskBuilder* tsk = scinew DepositionVelocity::Builder(task_name,0,N,_shared_state);
        register_task( task_name, tsk );

        _coal_models.push_back(task_name);
        _post_update_particle_tasks.push_back(task_name);

        has_rate_vel = true; // order hack
        rate_vel_name = task_name; // order hack

      } else if ( type == "rate_deposition" ) {

        TaskInterface::TaskBuilder* tsk = scinew RateDeposition::Builder(task_name,0,N);
        register_task( task_name, tsk );

        _coal_models.push_back(task_name);
        _post_update_particle_tasks.push_back(task_name);

        has_rate_dep = true; // order hack
        rate_dep_name = task_name; // order hack

      } else if ( type == "total_number_density" ) {

        TaskInterface::TaskBuilder* tsk = scinew TotNumDensity::Builder(task_name, 0);
        register_task( task_name, tsk );

        _active_tasks.push_back(task_name);
        _post_update_particle_tasks.push_back(task_name);

        temp_model_list.insert(temp_model_list.begin(), task_name); // order hack

      } else if  ( type == "fowy_devolatilization" ) {

        std::string dependent_type;
        std::string independent_type;
        db_model->findBlock("grid")->getAttribute("dependent_type", dependent_type);
        db_model->findBlock("grid")->getAttribute("independent_type", independent_type);

        if ( dependent_type == "CC" ){

          if ( independent_type == "CC"){

            TaskInterface::TaskBuilder* tsk = scinew
            FOWYDevol<CCVariable<double> >::Builder(task_name, 0, model_name, N);

            register_task( task_name, tsk );
            _active_tasks.push_back(task_name);
            _post_update_particle_tasks.push_back(task_name);

          } else {
            throw InvalidValue("Error: Independent grid type not recognized: "+independent_type,__FILE__,__LINE__);
          }

        } else {
          throw InvalidValue("Error: Dependent grid type not recognized.",__FILE__,__LINE__);
        }

        temp_model_list.insert(temp_model_list.end(), task_name); // order hack

      } else if  ( type == "shaddix_oxidation" ) {

        std::string dependent_type;
        //std::string independent_type;
        db_model->findBlock("grid")->getAttribute("dependent_type", dependent_type);
        //db_model->findBlock("grid")->getAttribute("independent_type", independent_type);

        if ( dependent_type == "CC" ){


          TaskInterface::TaskBuilder* tsk = scinew
          ShaddixOxidation<CCVariable<double> >::Builder(task_name, 0, model_name, N);

          register_task( task_name, tsk );
          _active_tasks.push_back(task_name);
          _post_update_particle_tasks.push_back(task_name);

        } else {
          throw InvalidValue("Error: Dependent grid type not recognized.",__FILE__,__LINE__);
        }

        temp_model_list.insert(temp_model_list.end(), task_name); // order hack

      } else if  ( type == "shaddix_enthalpy" ) {

        std::string dependent_type;
        //std::string independent_type;
        db_model->findBlock("grid")->getAttribute("dependent_type", dependent_type);
        //db_model->findBlock("grid")->getAttribute("independent_type", independent_type);

        if ( dependent_type == "CC" ){

          TaskInterface::TaskBuilder* tsk = scinew
          ShaddixEnthalpy<CCVariable<double> >::Builder(task_name, 0, model_name, N);

          register_task( task_name, tsk );
          _active_tasks.push_back(task_name);
          _post_update_particle_tasks.push_back(task_name);

        } else {
          throw InvalidValue("Error: Dependent grid type not recognized.",__FILE__,__LINE__);
        }

        temp_model_list.insert(temp_model_list.end(), task_name); // order hack

      } else {

        throw InvalidValue("Error: Particle model not recognized.",__FILE__,__LINE__);

      }

    }

    //---- order hack ----
    if ( has_rate_dep ){
      temp_model_list.push_back(rate_dep_name);
    }
    if ( has_rate_vel ){
      temp_model_list.push_back(rate_vel_name);
    }

    _post_update_particle_tasks = temp_model_list;
    //---- end order hack ---

  }
}

void
ParticleModelFactory::build_all_tasks( ProblemSpecP& db )
{

  if ( db->findBlock("ParticleModels")){

    ProblemSpecP db_pm = db->findBlock("ParticleModels");

    for (ProblemSpecP db_model = db_pm->findBlock("model"); db_model != nullptr; db_model = db_model->findNextBlock("model")){

      std::string model_name;
      std::string type;
      db_model->getAttribute("label",model_name );
      db_model->getAttribute("type", type );

      print_task_setup_info( model_name, type );
      TaskInterface* tsk = retrieve_task(model_name);

      tsk->problemSetup( db_model );

      tsk->create_local_labels();

    }

  }
}
