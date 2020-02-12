/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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
#include <CCA/Components/Arches/ParticleModels/Burnout.h>
#include <CCA/Components/Arches/ParticleModels/Constant.h>
#include <CCA/Components/Arches/ParticleModels/DepositionVelocity.h>
#include <CCA/Components/Arches/ParticleModels/DepositionEnthalpy.h>
#include <CCA/Components/Arches/ParticleModels/DragModel.h>
#include <CCA/Components/Arches/ParticleModels/ExampleParticleModel.h>
#include <CCA/Components/Arches/ParticleModels/FOWYDevol.h>
#include <CCA/Components/Arches/ParticleModels/RateDeposition.h>
#include <CCA/Components/Arches/ParticleModels/ShaddixEnthalpy.h>
#include <CCA/Components/Arches/ParticleModels/ShaddixOxidation.h>
#include <CCA/Components/Arches/ParticleModels/TotNumDensity.h>
#include <CCA/Components/Arches/ParticleModels/CharOxidationps.h>
#include <CCA/Components/Arches/ParticleModels/PartVariablesDQMOM.h>
#include <CCA/Components/Arches/ParticleModels/DQMOMNoInversion.h>
#include <CCA/Components/Arches/ParticleModels/FaceParticleVel.h>
#include <CCA/Components/Arches/ParticleModels/WDragModel.h>


using namespace Uintah;

ParticleModelFactory::ParticleModelFactory( const ApplicationCommon* arches ) :
TaskFactoryBase(arches)
{

  _factory_name = "ParticleModelFactory";

}

ParticleModelFactory::~ParticleModelFactory()
{}

void
ParticleModelFactory::register_all_tasks( ProblemSpecP& db )
{

  int N = 0;

  if (db->findBlock("CQMOM") ) {

    N = ArchesCore::get_num_env(db, ArchesCore::CQMOM_METHOD);

  } else if (db->findBlock("DQMOM") ) {

    N = ArchesCore::get_num_env(db, ArchesCore::DQMOM_METHOD);

    //Currently, this allocates the source terms for the DQMOM transport
    // for both the production and new kokkos line of code. Thus, it
    // always needs to be 'on' for both forks.
    std::string task_name = "[DQMOMNoInversion]";
    TaskInterface::TaskBuilder* tsk = scinew DQMOMNoInversion::Builder( task_name, 0, N );
    register_task( task_name, tsk, db->findBlock("DQMOM") );

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

  bool has_rate_dep = false;
  bool has_rate_vel = false;
  bool has_rate_enth = false;
  std::string rate_dep_name;
  std::string rate_vel_name;
  std::string rate_enth_name;

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

            register_task( task_name, tsk, db_model );
            m_particle_models.push_back(task_name);

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

        if ( dependent_type == "CC" ){

          if ( independent_type == "CC"){

            TaskInterface::TaskBuilder* tsk = scinew
            DragModel<constCCVariable<double>, CCVariable<double> >::Builder(task_name, 0, model_name, N);

            register_task( task_name, tsk, db_model );
            m_particle_models.push_back(task_name);

          } else {
            throw InvalidValue("Error: Independent grid type not recognized.",__FILE__,__LINE__);
          }

          //else lagrangian particle type...need to add
        } else {
          throw InvalidValue("Error: Dependent grid type not recognized.",__FILE__,__LINE__);
        }

      } else if  ( type == "wdrag" ) {

        const int nQn_part = ArchesCore::get_num_env( db, ArchesCore::DQMOM_METHOD );
        for ( int i = 0; i < nQn_part; i++ ){
          std::stringstream ienv;
          ienv << i;
          std::string task_name_N = task_name + "_qn" + ienv.str();
          TaskInterface::TaskBuilder* tsk = scinew WDragModel<CCVariable<double> >::Builder(task_name_N, 0, i);
          register_task( task_name_N, tsk, db_model );
          m_particle_models.push_back(task_name_N);
        }
      } else if  ( type == "char_oxidation_ps" ) {

        const int nQn_part = ArchesCore::get_num_env( db, ArchesCore::DQMOM_METHOD );
        for ( int i = 0; i < nQn_part; i++ ){
          std::stringstream ienv;
          ienv << i;
          std::string task_name_N = task_name + "_qn" + ienv.str();
          TaskInterface::TaskBuilder* tsk = scinew CharOxidationps<CCVariable<double> >::Builder(task_name_N, 0, i);
          register_task( task_name_N, tsk, db_model );
          m_particle_models.push_back(task_name_N);
        }

      } else if  ( type == "gravity" ) {

        std::string dependent_type;
        std::string independent_type;
        db_model->findBlock("grid")->getAttribute("dependent_type", dependent_type);
        db_model->findBlock("grid")->getAttribute("independent_type", independent_type);

        if ( dependent_type == "CC" ){

          if ( independent_type == "CC"){

            TaskInterface::TaskBuilder* tsk = scinew
            BodyForce<CCVariable<double>, CCVariable<double> >::Builder(task_name, 0, model_name, N);

            register_task( task_name, tsk, db_model );
            m_particle_models.push_back(task_name);

          } else {
            throw InvalidValue("Error: Independent grid type not recognized.",__FILE__,__LINE__);
          }

          //else lagrangian particle type...need to add
        } else {
          throw InvalidValue("Error: Dependent grid type not recognized.",__FILE__,__LINE__);
        }

      } else if  ( type == "particle_face_velocity" ) {

        std::string dependent_type;
        if ( db_model->findBlock("grid") != nullptr ){
          db_model->findBlock("grid")->getAttribute("dependent_type", dependent_type);
        } else {
          throw InvalidValue("Error: You must specify the <grid dependent_type=*> for the particle_face_velocity model.", __FILE__, __LINE__);
        }

        if ( dependent_type == "CC" ){

          TaskInterface::TaskBuilder* tsk = scinew
          FaceParticleVel<CCVariable<double> >::Builder(task_name, 0, model_name);

          register_task( task_name, tsk, db_model );
          m_dqmom_transport_variables.push_back(task_name);

        }

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

          register_task( task_name, tsk, db_model );
          m_particle_models.push_back(task_name);

        } else {
          throw InvalidValue("Error: Dependent grid type not recognized.",__FILE__,__LINE__);
        }

      } else if ( type == "coal_density" ){

        TaskInterface::TaskBuilder* tsk = scinew CoalDensity::Builder(task_name,0,N);
        register_task( task_name, tsk, db_model );

        m_particle_properties.push_back(task_name);

      } else if ( type == "coal_temperature" ) {

        TaskInterface::TaskBuilder* tsk = scinew CoalTemperature::Builder(task_name,0,N);
        register_task( task_name, tsk, db_model );

        m_particle_properties.push_back(task_name);

      } else if ( type == "burnout" ) {

        TaskInterface::TaskBuilder* tsk = scinew Burnout::Builder(task_name,0,N);
        register_task( task_name, tsk, db_model );

        m_particle_properties.push_back(task_name);

      } else if ( type == "deposition_velocity" ) {

        TaskInterface::TaskBuilder* tsk = scinew DepositionVelocity::Builder(task_name,0,N,_materialManager);
        register_task( task_name, tsk, db_model );

        // This model is added to the particle models below per the hack
        has_rate_vel = true; // order hack
        rate_vel_name = task_name; // order hack

      } else if ( type == "deposition_enthalpy" ) {

        TaskInterface::TaskBuilder* tsk = scinew DepositionEnthalpy::Builder(task_name,0,N,_materialManager);
        register_task( task_name, tsk, db_model );

        // This model is added to the particle models below per the hack
        has_rate_enth = true; // order hack
        rate_enth_name = task_name; // order hack

      } else if ( type == "particle_variables_dqmom" ) {

        TaskInterface::TaskBuilder* tsk = scinew PartVariablesDQMOM::Builder(task_name,0);
        register_task( task_name, tsk, db_model );
        m_dqmom_transport_variables.push_back(task_name);

      } else if ( type == "rate_deposition" ) {

        TaskInterface::TaskBuilder* tsk = scinew RateDeposition::Builder(task_name,0,N);
        register_task( task_name, tsk, db_model );

        //This model is added below per the hack
        has_rate_dep = true; // order hack
        rate_dep_name = task_name; // order hack

      } else if ( type == "total_number_density" ) {

        TaskInterface::TaskBuilder* tsk = scinew TotNumDensity::Builder(task_name, 0);
        register_task( task_name, tsk, db_model );

        m_particle_properties.push_back(task_name);

      } else if  ( type == "fowy_devolatilization" ) {

        std::string dependent_type;
        std::string independent_type;
        db_model->findBlock("grid")->getAttribute("dependent_type", dependent_type);
        db_model->findBlock("grid")->getAttribute("independent_type", independent_type);

        if ( dependent_type == "CC" ){

          if ( independent_type == "CC"){

            TaskInterface::TaskBuilder* tsk = scinew
            FOWYDevol<CCVariable<double> >::Builder(task_name, 0, model_name, N);

            register_task( task_name, tsk, db_model );
            m_particle_models.push_back(task_name);

          } else {
            throw InvalidValue("Error: Independent grid type not recognized: "+independent_type,__FILE__,__LINE__);
          }

        } else {
          throw InvalidValue("Error: Dependent grid type not recognized.",__FILE__,__LINE__);
        }

      } else if  ( type == "shaddix_oxidation" ) {

        std::string dependent_type;
        //std::string independent_type;
        db_model->findBlock("grid")->getAttribute("dependent_type", dependent_type);
        //db_model->findBlock("grid")->getAttribute("independent_type", independent_type);

        if ( dependent_type == "CC" ){


          TaskInterface::TaskBuilder* tsk = scinew
          ShaddixOxidation<CCVariable<double> >::Builder(task_name, 0, model_name, N);

          register_task( task_name, tsk, db_model );
          //_active_tasks.push_back(task_name);
          m_particle_models.push_back(task_name);

        } else {
          throw InvalidValue("Error: Dependent grid type not recognized.",__FILE__,__LINE__);
        }

      } else if  ( type == "shaddix_enthalpy" ) {

        std::string dependent_type;
        //std::string independent_type;
        db_model->findBlock("grid")->getAttribute("dependent_type", dependent_type);
        //db_model->findBlock("grid")->getAttribute("independent_type", independent_type);

        if ( dependent_type == "CC" ){

          TaskInterface::TaskBuilder* tsk = scinew
          ShaddixEnthalpy<CCVariable<double> >::Builder(task_name, 0, model_name, N);

          register_task( task_name, tsk, db_model );
          //_active_tasks.push_back(task_name);
          m_particle_models.push_back(task_name);

        } else {
          throw InvalidValue("Error: Dependent grid type not recognized.",__FILE__,__LINE__);
        }

      } else {

        throw InvalidValue("Error: Particle model not recognized.",__FILE__,__LINE__);

      }

    }

    //---- order hack ----
    if ( has_rate_dep ){
      m_deposition_models.push_back(rate_dep_name);
    }
    if ( has_rate_vel ){
      m_deposition_models.push_back(rate_vel_name);
    }
    if ( has_rate_enth ){
      m_deposition_models.push_back(rate_enth_name);
    }
    //---- end order hack ---

  }
}
