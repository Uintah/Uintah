#include <CCA/Components/Arches/LagrangianParticles/LagrangianParticleFactory.h>
#include <CCA/Components/Arches/LagrangianParticles/UpdateParticlePosition.h>
#include <CCA/Components/Arches/LagrangianParticles/UpdateParticleVelocity.h>
#include <CCA/Components/Arches/LagrangianParticles/UpdateParticleSize.h>
#include <CCA/Components/Arches/ArchesParticlesHelper.h>
#include <CCA/Components/Arches/Task/TaskInterface.h>

using namespace Uintah;

LagrangianParticleFactory::LagrangianParticleFactory( const ApplicationCommon* arches ) :
TaskFactoryBase(arches)
{
  _factory_name = "lagrangian_particle_factory";
}

LagrangianParticleFactory::~LagrangianParticleFactory()
{
}

void
LagrangianParticleFactory::register_all_tasks( ProblemSpecP& db )
{

  /*
   *

     <LagrangianParticles>
     <!-- requied and parsed by the ParticleHelper: -->
     <ParticlesPerCell>N</ParticlesPerCell>
     <MaximumParticles>N</MaximumParticles>
     <ParticlePosition x="lab" y="lab" z="lab"/>

     <!-- local to Arches implementation -->
     <ParticleVariables>
      <variable  label="name">            <<--    this is a variable that will be loaded onto the particle (eg, temperature, size..)
        <model label="modelname"/>        <<--    this is a model effecting the RHS of this variable
      </variable>
     </ParticlesVariables>


     </LagrangianParticles>

     <Initialization>
     <task label="particle_vel_init" type="lagrangian_particle_vel">
      <velocity_label u="label" v="label" w="label"/>
     </task>
     </Initialization>

   *
   */

  if ( db->findBlock("LagrangianParticles")) {

    ProblemSpecP db_lp = db->findBlock("LagrangianParticles");

    for ( ProblemSpecP db_pv = db_lp->findBlock("ParticleVariables")->findBlock("variable"); db_pv != nullptr; db_pv = db_pv->findNextBlock("variable") ) {

      std::string label;
      db_pv->getAttribute("label", label);

      Uintah::ArchesParticlesHelper::mark_for_relocation(label);
      Uintah::ArchesParticlesHelper::needs_boundary_condition(label);

    }

    //UPDATE PARTICLE POSITION
    std::string task_name = "update_particle_position";
    TaskInterface::TaskBuilder* tsk = scinew UpdateParticlePosition::Builder(task_name, 0);
    register_task( task_name, tsk );

    //UPDATE PARTICLE VELOCITY
    task_name = "update_particle_velocity";
    tsk = scinew UpdateParticleVelocity::Builder(task_name, 0);
    register_task( task_name, tsk );

    //UPDATE PARTICLE SIZE
    task_name = "update_particle_size";
    tsk = scinew UpdateParticleSize::Builder(task_name, 0);
    register_task( task_name, tsk );

  }
}

void
LagrangianParticleFactory::build_all_tasks( ProblemSpecP& db )
{

  if ( db->findBlock("LagrangianParticles")) {

    ProblemSpecP db_lp = db->findBlock("LagrangianParticles");

    TaskInterface* tsk = retrieve_task( "update_particle_velocity");
    print_task_setup_info( "update_particle_velocity", "lagrangian velocity update");
    tsk->problemSetup( db_lp );

    tsk = retrieve_task( "update_particle_position");
    print_task_setup_info( "update_particle_position", "lagrangian position update");
    tsk->problemSetup( db_lp );

    tsk = retrieve_task( "update_particle_size");
    print_task_setup_info( "update_particle_size", "lagrangian size update");
    tsk->problemSetup( db_lp );

    tsk->create_local_labels();

  }
}
