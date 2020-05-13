#include <CCA/Components/Arches/ArchesExamples/ExampleFactory.h>
#include <CCA/Components/Arches/Task/TaskInterface.h>
//Add example includes here
#include <CCA/Components/Arches/ArchesExamples/Poisson1.h>


using namespace Uintah::ArchesExamples;

ExampleFactory::ExampleFactory( const ApplicationCommon* arches ) :
TaskFactoryBase(arches)
{
  _factory_name = "ExampleFactory";
}

ExampleFactory::~ExampleFactory()
{}

void
ExampleFactory::register_all_tasks( ProblemSpecP& db )
{

  ProblemSpecP exe = db->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("ArchesExample");
  if(exe){
	  std::string type;
	  exe->getAttribute("type", type);
	  if(type=="poisson1"){
		  TaskInterface::TaskBuilder* tsk = scinew Poisson1::Builder( type, 0 );
		  _poisson1_tasks.push_back(type);
		  register_task( type, tsk, db );
	  }
  }


}

void
ExampleFactory::add_task( ProblemSpecP& db ){

}

void
ExampleFactory::build_all_tasks( ProblemSpecP& db )
{
}


//--------------------------------------------------------------------------------------------------
void ExampleFactory::schedule_initialization( const LevelP& level,
                                                 SchedulerP& sched,
                                                 const MaterialSet* matls,
                                                 bool doing_restart ){

  const bool pack_tasks = false;
  schedule_task_group( "all_tasks", m_task_init_order, TaskInterface::INITIALIZE,
                       pack_tasks, level, sched, matls );
}
