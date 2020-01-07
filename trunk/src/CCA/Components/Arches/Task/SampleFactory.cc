#include <CCA/Components/Arches/Task/SampleFactory.h>
#include <CCA/Components/Arches/Task/SampleTask.h>
#include <CCA/Components/Arches/Task/TemplatedSampleTask.h>
#include <CCA/Components/Arches/Task/TaskInterface.h>

using namespace Uintah;

SampleFactory::SampleFactory( const ApplicationCommon* arches ) :
TaskFactoryBase( arches )
{}

SampleFactory::~SampleFactory()
{}

void
SampleFactory::register_all_tasks( ProblemSpecP& db )
{

  //This is meant to be the interface to the inputfile for the various tasks
  //associated with this factory.

  //The sample task:
  std::string tname = "sample_task";
  TaskInterface::TaskBuilder* sample_builder = scinew SampleTask::Builder(tname,0);
  register_task(tname, sample_builder);

  //The templated task:
  tname = "templated_task";
  TaskInterface::TaskBuilder* templated_sample_builder =
    scinew TemplatedSampleTask<CCVariable<double> >::Builder(tname,0);
  register_task(tname, templated_sample_builder);

}

void
SampleFactory::build_all_tasks( ProblemSpecP& db )
{

  //typedef std::vector<std::string> SV;

  TaskInterface* tsk1 = retrieve_task("sample_task");
  tsk1->problemSetup( db );
  tsk1->create_local_labels();

  TaskInterface* tsk2 = retrieve_task("templated_task");
  tsk2->problemSetup( db );
  tsk2->create_local_labels();

}
