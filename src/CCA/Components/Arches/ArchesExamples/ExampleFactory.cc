#include <CCA/Components/Arches/ArchesExamples/ExampleFactory.h>
#include <CCA/Components/Arches/Task/TaskInterface.h>
//Add example includes here
#include <CCA/Components/Arches/ArchesExamples/Poisson1.h>

using namespace Uintah;
using namespace Uintah::ArchesExamples;

ExampleFactory::ExampleFactory( const ApplicationCommon* arches ) :
TaskFactoryBase(arches){
  _factory_name = "ExampleFactory";
}

ExampleFactory::~ExampleFactory(){}

void
ExampleFactory::register_all_tasks( ProblemSpecP& db ){
  ProblemSpecP exe = db->getRootNode()->findBlock("CFD")->findBlock("ARCHES")->findBlock("ArchesExample");
  if(exe){
    std::string type;
    exe->getAttribute("type", type);

    //4. Update register_all_tasks to create and register the new arches task
    //------------------------------------- poisson1 example ---------------------------------------
    if(type=="poisson1"){
      TaskInterface::TaskBuilder* tsk = scinew Poisson1::Builder( type, 0 );
      _poisson1_tasks.push_back(type);
      register_task( type, tsk, db );
    }
    //----------------------------------------------------------------------------------------------

  }
}

