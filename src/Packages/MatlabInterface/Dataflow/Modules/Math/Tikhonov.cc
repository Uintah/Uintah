/*
 *  Tikhonov.cc:
 *
 *  Written by:
 *   oleg
 *   TODAY'S DATE HERE
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

#include <Packages/MatlabInterface/share/share.h>

namespace MatlabInterface {

using namespace SCIRun;

class MatlabInterfaceSHARE Tikhonov : public Module {
public:
  Tikhonov(const string& id);

  virtual ~Tikhonov();

  virtual void execute();

  virtual void tcl_command(TCLArgs&, void*);
};

extern "C" MatlabInterfaceSHARE Module* make_Tikhonov(const string& id) {
  return scinew Tikhonov(id);
}

Tikhonov::Tikhonov(const string& id)
  : Module("Tikhonov", id, Source, "Math", "MatlabInterface")
{
}

Tikhonov::~Tikhonov(){
}

void Tikhonov::execute(){
}

void Tikhonov::tcl_command(TCLArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace MatlabInterface


