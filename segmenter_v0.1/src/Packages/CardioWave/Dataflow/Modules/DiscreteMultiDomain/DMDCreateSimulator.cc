/*
 *  DMDCreateSimulator.cc:
 *
 *  Written by:
 *   jeroen
 *   TODAY'S DATE HERE
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

namespace CardioWave {

using namespace SCIRun;

class DMDCreateSimulator : public Module {
public:
  DMDCreateSimulator(GuiContext*);

  virtual ~DMDCreateSimulator();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);
};


DECLARE_MAKER(DMDCreateSimulator)
DMDCreateSimulator::DMDCreateSimulator(GuiContext* ctx)
  : Module("DMDCreateSimulator", ctx, Source, "DiscreteMultiDomain", "CardioWave")
{
}

DMDCreateSimulator::~DMDCreateSimulator(){
}

void
 DMDCreateSimulator::execute(){
}

void
 DMDCreateSimulator::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace CardioWave


