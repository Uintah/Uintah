/*
 *  Test.cc:
 *
 *  Written by:
 *   allen
 *   TODAY'S DATE HERE
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

#include <Packages/CollabVis/share/share.h>

namespace CollabVis {

using namespace SCIRun;

class CollabVisSHARE Test : public Module {
public:
  Test(GuiContext*);

  virtual ~Test();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);
};


DECLARE_MAKER(Test)
Test::Test(GuiContext* ctx)
  : Module("Test", ctx, Source, "Render", "CollabVis")
{
}

Test::~Test(){
}

void
 Test::execute(){
}

void
 Test::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace CollabVis


