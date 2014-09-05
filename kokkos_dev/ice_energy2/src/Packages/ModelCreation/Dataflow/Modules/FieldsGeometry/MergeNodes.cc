/*
 *  MergeNodes.cc:
 *
 *  Written by:
 *   jeroen
 *   TODAY'S DATE HERE
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

namespace ModelCreation {

using namespace SCIRun;

class MergeNodes : public Module {
public:
  MergeNodes(GuiContext*);

  virtual ~MergeNodes();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);
};


DECLARE_MAKER(MergeNodes)
MergeNodes::MergeNodes(GuiContext* ctx)
  : Module("MergeNodes", ctx, Source, "FieldsGeometry", "ModelCreation")
{
}

MergeNodes::~MergeNodes(){
}

void
 MergeNodes::execute(){
}

void
 MergeNodes::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace ModelCreation


