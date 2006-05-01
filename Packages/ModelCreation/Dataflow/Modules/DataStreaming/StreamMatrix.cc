/*
 *  StreamMatrix.cc:
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

class StreamMatrix : public Module {
public:
  StreamMatrix(GuiContext*);

  virtual ~StreamMatrix();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);
};


DECLARE_MAKER(StreamMatrix)
StreamMatrix::StreamMatrix(GuiContext* ctx)
  : Module("StreamMatrix", ctx, Source, "DataStreaming", "ModelCreation")
{
}

StreamMatrix::~StreamMatrix(){
}

void
 StreamMatrix::execute(){
}

void
 StreamMatrix::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace ModelCreation


