/*
 *  SphericalSurface.cc:
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

class SphericalSurface : public Module {
public:
  SphericalSurface(GuiContext*);

  virtual ~SphericalSurface();

  virtual void execute();

  virtual void tcl_command(GuiArgs&, void*);
};


DECLARE_MAKER(SphericalSurface)
SphericalSurface::SphericalSurface(GuiContext* ctx)
  : Module("SphericalSurface", ctx, Source, "FieldsExample", "ModelCreation")
{
}

SphericalSurface::~SphericalSurface(){
}

void
 SphericalSurface::execute(){
}

void
 SphericalSurface::tcl_command(GuiArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace ModelCreation


