#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/ScalarFieldPort.h>
#include <Core/Datatypes/ScalarField.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/TCLTask.h>
#include <Core/GuiInterface/GuiVar.h>

namespace Kurt {
using namespace SCIRun;


class Packages/KurtScalarFieldReader : public Module {
  ScalarFieldOPort* outport;
  ScalarFieldHandle handle;
  clString old_filebase;

  GuiString tcl_status;
  GuiString filebase; 
  GuiInt animate;
  GuiInt startFrame;
  GuiInt endFrame;
  GuiInt increment;

  bool read(const clString& fn);
  bool doAnimation();

public:
  Packages/KurtScalarFieldReader(const clString& id);
  virtual ~Packages/KurtScalarFieldReader();
  virtual void execute();
};
} // End namespace Kurt

