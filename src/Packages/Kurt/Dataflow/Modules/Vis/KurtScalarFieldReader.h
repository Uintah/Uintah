#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/ScalarFieldPort.h>
#include <Core/Datatypes/ScalarField.h>
#include <Core/Malloc/Allocator.h>
#include <Core/TclInterface/TCLTask.h>
#include <Core/TclInterface/TCLvar.h>

namespace Kurt {
using namespace SCIRun;


class Packages/KurtScalarFieldReader : public Module {
  ScalarFieldOPort* outport;
  ScalarFieldHandle handle;
  clString old_filebase;

  TCLstring tcl_status;
  TCLstring filebase; 
  TCLint animate;
  TCLint startFrame;
  TCLint endFrame;
  TCLint increment;

  bool read(const clString& fn);
  bool doAnimation();

public:
  Packages/KurtScalarFieldReader(const clString& id);
  virtual ~Packages/KurtScalarFieldReader();
  virtual void execute();
};
} // End namespace Kurt

