#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/ScalarFieldPort.h>
#include <SCICore/Datatypes/ScalarField.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLTask.h>
#include <SCICore/TclInterface/TCLvar.h>

namespace Kurt {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace SCICore::TclInterface;
using namespace SCICore::PersistentSpace;


class KurtScalarFieldReader : public Module {
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
  KurtScalarFieldReader(const clString& id);
  virtual ~KurtScalarFieldReader();
  virtual void execute();
};

}
}
