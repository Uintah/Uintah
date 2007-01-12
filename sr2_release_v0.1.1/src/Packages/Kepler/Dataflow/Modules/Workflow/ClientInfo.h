// license

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

namespace Kepler {

using namespace SCIRun;

class ClientInfo : public Module {
public:
  ClientInfo(GuiContext*);

  virtual ~ClientInfo();
  virtual void execute();
};

}
