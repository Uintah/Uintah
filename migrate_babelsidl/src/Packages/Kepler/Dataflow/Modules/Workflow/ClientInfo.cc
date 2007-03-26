// license

#include <Packages/Kepler/Dataflow/Modules/Workflow/ClientInfo.h>

namespace Kepler {

DECLARE_MAKER(ClientInfo)
ClientInfo::ClientInfo(GuiContext* ctx)
  : Module("ClientInfo", ctx, Sink, "Workflow", "Kepler")
{
}

ClientInfo::~ClientInfo()
{
}

void ClientInfo::execute()
{
}


}
