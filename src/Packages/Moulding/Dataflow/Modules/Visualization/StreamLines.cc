/*
 *  StreamLines.cc:
 *
 *  Written by:
 *   moulding
 *   TODAY'S DATE HERE
 *
 */

#include <PSECore/Dataflow/Module.h>
#include <SCICore/Malloc/Allocator.h>

#include <Moulding/share/share.h>

namespace Moulding {
namespace Modules {

using namespace PSECore::Dataflow;

class MouldingSHARE StreamLines : public Module {
public:
  StreamLines(const clString& id);

  virtual ~StreamLines();

  virtual void execute();

  virtual void tcl_command(TCLArgs&, void*);
};

extern "C" MouldingSHARE Module* make_StreamLines(const clString& id) {
  return scinew StreamLines(id);
}

StreamLines::StreamLines(const clString& id)
  : Module("StreamLines", id, Source, "Visualization", "Moulding")
{
}

StreamLines::~StreamLines(){
}

void StreamLines::execute(){
}

void StreamLines::tcl_command(TCLArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace Modules
} // End namespace Moulding


