/*
 *  VolumeRender.cc:
 *
 *  Written by:
 *   root
 *   TODAY'S DATE HERE
 *
 */

#include <PSECore/Dataflow/Module.h>
#include <SCICore/Malloc/Allocator.h>

#include <Moulding/share/share.h>

namespace Moulding {
namespace Modules {

using namespace PSECore::Dataflow;

class MouldingSHARE VolumeRender : public Module {
public:
  VolumeRender(const clString& id);

  virtual ~VolumeRender();

  virtual void execute();

  virtual void tcl_command(TCLArgs&, void*);
};

extern "C" MouldingSHARE Module* make_VolumeRender(const clString& id) {
  return new VolumeRender(id);
}

VolumeRender::VolumeRender(const clString& id)
  : Module("VolumeRender", id, Source,"Visualization","Moulding")
{
}

VolumeRender::~VolumeRender(){
}

void VolumeRender::execute(){
}

void VolumeRender::tcl_command(TCLArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace Modules
} // End namespace Moulding


