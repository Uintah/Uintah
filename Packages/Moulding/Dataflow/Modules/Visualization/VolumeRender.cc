/*
 *  VolumeRender.cc:
 *
 *  Written by:
 *   root
 *   TODAY'S DATE HERE
 *
 */

#include <Dataflow/Network/Module.h>
#include <Core/Malloc/Allocator.h>

#include <Packages/Moulding/share/share.h>

namespace Moulding {

using namespace SCIRun;

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
  : Module("VolumeRender", id, Source)
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

} // End namespace Moulding


