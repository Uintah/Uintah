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

#include <Dataflow/Ports/ScalarFieldPort.h>

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
  return scinew VolumeRender(id);
}

VolumeRender::VolumeRender(const clString& id)
  : Module("VolumeRender", id, Source,"Visualization","Moulding")
{
}

VolumeRender::~VolumeRender()
{
}

void VolumeRender::execute()
{
  ScalarFieldIPort* volume_port = (ScalarFieldIPort*)get_iport("Volume");
  if (!volume_port) {
    std::cerr << "VolumeRender: unable to get port named \"Volume\"." << endl;
    return;
  }

  ScalarFieldHandle volume;
  if (!volume_port->get(volume)) {
    std::cerr << "VolumeRender: no data for port named \"Volume\"." << endl;
    return;
  }
  
  std::cerr << "VolumeRender: found a scalar field on port named \"Volume\"!" 
            << endl;
}

void VolumeRender::tcl_command(TCLArgs& args, void* userdata)
{
  Module::tcl_command(args, userdata);
}

} // End namespace Moulding


