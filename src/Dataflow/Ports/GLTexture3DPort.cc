
#include <Dataflow/Ports/GLTexture3DPort.h>
#include <Dataflow/share/share.h>
#include <Core/Malloc/Allocator.h>

namespace SCIRun {



extern "C" {
PSECORESHARE IPort* make_GLTexture3DIPort(Module* module,
					 const clString& name) {
  return scinew SimpleIPort<GLTexture3DHandle>(module,name);
}
PSECORESHARE OPort* make_GLTexture3DOPort(Module* module,
					 const clString& name) {
  return scinew SimpleOPort<GLTexture3DHandle>(module,name);
}
}

template<> clString SimpleIPort<GLTexture3DHandle>::port_type("GLTexture3D");
template<> clString SimpleIPort<GLTexture3DHandle>::port_color("gray40");


} // End namespace SCIRun

