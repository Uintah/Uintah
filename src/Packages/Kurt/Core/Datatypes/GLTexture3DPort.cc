#include <Packages/Kurt/Core/Datatypes/GLTexture3DPort.h>
#include <Packages/Kurt/share/share.h>
#include <Core/Malloc/Allocator.h>

using namespace SCIRun;
using namespace Kurt;

extern "C" {

KurtSHARE IPort*
make_GLTexture3DIPort(Module* module,
		      const clString& name) {
  return scinew SimpleIPort<GLTexture3DHandle>(module,name);
}

KurtSHARE OPort*
make_GLTexture3DOPort(Module* module,
		      const clString& name) {
  return scinew SimpleOPort<GLTexture3DHandle>(module,name);
}
}
template<> clString SimpleIPort<GLTexture3DHandle>::port_type("GLTexture3D");
template<> clString SimpleIPort<GLTexture3DHandle>::port_color("gray40");


