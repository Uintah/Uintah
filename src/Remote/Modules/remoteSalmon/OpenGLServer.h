
/*
 *  OpenGLServer.h: add a network daemon to the opengl renderer
 *
 *  Written by:
 *   David Hart
 *   Department of Computer Science
 *   University of Utah
 *
 *  Copyright (C) 2000 SCI Group
 */

#ifndef SCI_project_module_OpenGLServer_h
#define SCI_project_module_OpenGLServer_h

#include <SCICore/Thread/Thread.h>

#include <PSECommon/Modules/Salmon/OpenGL.h>

#include <Remote/Modules/remoteSalmon/socketServer.h>

#define DO_GETGLSTATE 6

namespace Remote {
namespace Modules {

using namespace SCICore::Thread;
using namespace SCICore::GeomSpace;
using PSECore::Datatypes::GeometryData;
using namespace PSECommon::Modules;


class socketServer;
  
//----------------------------------------------------------------------
class OpenGLServer : public OpenGL {
public:
  socketServer* socketserver;
  Thread* socketserverthread;
  Mailbox<void*> gl_in_mb;
  Mailbox<void*> gl_out_mb;
  
  OpenGLServer();
  void collect_triangles(Salmon* salmon, Roe* roe, GeomObj* obj);
  virtual void real_getData(int datamask, FutureValue<GeometryData*>* result);
  virtual void redraw_loop();
  virtual void getGLState(int which);
};

} // namespace Modules
} // namespace Remote

#endif // SCI_project_module_OpenGLServer_h
