

/*
 *  socketServer.h:
 *
 *  Written by:
 *   David Hart
 *   Department of Computer Science
 *   University of Utah
 *   May 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#ifndef SCI_project_module_socketServer_h
#define SCI_project_module_socketServer_h

#include <SCICore/OS/sock.h>

#include <PSECommon/Modules/Salmon/OpenGL.h>

#include <Remote/Modules/remoteSalmon/OpenGLServer.h>
#include <Remote/Modules/remoteSalmon/message.h>
#include <Remote/Modules/remoteSalmon/ZImage.h>
#include <Remote/Modules/remoteSalmon/macros.h>
#include <Remote/Modules/remoteSalmon/HeightSimp.h>
#include <Remote/Modules/remoteSalmon/Vector.h>
#include <Remote/Modules/remoteSalmon/RenderModel.h>

namespace Remote {
namespace Modules {

using SCICore::Thread::Runnable;
using SCICore::Thread::Thread;
using SCICore::OS::Socket;

class OpenGLServer;

//----------------------------------------------------------------------
class socketServer : public Runnable {
  
  OpenGLServer* opengl;

  Socket* listensock;
  vector<Socket*> socks;
  
  void addConnection(Socket* sock);
  void removeConnection(Socket* sock);
  
  void sendView(Socket* sock);
  void receiveView(Socket* sock);
  void sendMesh(Socket* sock);
  void sendZBuffer(Socket* sock);
  void sendGeom(Socket* sock);

public:

  bool serverRunning;
  
  socketServer(OpenGLServer* opengl);
  virtual ~socketServer();
  virtual void run();
  
};

} // namespace Modules
} // namespace Remote

#endif // SCI_project_module_socketServer_h
