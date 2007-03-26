

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

#include <Core/OS/sock.h>

#include <Dataflow/Modules/Salmon/OpenGL.h>

#include <Packages/Remote/Tools/macros.h>
#include <Packages/Remote/Tools/Image/ZImage.h>

#include <Packages/Remote/Dataflow/Modules/remoteSalmon/message.h>
#include <Packages/Remote/Dataflow/Modules/remoteSalmon/OpenGLServer.h>
#include <Packages/Remote/Dataflow/Modules/remoteSalmon/HeightSimp.h>
#include <Packages/Remote/Dataflow/Modules/remoteSalmon/RenderModel.h>

namespace Remote {
using namespace Remote::Tools;
using namespace SCIRun;

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
} // End namespace Remote


#endif // SCI_project_module_socketServer_h
