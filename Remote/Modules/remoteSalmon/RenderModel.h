
//
// RenderModel.h
//

#ifndef __rendermodel_h__
#define __rendermodel_h__

#include <SCICore/OS/sock.h>

#include <Remote/Modules/remoteSalmon/Vector.h>

using SCICore::OS::Socket;

void RenderModel(Model &M);
void LoadTexture(int TexInd);
void LoadTextures(Model &M);
void sendModel(Model& M, Socket* sock);
Model* receiveModel(Socket* sock);

struct view {
  dVector pos;
  dVector dir;
};

#endif // __rendermodel_h__
