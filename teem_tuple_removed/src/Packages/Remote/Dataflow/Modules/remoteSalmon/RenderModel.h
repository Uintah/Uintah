
// RenderModel.h

#ifndef __rendermodel_h__
#define __rendermodel_h__

#include <Core/OS/sock.h>
#include <Packages/Remote/Tools/Math/Vector.h>

namespace Remote {
using namespace Remote::Tools;
using Remote::Tools::Vector;
using namespace SCIRun;

void RenderModel(Model &M);
void LoadTexture(int TexInd);
void LoadTextures(Model &M);
void sendModel(Model& M, Socket* sock);
Model* receiveModel(Socket* sock);

struct view {
  Vector pos;
  Vector lookat;
};
} // End namespace Remote


#endif // __rendermodel_h__
