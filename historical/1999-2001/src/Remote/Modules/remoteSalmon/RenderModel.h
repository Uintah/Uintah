
//
// RenderModel.h
//

#ifndef __rendermodel_h__
#define __rendermodel_h__

#include <SCICore/OS/sock.h>
#include <Remote/Tools/Math/Vector.h>

namespace Remote {
namespace Modules {

using namespace Remote::Tools;
using Remote::Tools::Vector;
using SCICore::OS::Socket;

void RenderModel(Model &M);
void LoadTexture(int TexInd);
void LoadTextures(Model &M);
void sendModel(Model& M, Socket* sock);
Model* receiveModel(Socket* sock);

struct view {
  Vector pos;
  Vector lookat;
};

} // namespace Modules
} // namespace Remote

#endif // __rendermodel_h__
