#include <Core/GeometryPiece/ShellGeometryFactory.h>
#include <Core/GeometryPiece/PlaneShellPiece.h>
#include <Core/GeometryPiece/SphereShellPiece.h>
#include <Core/GeometryPiece/CylinderShellPiece.h>
#include <Core/GeometryPiece/GUVSphereShellPiece.h>

#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/ProblemSpec/ProblemSpecP.h>

#include <iostream>
#include <string>

using std::cerr;
using std::endl;

using namespace Uintah;

GeometryPiece *
ShellGeometryFactory::create( ProblemSpecP & ps )
{
  std::string go_type = ps->getNodeName();

  if (     go_type == PlaneShellPiece::TYPE_NAME ) {
    return scinew PlaneShellPiece(ps);
  }
  else if( go_type == SphereShellPiece::TYPE_NAME ) {
    return scinew SphereShellPiece(ps);
  }
  else if (go_type == CylinderShellPiece::TYPE_NAME ) {
    return scinew CylinderShellPiece(ps);
  }
  else if (go_type == GUVSphereShellPiece::TYPE_NAME ) {
    return scinew GUVSphereShellPiece(ps);
  }
  return NULL;
}
