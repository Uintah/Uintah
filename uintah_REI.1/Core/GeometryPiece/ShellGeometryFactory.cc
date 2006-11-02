#include <Packages/Uintah/Core/GeometryPiece/ShellGeometryFactory.h>
#include <Packages/Uintah/Core/GeometryPiece/PlaneShellPiece.h>
#include <Packages/Uintah/Core/GeometryPiece/SphereShellPiece.h>
#include <Packages/Uintah/Core/GeometryPiece/CylinderShellPiece.h>
#include <Packages/Uintah/Core/GeometryPiece/GUVSphereShellPiece.h>

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

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
