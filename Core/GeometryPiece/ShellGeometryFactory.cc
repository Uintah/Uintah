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
ShellGeometryFactory::create( const ProblemSpecP& ps )
{
  for(ProblemSpecP child = ps->findBlock(); child != 0; child = child->findNextBlock()) {

    std::string go_type = child->getNodeName();

    if (     go_type == PlaneShellPiece::TYPE_NAME ) {
      return scinew PlaneShellPiece(child);
    }
    else if( go_type == SphereShellPiece::TYPE_NAME ) {
      return scinew SphereShellPiece(child);
    }
    else if (go_type == CylinderShellPiece::TYPE_NAME ) {
      return scinew CylinderShellPiece(child);
    }
    else if (go_type == GUVSphereShellPiece::TYPE_NAME ) {
      return scinew GUVSphereShellPiece(child);
    }
    else {
      if (ps->doWriteMessages()) {
	cerr << "Unknown Shell Geometry Piece Type " << "(" << go_type << ")\n";
      }
      break;
    }
  }
  return NULL;
}
