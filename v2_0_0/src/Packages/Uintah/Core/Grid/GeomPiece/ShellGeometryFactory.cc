#include <Packages/Uintah/Core/Grid/GeomPiece/ShellGeometryFactory.h>
#include <Packages/Uintah/Core/Grid/GeomPiece/PlaneShellPiece.h>
#include <Packages/Uintah/Core/Grid/GeomPiece/SphereShellPiece.h>
#include <Packages/Uintah/Core/Grid/GeomPiece/CylinderShellPiece.h>

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

#include <iostream>
#include <string>

using std::cerr;
using std::endl;

using namespace Uintah;

void 
ShellGeometryFactory::create(const ProblemSpecP& ps,
			     std::vector<GeometryPiece*>& objs)

{
  for(ProblemSpecP child = ps->findBlock(); child != 0;
      child = child->findNextBlock()) {
    std::string go_type = child->getNodeName();
    if (go_type == "plane")
      objs.push_back(new PlaneShellPiece(child));
      
    else if (go_type == "sphere")
      objs.push_back(new SphereShellPiece(child));

    else if (go_type == "cylinder")
      objs.push_back(new CylinderShellPiece(child));

    else 
      if (ps->doWriteMessages())
	cerr << "Unknown Shell Geometry Piece Type " << "(" << go_type << ")" 
	     << endl;
    string name;
    if(child->getAttribute("name", name))
      objs[objs.size()-1]->setName(name);
  }
}
