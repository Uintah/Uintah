#include <Packages/Uintah/Core/Grid/GeometryPieceFactory.h>
#include <Packages/Uintah/Core/Grid/BoxGeometryPiece.h>
#include <Packages/Uintah/Core/Grid/SphereGeometryPiece.h>
#include <Packages/Uintah/Core/Grid/SphereMembraneGeometryPiece.h>
#include <Packages/Uintah/Core/Grid/CylinderGeometryPiece.h>
#include <Packages/Uintah/Core/Grid/TriGeometryPiece.h>
#include <Packages/Uintah/Core/Grid/UnionGeometryPiece.h>
#include <Packages/Uintah/Core/Grid/DifferenceGeometryPiece.h>
#include <Packages/Uintah/Core/Grid/IntersectionGeometryPiece.h>

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>

#include <iostream>
#include <string>

using std::cerr;
using std::endl;

using namespace Uintah;

void GeometryPieceFactory::create(const ProblemSpecP& ps,
				  std::vector<GeometryPiece*>& objs)

{
   for(ProblemSpecP child = ps->findBlock(); child != 0;
       child = child->findNextBlock()){
      std::string go_type = child->getNodeName();
      if (go_type == "box")
	 objs.push_back(new BoxGeometryPiece(child));
      
      else if (go_type == "sphere")
	 objs.push_back(new SphereGeometryPiece(child));

      else if (go_type == "sphere_membrane")
	 objs.push_back(new SphereMembraneGeometryPiece(child));

      else if (go_type ==  "cylinder")
	 objs.push_back(new CylinderGeometryPiece(child));

      else if (go_type == "tri")
	 objs.push_back(new TriGeometryPiece(child));
 
      else if (go_type == "union")
	 objs.push_back(new UnionGeometryPiece(child));
   
      else if (go_type == "difference")
	 objs.push_back(new DifferenceGeometryPiece(child));

      else if (go_type == "intersection")
	 objs.push_back(new IntersectionGeometryPiece(child));
      else if (go_type == "res" || go_type == "velocity" || go_type == "temperature") {
	 // Ignore...
      
      } else {
	if (ps->doWriteMessages())
	  cerr << "Unknown Geometry Piece Type " << "(" << go_type << ")" 
	       << endl;
	//	exit(1);
      }
   }
}
