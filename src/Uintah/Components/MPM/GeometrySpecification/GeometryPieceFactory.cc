#include "GeometryPieceFactory.h"
#include "BoxGeometryPiece.h"
#include "SphereGeometryPiece.h"
#include "CylinderGeometryPiece.h"
#include "TriGeometryPiece.h"
#include "UnionGeometryPiece.h"
#include "DifferenceGeometryPiece.h"
#include "IntersectionGeometryPiece.h"
#include <Uintah/Interface/ProblemSpec.h>
#include <Uintah/Interface/ProblemSpecP.h>
#include <iostream>
#include <string>

using std::cerr;
using std::endl;

using namespace Uintah::Components;
using Uintah::Interface::ProblemSpecP;

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
      
      else {
	cerr << "Unknown Geometry Piece Type " << go_type << endl;
	//	exit(1);
      }
   }
}
