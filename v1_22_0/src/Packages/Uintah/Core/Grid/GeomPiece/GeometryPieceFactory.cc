#include <Packages/Uintah/Core/Grid/GeomPiece/GeometryPieceFactory.h>
#include <Packages/Uintah/Core/Grid/GeomPiece/ShellGeometryFactory.h>
#include <Packages/Uintah/Core/Grid/GeomPiece/BoxGeometryPiece.h>
#include <Packages/Uintah/Core/Grid/GeomPiece/SphereGeometryPiece.h>
#include <Packages/Uintah/Core/Grid/GeomPiece/SphereMembraneGeometryPiece.h>
#include <Packages/Uintah/Core/Grid/GeomPiece/CylinderGeometryPiece.h>
#include <Packages/Uintah/Core/Grid/GeomPiece/SmoothCylGeomPiece.h>
#include <Packages/Uintah/Core/Grid/GeomPiece/CorrugEdgeGeomPiece.h>
#include <Packages/Uintah/Core/Grid/GeomPiece/ConeGeometryPiece.h>
#include <Packages/Uintah/Core/Grid/GeomPiece/TriGeometryPiece.h>
#include <Packages/Uintah/Core/Grid/GeomPiece/UnionGeometryPiece.h>
#include <Packages/Uintah/Core/Grid/GeomPiece/DifferenceGeometryPiece.h>
#include <Packages/Uintah/Core/Grid/GeomPiece/IntersectionGeometryPiece.h>
#include <Packages/Uintah/Core/Grid/GeomPiece/FileGeometryPiece.h>
#include <Packages/Uintah/Core/Grid/GeomPiece/NullGeometryPiece.h>
#include <Core/Malloc/Allocator.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <string>
#include <sgi_stl_warnings_on.h>

using std::cerr;
using std::endl;

using namespace Uintah;

void GeometryPieceFactory::create(const ProblemSpecP& ps,
				  std::vector<GeometryPiece*>& objs)

{
   for(ProblemSpecP child = ps->findBlock(); child != 0;
       child = child->findNextBlock()){
      std::string go_type = child->getNodeName();
      
      if (go_type == "shell") 
        ShellGeometryFactory::create(child, objs);
      
      else if (go_type == "box")
	 objs.push_back(scinew BoxGeometryPiece(child));
      
      else if (go_type == "sphere")
	 objs.push_back(scinew SphereGeometryPiece(child));

      else if (go_type == "sphere_membrane")
	 objs.push_back(scinew SphereMembraneGeometryPiece(child));

      else if (go_type ==  "cylinder")
	 objs.push_back(scinew CylinderGeometryPiece(child));

      else if (go_type ==  "smoothcyl")
	 objs.push_back(scinew SmoothCylGeomPiece(child));

      else if (go_type ==  "corrugated")
	 objs.push_back(scinew CorrugEdgeGeomPiece(child));

      else if (go_type ==  "cone")
	 objs.push_back(scinew ConeGeometryPiece(child));

      else if (go_type == "tri")
	 objs.push_back(scinew TriGeometryPiece(child));
 
      else if (go_type == "union")
	 objs.push_back(scinew UnionGeometryPiece(child));
   
      else if (go_type == "difference")
	 objs.push_back(scinew DifferenceGeometryPiece(child));

      else if (go_type == "file")
	 objs.push_back(scinew FileGeometryPiece(child));

      else if (go_type == "intersection")
	 objs.push_back(scinew IntersectionGeometryPiece(child));

      else if (go_type == "null")
	objs.push_back(scinew NullGeometryPiece(child));

      else if (go_type == "res" || go_type == "velocity" || 
               go_type == "temperature") {
	 // Ignore...
      
      } else {
	if (ps->doWriteMessages())
	  cerr << "Unknown Geometry Piece Type " << "(" << go_type << ")" 
	       << endl;
	//	exit(1);
      }
      string name;
      if(child->getAttribute("name", name)){
	objs[objs.size()-1]->setName(name);
      }
   }
}
