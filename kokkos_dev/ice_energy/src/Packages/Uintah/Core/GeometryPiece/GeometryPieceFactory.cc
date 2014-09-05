#include <Packages/Uintah/Core/GeometryPiece/GeometryPieceFactory.h>
#include <Packages/Uintah/Core/GeometryPiece/ShellGeometryFactory.h>
#include <Packages/Uintah/Core/GeometryPiece/BoxGeometryPiece.h>
#include <Packages/Uintah/Core/GeometryPiece/SphereGeometryPiece.h>
#include <Packages/Uintah/Core/GeometryPiece/SphereMembraneGeometryPiece.h>
#include <Packages/Uintah/Core/GeometryPiece/CylinderGeometryPiece.h>
#include <Packages/Uintah/Core/GeometryPiece/SmoothCylGeomPiece.h>
#include <Packages/Uintah/Core/GeometryPiece/CorrugEdgeGeomPiece.h>
#include <Packages/Uintah/Core/GeometryPiece/ConeGeometryPiece.h>
#include <Packages/Uintah/Core/GeometryPiece/TriGeometryPiece.h>
#include <Packages/Uintah/Core/GeometryPiece/UnionGeometryPiece.h>
#include <Packages/Uintah/Core/GeometryPiece/DifferenceGeometryPiece.h>
#include <Packages/Uintah/Core/GeometryPiece/IntersectionGeometryPiece.h>
#include <Packages/Uintah/Core/GeometryPiece/FileGeometryPiece.h>
#include <Packages/Uintah/Core/GeometryPiece/NullGeometryPiece.h>
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
               go_type == "temperature" || go_type == "#comment")  {
        // Ignoring. 
        continue;    // restart loop to avoid accessing name of empty object
      
      } else {
	if (ps->doWriteMessages())
	  cerr << "WARNING: Unknown Geometry Piece Type " << "(" << go_type << ")" 
	       << endl;
        continue;    // restart loop to avoid accessing name of empty object
      }
      string name;
      if(child->getAttribute("name", name)){
	objs[objs.size()-1]->setName(name);
      }
   }
}
