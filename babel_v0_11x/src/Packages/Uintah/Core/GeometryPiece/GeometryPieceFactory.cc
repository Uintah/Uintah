#include <Packages/Uintah/Core/GeometryPiece/GeometryPieceFactory.h>

#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/GeometryPiece/ShellGeometryFactory.h>
#include <Packages/Uintah/Core/GeometryPiece/BoxGeometryPiece.h>
#include <Packages/Uintah/Core/GeometryPiece/NaaBoxGeometryPiece.h>
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

#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpecP.h>
#include <Packages/Uintah/Core/Parallel/Parallel.h>

#include <Core/Malloc/Allocator.h>
#include <Core/Util/DebugStream.h>

#include <sgi_stl_warnings_off.h>
#include   <iostream>
#include   <string>
#include <sgi_stl_warnings_on.h>

using namespace Uintah;
using namespace std;

DebugStream dbg( "GeometryPieceFactory", false );

// Static class variable definition:
map<string,GeometryPiece*> GeometryPieceFactory::namedPieces_;

void
GeometryPieceFactory::create( const ProblemSpecP& ps,
                              vector<GeometryPiece*>& objs)
{
   for(ProblemSpecP child = ps->findBlock(); child != 0;
       child = child->findNextBlock()){
      string go_type = child->getNodeName();
      string go_label;
      if( !child->getAttribute( "label", go_label ) ) {
        child->getAttribute( "name", go_label );
      }

      if( go_label != "" ) {
        dbg << "Looking at GeometryPiece (" << go_type << ") labeled: " << go_label << "\n";
      }

      if( go_label != "" ) {

        ProblemSpecP childBlock = child->findBlock();
        GeometryPiece * gp = namedPieces_[go_label];

        if( gp && childBlock ) {
          cout << "Error: GeometryPiece (" << go_type << ")" << " labeled: '" << go_label 
               << "' has already been specified...  You can't change its values.\n"
               << "Please just reference the original by only using the label (no values)\n";
          throw ProblemSetupException("Duplicate GeomPiece definition not allowed",
                                      __FILE__, __LINE__);
        }

        if( !childBlock ) {
          dbg << "Referencing already created GeomPiece: " << go_label << "\n";
          if( gp != NULL ) {
            objs.push_back( gp );
          } else {
            cout << "Error... couldn't find the referenced GeomPiece (" << go_type 
                 << ")" << " labeled '" << go_label 
                 << "'!\n";
            throw ProblemSetupException("Referenced GeomPiece does not exist",
                                        __FILE__, __LINE__);
          }
          return;
        }
      }

      GeometryPiece * newGeomPiece = NULL;
      if (go_type == "shell") 
        ShellGeometryFactory::create(child, objs);
      
      else if (go_type == "box")
        newGeomPiece = scinew BoxGeometryPiece(child);

      else if (go_type == "parallelepiped")
        newGeomPiece = scinew NaaBoxGeometryPiece(child);
      
      else if (go_type == "sphere")
        newGeomPiece = scinew SphereGeometryPiece(child);

      else if (go_type == "sphere_membrane")
        newGeomPiece = scinew SphereMembraneGeometryPiece(child);

      else if (go_type ==  "cylinder")
        newGeomPiece = scinew CylinderGeometryPiece(child);

      else if (go_type ==  "smoothcyl")
        newGeomPiece = scinew SmoothCylGeomPiece(child);

      else if (go_type ==  "corrugated")
        newGeomPiece = scinew CorrugEdgeGeomPiece(child);

      else if (go_type ==  "cone")
        newGeomPiece = scinew ConeGeometryPiece(child);

      else if (go_type == "tri")
        newGeomPiece = scinew TriGeometryPiece(child);
 
      else if (go_type == "union")
        newGeomPiece = scinew UnionGeometryPiece(child);
   
      else if (go_type == "difference")
        newGeomPiece = scinew DifferenceGeometryPiece(child);

      else if (go_type == "file")
        newGeomPiece = scinew FileGeometryPiece(child);

      else if (go_type == "intersection")
        newGeomPiece = scinew IntersectionGeometryPiece(child);

      else if (go_type == "null")
	newGeomPiece = scinew NullGeometryPiece(child);

      else if (go_type == "res" || go_type == "velocity" || 
               go_type == "temperature" || go_type == "#comment")  {
        // Ignoring. 
        continue;    // restart loop to avoid accessing name of empty object
      
      } else {
	if (ps->doWriteMessages() && Parallel::getMPIRank() == 0)
	  cerr << "WARNING: Unknown Geometry Piece Type " << "(" << go_type << ")" 
	       << endl;
        continue;    // restart loop to avoid accessing name of empty object
      }
      // Look for the "name" of the object.  (Can also be referenced as "label").
      string name;
      if(child->getAttribute("name", name)){
	newGeomPiece->setName(name);
      } else if(child->getAttribute("label", name)){
	newGeomPiece->setName(name);
      }
      if( name != "" ) {
        namedPieces_[name] = newGeomPiece;
      }
      objs.push_back( newGeomPiece );
   }
}
