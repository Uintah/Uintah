#include "GeometryObjectFactory.h"
#include "BoxGeometryObject.h"
#include "SphereGeometryObject.h"
#include "CylinderGeometryObject.h"
#include "TriGeometryObject.h"
#include "UnionGeometryObject.h"
#include "DifferenceGeometryObject.h"
#include "IntersectionGeometryObject.h"
#include <Uintah/Interface/ProblemSpec.h>
#include <Uintah/Interface/ProblemSpecP.h>
#include <iostream>
#include <string>

using std::cerr;
using std::endl;

using namespace Uintah::Components;
using Uintah::Interface::ProblemSpecP;

void GeometryObjectFactory::create(const ProblemSpecP& ps,
				   std::vector<GeometryObject*>& objs)

{
   for(ProblemSpecP child = ps->findBlock(); child != 0;
       child = child->findNextBlock()){
      std::string go_type = child->getNodeName();
      if (go_type == "box")
	 objs.push_back(new BoxGeometryObject(child));
      
      else if (go_type == "sphere")
	 objs.push_back(new SphereGeometryObject(child));

      else if (go_type ==  "cylinder")
	 objs.push_back(new CylinderGeometryObject(child));

      else if (go_type == "tri")
	 objs.push_back(new TriGeometryObject(child));
 
      else if (go_type == "union")
	 objs.push_back(new UnionGeometryObject(child));
   
      else if (go_type == "difference")
	 objs.push_back(new DifferenceGeometryObject(child));

      else if (go_type == "intersection")
	 objs.push_back(new IntersectionGeometryObject(child));
      
      else {
	cerr << "Unknown Geometry Object Type " << go_type << endl;
	//	exit(1);
      }
   }
}
