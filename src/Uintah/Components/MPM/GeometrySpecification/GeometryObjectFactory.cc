#include "GeometryObjectFactory.h"
#include <fstream>
#include <iostream>
#include <string>
using std::cerr;
using std::ifstream;
using std::ofstream;

using namespace Uintah::Components;

#ifdef WONT_COMPILE_YET
void GeometryObjectFactory::create(ProblemSpecP& ps,
				   vector<GeometryObject*>& objs)

{
   for(ProblemSpecP child = ps->findBlock(); child != 0;
       child = child->findNextBlock()){
      string go_type = child->getNodeName();
      if (go_type == "box")
	 objs.push_back(new BoxGeometryObject(ps));
      
      else if (go_type == "sphere")
	 return new SphereGeometryObject(ps);

      else if (go_type ==  "cylinder")
	 return new CylinderGeometryObject(ps);

      else if (go_type == "tri")
	 return new TriGeometryObjectPlas(ps);
 
      else if (go_type == "union")
	 return new UnionGeometryObject(ps);
   
      else if (go_type == "difference")
	 return new DifferenceGeometryObject(ps);

      else if (go_type == "instersection")
	 return new IntersectionGeometryObject(ps);
   
  else {
      cerr << "Unknown Geometry Object Type R (" << go_type << ") aborting\n";
      exit(1);
  }
}
#endif


#endif


