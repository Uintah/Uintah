#include "GeometryObjectFactory.h"
#include <fstream>
#include <iostream>
#include <string>
using std::cerr;
using std::ifstream;
using std::ofstream;

using namespace Uintah::Components;


void GeometryObjectFactory::readParameters(ProblemSpecP ps, 
					      std::string go_type,
					      double *p_array)
{
  if (go_type == "box")
    BoxGeometryObject::readParameters(ps, p_array);
     
  else if (go_type == "sphere")
    SphereGeometryObject::readParameters(ps, p_array);

  else if (go_type ==  "cylinder")
    CylinderGeometryObject::readParameters(ps, p_array);

  else if (go_type == "tri")
    TriGeometryObjectPlas::readParameters(ps, p_array);
 
  else if (go_type == "union")
    UnionGeometryObject::readParameters(ps, p_array);
   
  else if (go_type == "difference")
    DifferenceGeometryObject::readParameters(ps, p_array);

  else if (go_type == "instersection")
    IntersectionGeometryObject::readParameters(ps, p_array);
   
  else {
      cerr << "Unknown Geometry Object Type R (" << go_type << ") aborting\n";
      exit(1);
  }
}

#ifdef WONT_COMPILE_YET  

GeometryObject* GeometryObjectFactory::readParametersAndCreate(
					     ProblemSpecP ps,
					     std::string go_type)
{
  if (go_type == "box")
    return(BoxGeometryObject::readParametersAndCreate(ps));
 
  else if (go_type =="sphere")
    return(SphereGeometryObject::readParametersAndCreate(ps));

  else if (go_type == "cylinder")
    return(CylinderGeometryObject::readParametersAndCreate(ps));

  else if (go_type == "tri")
    return(TriGeometryObject::readParametersAndCreate(ps));

  else if (go_type == "union")
    return(UnionGeometryObject::readParametersAndCreate(ps));

  else if (go_type == "difference")
    return(DifferenceGeometryObject::readParametersAndCreate(ps));

  else if (go_type == "instersection")
    return(IntersectionGeometryObject::readParametersAndCreate(ps));
   
  else {
      cerr << "Unknown Geometry Object Type RaC (" << go_type << ") aborting\n";
      exit(1);
  }
  return(0);
}

GeometryObject* GeometryObjectFactory::readRestartParametersAndCreate(
					     ProblemSpecP ps,
					     std::string go_type)
{
 
  if (go_type == "box")
    return(BoxGeometryObject::readRestartParametersAndCreate(ps));

  else if (go_type == "sphere")
    return(SphereGeometryObject::readRestartParametersAndCreate(ps));

  else if (go_type == "cylinder")
    return(CylinderGeometryObject::readRestartParametersAndCreate(ps));

  else if (go_type == "tri")
    return(TriGeometryObject::readRestartParametersAndCreate(ps));

  else if (go_type == "union")
    return(UnionGeometryObject::readRestartParametersAndCreate(ps));

  else if (go_type == "difference")
    return(DifferenceGeometryObject::readRestartParametersAndCreate(ps));

  else if (go_type == "intersection")
    return(IntersectionGeometryObject::readRestartParametersAndCreate(ps));
  
  else {
      cerr << "Unknown Geometry Object Type (" << go_type << ") aborting\n";
      exit(1);
  }
  return(0);
}

GeometryObject* GeometryObjectFactory::create(std::string go_type,
						    double *p_array)
{
  if (go_type == "box")
    return(BoxGeometryObject::create(p_array));

  else if (go_type == "sphere")
    return(SphereGeometryObject::create(p_array));

  else if (go_type == "cylinder")
    return(CylinderGeometryObject::create(p_array));

  else if (go_type == "tri")
    return(TriGeometryObject::create(p_array));

  else if (go_type == "union")
    return(UnionGeometryObject::create(p_array));

  else if (go_type == "difference")
    return(DifferenceGeometryObject::create(p_array));

  else if (go_type == "intersection")
    return(IntersectionGeometryObject::create(p_array));
   
  else {
    cerr << "Unknown Geometry Object Type c (" << go_type << ") aborting\n";
    exit(1);
  }
  return(0);
}

#endif


