#include <Packages/Uintah/Core/Grid/BoundCondReader.h>
#include <Packages/Uintah/Core/Grid/BoundCondBase.h>
#include <Packages/Uintah/Core/Grid/BCDataBase.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Grid/BoundCondFactory.h>
#include <Packages/Uintah/Core/Grid/SideBCData.h>
#include <Packages/Uintah/Core/Grid/CircleBCData.h>
#include <Packages/Uintah/Core/Grid/RectangleBCData.h>
#include <Packages/Uintah/Core/Grid/UnionBCData.h>
#include <Packages/Uintah/Core/Grid/DifferenceBCData.h>
using namespace std;
using namespace Uintah;
#include <iostream>
#include <utility>
#include <typeinfo>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <string>

BCReader::BCReader() 
{
}

BCReader::~BCReader()
{
}

void
BCReader::read(ProblemSpecP& bc_ps)
{  
  for (ProblemSpecP face_ps = bc_ps->findBlock("Face");
       face_ps != 0; face_ps=face_ps->findNextBlock("Face"))  // end of face_ps
    {
      map<string,string> values;
      face_ps->getAttributes(values);
      
      Patch::FaceType face_side = Patch::invalidFace;
      
      // Have three possible types for the boundary condition face:
      // a. side (original -- entire side is one bc)
      // b. cirle (part of the side consists of a circle)
      // c. rectangle (part of the side consists of a rectangle)
      // This allows us to specify variable boundary conditions on a given
      // side.  Will use the notion of a UnionBoundaryCondtion and Difference
      // BoundaryCondition.
      
      // Do the side case:
      std::string fc;
      BCDataBase* bc;
      if (values.find("side") != values.end()) {
	fc = values["side"];
	bc = new SideBCData();
      }
      // Do the circle case:
      if (values.find("circle") != values.end()) {
	fc = values["circle"];
	string origin = values["origin"];
	string radius = values["radius"];
	stringstream origin_stream(origin);
	stringstream radius_stream(radius);
	double r,o[3];
	radius_stream >> r;
	origin_stream >> o[0] >> o[1] >> o[2];
	Point p(o[0],o[1],o[2]);
	bc = new CircleBCData(p,r);
      }
      // Do the rectangle case:
      if (values.find("rectangle") != values.end()) {
	fc = values["rectangle"];
	string low = values["lower"];
	string up = values["upper"];
	stringstream low_stream(low), up_stream(up);
	double lower[3],upper[3];
	low_stream >> lower[0] >> lower[1] >> lower[2];
	up_stream >> upper[0] >> upper[1] >> upper[2];
	Point l(lower[0],lower[1],lower[2]),u(upper[0],upper[1],upper[2]);
	bc = new RectangleBCData(l,u);
      }
      
      if (fc ==  "x-")
	face_side = Patch::xminus;
      if (fc == "x+")
	face_side = Patch::xplus;
      if (fc == "y-")
	face_side = Patch::yminus;
      if (fc == "y+")
	face_side = Patch::yplus;
      if (fc == "z-")
	face_side = Patch::zminus;
      if (fc == "z+")
	face_side = Patch::zplus;
      
      BCData bc_data;
      BoundCondFactory::create(face_ps,bc_data);

      bc->addBCData(bc_data);

      d_bc[face_side].addBCData(bc);
      
    }

  // Need to take the individual boundary conditions and combine them into
  // a single different (side and the union of any holes (circles or
  // rectangles.  This only happens if there are more than 1 bc_data per
  // face.
  
#if 0
  for (Patch::FaceType face = Patch::startFace; 
       face <= Patch::endFace; face=Patch::nextFace(face)) {
    for (int child = 0; child < getBC(face).getNumberChildren(); child++) 
      cout << "Before Face = " << face << " BC = " 
	   << typeid(*d_bc[face].getChild(child)).name() << endl;
  } 
#endif
  combineBCS();
#if 0
  for (Patch::FaceType face = Patch::startFace; 
       face <= Patch::endFace; face=Patch::nextFace(face)) {
    for (int child = 0; child < getBC(face).getNumberChildren(); child++) 
      cout << "After Face = " << face << " BC = " 
	   << typeid(*d_bc[face].getChild(child)).name() << endl;
  } 
#endif
}

void BCReader::getBC(Patch::FaceType& face, BCData& bc_data)
{
  d_bc[face].getBCData(bc_data,0);
}

const BCDataArray BCReader::getBC(Patch::FaceType& face) const
{
  map<Patch::FaceType,BCDataArray > m = this->d_bc;
  return m[face];
}

void BCReader::combineBCS()
{
  
  for (Patch::FaceType face = Patch::startFace; 
       face <= Patch::endFace; face=Patch::nextFace(face)) {
    if (getBC(face).getNumberChildren() == 1)
      continue;

    BCDataArray original = getBC(face);
    BCDataArray rearranged;
    int side_child_index;
    BCDataBase* side_bc;
    for (int child = 0; child < getBC(face).getNumberChildren(); child++) {
      // Find the child that is the "side" bc
      if(typeid(SideBCData) == typeid(*original.getChild(child))) {
	side_child_index = child;
	side_bc = original.getChild(child)->clone();
	break; 
      }
    }
    // Create a unionbcdata for all the remaining bcs.  
    UnionBCData* union_bc = new UnionBCData();
    for (int child = 0; child < getBC(face).getNumberChildren(); child++) {
      if (child != side_child_index) {
	BCDataBase* bc = original.getChild(child)->clone();
	union_bc->addBCData(bc);
      }
    }

    // Create a differencebcdata for the side and the unionbc
    BCDataBase* difference_bc = new DifferenceBCData(side_bc,union_bc);
    
    rearranged.addBCData(difference_bc);
    
    // Take the individual bc's and add them in to the rearranged list. 
    for (int child = 0; child < getBC(face).getNumberChildren(); child++) {
      if (child != side_child_index) {
	BCDataBase* bc = original.getChild(child)->clone();
	rearranged.addBCData(bc);
      }
    }

    for (int child = 0; child < rearranged.getNumberChildren();
	 child++) 
      cout << "Rearranged = " << typeid(*rearranged.getChild(child)).name()
	   << endl;
    // Reassign the rearranged data
    d_bc[face] = rearranged;

    for (int child = 0; child < d_bc[face].getNumberChildren();
	 child++) 
      cout << "d_bcs = " << typeid(*d_bc[face].getChild(child)).name()
	   << endl;
  }
}
