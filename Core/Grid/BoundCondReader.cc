#include <Packages/Uintah/Core/Grid/BoundCondReader.h>
#include <Packages/Uintah/Core/Grid/BoundCondBase.h>
#include <Packages/Uintah/Core/Grid/BCGeomBase.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Grid/BoundCondFactory.h>
#include <Packages/Uintah/Core/Grid/BCDataArray.h>
#include <Packages/Uintah/Core/Grid/SideBCData.h>
#include <Packages/Uintah/Core/Grid/CircleBCData.h>
#include <Packages/Uintah/Core/Grid/RectangleBCData.h>
#include <Packages/Uintah/Core/Grid/UnionBCData.h>
#include <Packages/Uintah/Core/Grid/DifferenceBCData.h>
#include <Packages/Uintah/Core/Grid/BCData.h>
#include <Core/Malloc/Allocator.h>

#include <iostream>
#include <utility>
#include <typeinfo>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <string>
#include <map>
#include <list>

#define PRINT
#undef PRINT

using namespace std;
using namespace Uintah;

BCReader::BCReader() 
{
  d_bcs.resize(6);
}

BCReader::~BCReader()
{
}

BCGeomBase* BCReader::createBoundaryConditionFace(ProblemSpecP& face_ps,
						  Patch::FaceType& face_side)
{

  map<string,string> values;
  face_ps->getAttributes(values);
      
  // Have three possible types for the boundary condition face:
  // a. side (original -- entire side is one bc)
  // b. cirle (part of the side consists of a circle)
  // c. rectangle (part of the side consists of a rectangle)
  // This allows us to specify variable boundary conditions on a given
  // side.  Will use the notion of a UnionBoundaryCondtion and Difference
  // BoundaryCondition.
  
    
  // Do the side case:
  std::string fc;
  BCGeomBase* bcGeom;
  if (values.find("side") != values.end()) {
    fc = values["side"];
    bcGeom = scinew SideBCData();
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
    bcGeom = scinew CircleBCData(p,r);
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
    bcGeom = scinew RectangleBCData(l,u);
  }
#ifdef PRINT
  cout << "Face = " << fc << endl;      
#endif
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
  
  return bcGeom;
}

void
BCReader::read(ProblemSpecP& bc_ps)
{  
  for (ProblemSpecP face_ps = bc_ps->findBlock("Face");
       face_ps != 0; face_ps=face_ps->findNextBlock("Face")) {

      Patch::FaceType face_side;
      BCGeomBase* bcGeom = createBoundaryConditionFace(face_ps,face_side);
#ifdef PRINT
      cout << endl << endl << "Face = " << face_side << " Geometry type = " 
	   << typeid(*bcGeom).name() << " " << bcGeom << endl;
#endif
	      
      multimap<int, BoundCondBase*> bctype_data;

      for (ProblemSpecP child = face_ps->findBlock("BCType"); child != 0;
	   child = child->findNextBlock("BCType")) {
	int mat_id;
	BoundCondBase* bc;
	BoundCondFactory::create(child,bc,mat_id);
#ifdef PRINT
	cout << "Inserting into mat_id = " << mat_id << " bc = " <<
	  bc->getType() << " bctype = " << typeid(*bc).name() << endl;
#endif
	// This is for the old boundary conditions.
	d_bcs[face_side].setBCValues(mat_id,bc);
	bctype_data.insert(pair<int,BoundCondBase*>(mat_id,bc->clone()));
	delete bc;
      }
#ifdef PRINT
      // Print out all of the bcs just created
      multimap<int,BoundCondBase*>::const_iterator it;
      for (it = bctype_data.begin(); it != bctype_data.end(); it++) {
	cout << "Getting out mat_id = " << it->first << " bc = " 
	     << it->second->getType() << " bctype = " 
	     << typeid(*(it->second)).name() << endl;
      }
#endif

      // Search through the newly created boundary conditions and create
      // new BCGeomBase* clones if there are multi materials specified 
      // in the give <Face>.  This is usually a problem when Pressure is
      // specified for material id = 0, and other bcs such as velocity,
      // temperature, etc. for material_id != 0.

      map<int, BCGeomBase*> bcgeom_data;
      map<int, BCGeomBase*>::const_iterator bc_geom_itr,mat_all_itr;

      // Search through the bctype_data and make sure that there are
      // enough bcGeom clones for each material.
      multimap<int,BoundCondBase*>::const_iterator itr;
      for (itr = bctype_data.begin(); itr != bctype_data.end(); itr++) {
	bc_geom_itr =  bcgeom_data.find(itr->first);
	// Clone it
	if (bc_geom_itr == bcgeom_data.end()) {
	  bcgeom_data[itr->first] = bcGeom->clone();
	}
#ifdef PRINT
	cout << "Storing in  = " << typeid(bcgeom_data[itr->first]).name()
	     << " " << bcgeom_data[itr->first] << " " 
	     << typeid(*(itr->second)).name()
	     << endl;
#endif
	bcgeom_data[itr->first]->addBC(itr->second);
      }
      for (bc_geom_itr = bcgeom_data.begin(); bc_geom_itr != bcgeom_data.end();
	   bc_geom_itr++) {
	d_BCReaderData[face_side].addBCData(bc_geom_itr->first,
					    bcgeom_data[bc_geom_itr->first]->clone());
	delete bc_geom_itr->second;
      }
						   
#ifdef PRINT 
      cout << "Printing out bcDataArray . . " << endl;
      d_BCReaderData[face_side].print();
#endif

      delete bcGeom;

      // Delete stuff in bctype_data
      multimap<int, BoundCondBase*>::const_iterator m_itr;
      for (m_itr = bctype_data.begin(); m_itr != bctype_data.end(); ++m_itr) 
	delete m_itr->second;
  }

  // Find the mat_id = "all" (-1) information and store it in each 
  // materials boundary condition section.
  BCDataArray::bcDataArrayType::const_iterator  mat_all_itr, bc_geom_itr;
  for (Patch::FaceType face = Patch::startFace; 
       face <= Patch::endFace; face=Patch::nextFace(face)) {
    mat_all_itr = d_BCReaderData[face].d_BCDataArray.find(-1);
    if (mat_all_itr != d_BCReaderData[face].d_BCDataArray.end()) 
      for (bc_geom_itr = d_BCReaderData[face].d_BCDataArray.begin(); 
	   bc_geom_itr != d_BCReaderData[face].d_BCDataArray.end(); 
	   bc_geom_itr++) {
	if (bc_geom_itr != mat_all_itr) {
	  vector<BCGeomBase*>::const_iterator itr;
	  for (itr = mat_all_itr->second.begin(); 
	       itr != mat_all_itr->second.end(); ++itr)
	    d_BCReaderData[face].addBCData(bc_geom_itr->first,
					   (*itr)->clone());
	}
      }
  }


  // Need to take the individual boundary conditions and combine them into
  // a single different (side and the union of any holes (circles or
  // rectangles.  This only happens if there are more than 1 bc_data per
  // face.
  
#ifdef PRINT
  cout << endl << "Before combineBCS() . . ." << endl << endl;
  for (Patch::FaceType face = Patch::startFace; 
       face <= Patch::endFace; face=Patch::nextFace(face)) {
    cout << endl << endl << "Before Face . . ." << face << endl;
    d_BCReaderData[face].print();
  } 
  
#endif
  combineBCS();

#ifdef PRINT
  cout << endl << "After combineBCS() . . ." << endl << endl;
  for (Patch::FaceType face = Patch::startFace; 
       face <= Patch::endFace; face=Patch::nextFace(face)) {
    cout << "After Face . . .  " << face << endl;
    d_BCReaderData[face].print();
  } 
#endif

}


// For old boundary conditions
void BCReader::getBC(Patch::FaceType& face, BoundCondData& bc_data)
{
  bc_data = d_bcs[face];
}

const BCDataArray BCReader::getBCDataArray(Patch::FaceType& face) const
{
  map<Patch::FaceType,BCDataArray > m = this->d_BCReaderData;
  return m[face];
}



void BCReader::combineBCS()
{
  for (Patch::FaceType face = Patch::startFace; 
       face <= Patch::endFace; face=Patch::nextFace(face)) {
#ifdef PRINT
    cout << endl << "Working on Face = " << face << endl;
    cout << endl << "Original inputs" << endl;
#endif
    BCDataArray rearranged;
    BCDataArray& original = d_BCReaderData[face];
#ifdef PRINT
    original.print();
    cout << endl;
#endif
    BCDataArray::bcDataArrayType::const_iterator mat_id_itr;
    for (mat_id_itr = original.d_BCDataArray.begin(); 
	 mat_id_itr != original.d_BCDataArray.end(); 
	 ++mat_id_itr) {
      int mat_id = mat_id_itr->first;
#ifdef PRINT
      cout << "Mat ID = " << mat_id << endl;
#endif

      // Find all of the BCData types that are in a given BCDataArray
      vector<BCGeomBase*>::const_iterator vec_itr;
      map<BCData,vector<BCGeomBase*> > bcdata_bcgeom;
      for (vec_itr = mat_id_itr->second.begin(); 
	   vec_itr != mat_id_itr->second.end(); ++vec_itr) {
	BCData bc_data;
	(*vec_itr)->getBCData(bc_data);
	bcdata_bcgeom[bc_data].push_back((*vec_itr)->clone());
      }
	
      map<BCData,vector<BCGeomBase*> >::iterator bcd_itr;
      for (bcd_itr = bcdata_bcgeom.begin(); bcd_itr != bcdata_bcgeom.end(); 
	   ++bcd_itr) {
#ifdef PRINT
	cout << "Printing out the bcd types" << endl;
	bcd_itr->first.print();
#endif
	
	if (count_if(bcd_itr->second.begin(),
		     bcd_itr->second.end(),
		     cmp_type<SideBCData>) == 1 && 
	    bcd_itr->second.size() == 1) {
	  BCGeomBase* bc = bcd_itr->second[0];
	  rearranged.addBCData(mat_id,bc->clone());
	} else {
	  // Find the child that is the "side" bc
	  BCGeomBase* side_bc;
	  vector<BCGeomBase*>::const_iterator index;
	  index = find_if(bcd_itr->second.begin(),
			  bcd_itr->second.end(),
			  cmp_type<SideBCData>);
	
	  
	  if (index != bcd_itr->second.end()) {
#ifdef PRINT
	    cout << "Found the side bc data" << endl;
#endif
	    side_bc = (*index)->clone();
	  } else {
#ifdef PRINT
	    cout << "Didnt' find the side bc data" << endl;
#endif
	    index = find_if(original.d_BCDataArray[-1].begin(),
			    original.d_BCDataArray[-1].end(),
			    cmp_type<SideBCData>);
	    if (index != d_BCReaderData[face].d_BCDataArray[-1].end()){
#ifdef PRINT
	      cout << "Using the 'all' case" << endl;
#endif
	      side_bc = (*index)->clone();
	    }
	  }
	  
	  // Create a unionbcdata for all the remaining bcs and remove the 
	  // sidebcdata.  
	  UnionBCData* union_bc = scinew UnionBCData();
	  for (vector<BCGeomBase*>::const_iterator i = bcd_itr->second.begin();
	       i != bcd_itr->second.end(); ++i)
	    union_bc->child.push_back((*i)->clone());
	  vector<BCGeomBase*>::iterator itr, new_end = 
	    remove_if(union_bc->child.begin(),
		      union_bc->child.end(),
		      cmp_type<SideBCData>);
	  
#ifdef PRINT
	  cout << endl << "Before deleting" << endl;
	  for_each(union_bc->child.begin(),
		   union_bc->child.end(),
		   Uintah::print);
#endif
	  
	  for(itr = new_end; itr != union_bc->child.end(); ++itr)
	    delete *itr;
	  union_bc->child.erase(new_end,union_bc->child.end());
#ifdef PRINT  
	  cout << endl << "After deleting" << endl;
	  for_each(union_bc->child.begin(),
		   union_bc->child.end(),
		   Uintah::print);
#endif
	  
	  // Create a differencebcdata for the side and the unionbc
	  DifferenceBCData* difference_bc = 
	    scinew DifferenceBCData(side_bc,union_bc);
	  rearranged.addBCData(mat_id,difference_bc->clone());
	  
	  // Take the individual bcs and add them to the rearranged list.
	  // These are found in the union_bcs (doesn't have the SideDataBC).
	  vector<BCGeomBase*>::const_iterator it;
	  for (it = union_bc->child.begin(); 
	       it != union_bc->child.end(); ++it) 
	    rearranged.addBCData(mat_id,(*it)->clone());
#ifdef PRINT  
	  cout << endl << "Printing out BCGeomBase types in rearranged" 
	       << endl;
	  for_each(rearranged.d_BCDataArray[mat_id].begin(),
		   rearranged.d_BCDataArray[mat_id].end(),
		   Uintah::print);
#endif
	  delete side_bc;
	  delete union_bc;
	  delete difference_bc;
	}
      }
      // Delete the bcdata_bcgeom stuff
      for (bcd_itr = bcdata_bcgeom.begin(); bcd_itr != bcdata_bcgeom.end();
	   ++bcd_itr) {
	for (vec_itr=bcd_itr->second.begin();vec_itr != bcd_itr->second.end();
	     ++vec_itr)
	  delete *vec_itr;
	bcd_itr->second.clear();
      }
      bcdata_bcgeom.clear();
      
    }

#ifdef PRINT
    cout << endl << "Printing out rearranged list" << endl;
    rearranged.print();
#endif
    // Reassign the rearranged data
    d_BCReaderData[face] = rearranged;
    
#ifdef PRINT
    cout << endl << "Printing out rearranged from d_BCReaderData list" << endl;
    d_BCReaderData[face].print();
#endif

  }

#ifdef PRINT
  cout << endl << "Printing out in combineBCS()" << endl;
  for (Patch::FaceType face = Patch::startFace; 
       face <= Patch::endFace; face=Patch::nextFace(face)) {
    cout << "After Face . . .  " << face << endl;
    d_BCReaderData[face].print();
  } 
#endif


}


bool BCReader::compareBCData(BCGeomBase* b1, BCGeomBase* b2)
{
  return false;
}

namespace Uintah {

void print(BCGeomBase* p) {
  cout << "type = " << typeid(*p).name() << endl;
}

}



