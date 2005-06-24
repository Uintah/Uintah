#include <Packages/Uintah/Core/Grid/BoundaryConditions/BoundCondReader.h>
#include <Packages/Uintah/Core/Grid/BoundaryConditions/BoundCondBase.h>
#include <Packages/Uintah/Core/Grid/BoundaryConditions/BCGeomBase.h>
#include <Packages/Uintah/Core/ProblemSpec/ProblemSpec.h>
#include <Packages/Uintah/Core/Grid/BoundaryConditions/BoundCondFactory.h>
#include <Packages/Uintah/Core/Grid/BoundaryConditions/BCDataArray.h>
#include <Packages/Uintah/Core/Grid/BoundaryConditions/SideBCData.h>
#include <Packages/Uintah/Core/Grid/BoundaryConditions/CircleBCData.h>
#include <Packages/Uintah/Core/Grid/BoundaryConditions/AnnulusBCData.h>
#include <Packages/Uintah/Core/Grid/BoundaryConditions/RectangleBCData.h>
#include <Packages/Uintah/Core/Grid/BoundaryConditions/UnionBCData.h>
#include <Packages/Uintah/Core/Grid/BoundaryConditions/DifferenceBCData.h>
#include <Packages/Uintah/Core/Grid/BoundaryConditions/BCData.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/DebugStream.h>

#include <sgi_stl_warnings_off.h>
#include <utility>
#include <typeinfo>
#include <sstream>
#include <iostream>
#include <algorithm>
#include <string>
#include <map>
#include <sgi_stl_warnings_on.h>

using namespace std;
using namespace Uintah;

// export SCI_DEBUG="BCR_DBG:+,OLD_BC_DBG:+"
static DebugStream BCR_dbg ("BCR_DBG", false);


BoundCondReader::BoundCondReader() 
{
}

BoundCondReader::~BoundCondReader()
{
}

void BoundCondReader::whichPatchFace(const std::string fc,
                                Patch::FaceType& face_side, 
                                int& plusMinusFaces,
                                int& p_dir)
{
  if (fc ==  "x-"){
    plusMinusFaces = -1;
    p_dir = 0;
    face_side = Patch::xminus;
  }
  if (fc == "x+"){
    plusMinusFaces = 1;
    p_dir = 0;
    face_side = Patch::xplus;
  }
  if (fc == "y-"){
    plusMinusFaces = -1;
    p_dir = 1;
    face_side = Patch::yminus;
  }
  if (fc == "y+"){
    plusMinusFaces = 1;
    p_dir = 1;
    face_side = Patch::yplus;
  }
  if (fc == "z-"){
    plusMinusFaces = -1;
    p_dir = 2;
    face_side = Patch::zminus;
  }
  if (fc == "z+"){
    plusMinusFaces = 1;
    p_dir = 2;
    face_side = Patch::zplus;
  }
}

BCGeomBase* BoundCondReader::createBoundaryConditionFace(ProblemSpecP& face_ps,
                                              const ProblemSpecP& grid_ps,
						    Patch::FaceType& face_side)
{

  // Determine the Level 0 grid high and low points, need by 
  // the bullet proofing
  Point grid_LoPt(1e30, 1e30, 1e30);
  Point grid_HiPt(-1e30, -1e30, -1e30); 
  
  for(ProblemSpecP level_ps = grid_ps->findBlock("Level");
      level_ps != 0; level_ps = level_ps->findNextBlock("Level")){

     //find upper/lower corner
     for(ProblemSpecP box_ps = level_ps->findBlock("Box");
        box_ps != 0; box_ps = box_ps->findNextBlock("Box")){
       Point lower;
       Point upper;
       box_ps->require("lower", lower);
       box_ps->require("upper", upper);
       grid_LoPt=Min(lower, grid_LoPt);
       grid_HiPt=Max(upper, grid_HiPt);
     }
   }


  map<string,string> values;
  face_ps->getAttributes(values);
      
  // Have three possible types for the boundary condition face:
  // a. side (original -- entire side is one bc)
  // b. cirle (part of the side consists of a circle)
  // c. rectangle (part of the side consists of a rectangle)
  // This allows us to specify variable boundary conditions on a given
  // side.  Will use the notion of a UnionBoundaryCondtion and Difference
  // BoundaryCondition.
  
  std::string fc;
  int plusMinusFaces, p_dir;
  BCGeomBase* bcGeom;
  if (values.find("side") != values.end()) {
    fc = values["side"];
    whichPatchFace(fc, face_side, plusMinusFaces, p_dir);
    bcGeom = scinew SideBCData();
  }
  else if (values.find("circle") != values.end()) {
    fc = values["circle"];
    whichPatchFace(fc, face_side, plusMinusFaces, p_dir);
    string origin = values["origin"];
    string radius = values["radius"];
    stringstream origin_stream(origin);
    stringstream radius_stream(radius);
    double r,o[3];
    radius_stream >> r;
    origin_stream >> o[0] >> o[1] >> o[2];
    Point p(o[0],o[1],o[2]);
    
    
    //  bullet proofing-- origin must be on the same plane as the face
    bool test = true;
    if(plusMinusFaces == -1){    // x-, y-, z- faces
      test = (p(p_dir) != grid_LoPt(p_dir));
    }
    if(plusMinusFaces == 1){     // x+, y+, z+ faces
      test = (p(p_dir) != grid_HiPt(p_dir));
    }    
    
    if(test){
      ostringstream warn;
      warn<<"ERROR: Input file\n The Circle BC geometry is not correctly specified."
          << " The origin " << p << " must be on the same plane" 
          << " as face (" << fc <<"). Double check the origin and Level:box spec. \n\n";
      throw ProblemSetupException(warn.str());
    }
    
    
    if (origin == "" || radius == "") {
      ostringstream warn;
      warn<<"ERROR\n Circle BC geometry not correctly specified \n"
          << " you must specify origin [x,y,z] and radius [r] \n\n";
      throw ProblemSetupException(warn.str());
    }
    
    
    bcGeom = scinew CircleBCData(p,r);
  }
  else if (values.find("annulus") != values.end()) {
    fc = values["annulus"];
    whichPatchFace(fc, face_side, plusMinusFaces, p_dir);
    string origin = values["origin"];
    string in_radius = values["inner_radius"];
    string out_radius = values["outer_radius"];
    stringstream origin_stream(origin);
    stringstream in_radius_stream(in_radius);
    stringstream out_radius_stream(out_radius);
    double i_r,o_r,o[3];
    in_radius_stream >> i_r;
    out_radius_stream >> o_r;
    origin_stream >> o[0] >> o[1] >> o[2];
    Point p(o[0],o[1],o[2]);

    //  bullet proofing-- origin must be on the same plane as the face
    bool test = true;
    if(plusMinusFaces == -1){    // x-, y-, z- faces
      test = (p(p_dir) != grid_LoPt(p_dir));
    }
    if(plusMinusFaces == 1){     // x+, y+, z+ faces
      test = (p(p_dir) != grid_HiPt(p_dir));
    }    
    
    if(test){
      ostringstream warn;
      warn<<"ERROR: Input file\n The Annulus BC geometry is not correctly specified."
          << " The origin " << p << " must be on the same plane" 
          << " as face (" << fc <<"). Double check the origin and Level:box spec. \n\n";
      throw ProblemSetupException(warn.str());
    }
    
    if (origin == "" || in_radius == "" || out_radius == "" ) {
      ostringstream warn;
      warn<<"ERROR\n Annulus BC geometry not correctly specified \n"
          << " you must specify origin [x,y,z], inner_radius [r] outer_radius [r] \n\n";
      throw ProblemSetupException(warn.str());
    }
    bcGeom = scinew AnnulusBCData(p,i_r,o_r);
  }
  else if (values.find("rectangle") != values.end()) {
    fc = values["rectangle"];
    whichPatchFace(fc, face_side, plusMinusFaces, p_dir);
    string low = values["lower"];
    string up = values["upper"];
    stringstream low_stream(low), up_stream(up);
    double lower[3],upper[3];
    low_stream >> lower[0] >> lower[1] >> lower[2];
    up_stream >> upper[0] >> upper[1] >> upper[2];
    Point l(lower[0],lower[1],lower[2]),u(upper[0],upper[1],upper[2]);
   
    //  bullet proofing-- rectangle must be on the same plane as the face
    bool test = (l(p_dir) != grid_LoPt(p_dir) && u(p_dir) != grid_HiPt(p_dir));   
    
    if(test){
      ostringstream warn;
      warn<<"ERROR: Input file\n The rectangle BC geometry is not correctly specified."
          << " The low " << l << " high " << u << " points must be on the same plane" 
          << " as face (" << fc <<"). Double check against and Level:box spec. \n\n";
      throw ProblemSetupException(warn.str());
    }   
   
    if (low == "" || up == "") {
      ostringstream warn;
      warn<<"ERROR\n Rectangle BC geometry not correctly specified \n"
          << " you must specify lower [x,y,z] and upper[x,y,z] \n\n";
      throw ProblemSetupException(warn.str());
    }
    if ( (l.x() >  u.x() || l.y() >  u.y() || l.z() >  u.z()) ||
         (l.x() == u.x() && l.y() == u.y() && l.z() == u.z())){
      ostringstream warn;
      warn<<"ERROR\n Rectangle BC geometry not correctly specified \n"
          << " lower pt "<< l <<" upper pt " << u;
      throw ProblemSetupException(warn.str());
    }
    
    bcGeom = scinew RectangleBCData(l,u);
  }
  
  else {
    ostringstream warn;
    warn << "ERROR\n Boundary condition geometry not correctly specified "
      " Valid options (side, circle, rectangle, annulus";
    throw ProblemSetupException(warn.str());  
  }
  

  BCR_dbg << "Face = " << fc << endl;
  return bcGeom;
}



void
BoundCondReader::read(ProblemSpecP& bc_ps, const ProblemSpecP& grid_ps)
{  
  // This function first looks for the geometric specification for the 
  // boundary condition which includes the tags side, circle and rectangle.
  // The function createBoundaryConditionFace parses the tag and creates the
  // appropriate class.  Once this class is created, then the actual boundary
  // conditions are parsed from the input file (Pressure, Density, etc.).
  // Boundary conditions can be specified for various materials within a 
  // face.  This complicates things, so we have to check for this and then 
  // separate them out.  Once things are separated out, we then must take
  // all the boundary conditions for a given face and material id and combine
  // them so that any circle or rectangles that are specified can be combined
  // appropriately with the side case.  Multiple circle/rectangles are added
  // together and stored in a Union class.  This union class is then subtracted
  // off from the side class resulting in a difference class.  The difference
  // class represents the region of the side minus any circles/rectangle.  
  
  for (ProblemSpecP face_ps = bc_ps->findBlock("Face");
       face_ps != 0; face_ps=face_ps->findNextBlock("Face")) {

      Patch::FaceType face_side;
      BCGeomBase* bcGeom = createBoundaryConditionFace(face_ps,grid_ps,face_side);
      BCR_dbg << endl << endl << "Face = " << face_side << " Geometry type = " 
	      << typeid(*bcGeom).name() << " " << bcGeom << endl;
	      
      multimap<int, BoundCondBase*> bctype_data;

      for (ProblemSpecP child = face_ps->findBlock("BCType"); child != 0;
	   child = child->findNextBlock("BCType")) {
	int mat_id;
	BoundCondBase* bc;
	BoundCondFactory::create(child,bc,mat_id);
	BCR_dbg << "Inserting into mat_id = " << mat_id << " bc = " 
		<<  bc->getType() << " bctype = " << typeid(*bc).name() 
		<<  " "  << bc  << endl;

	bctype_data.insert(pair<int,BoundCondBase*>(mat_id,bc->clone()));
	delete bc;
      }

      // Print out all of the bcs just created
      multimap<int,BoundCondBase*>::const_iterator it;
      for (it = bctype_data.begin(); it != bctype_data.end(); it++) {
	BCR_dbg << "Getting out mat_id = " << it->first << " bc = " 
		<< it->second->getType() << " bctype = " 
		<< typeid(*(it->second)).name() << endl;
      }

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

	BCR_dbg << "Storing in  = " << typeid(bcgeom_data[itr->first]).name()
		<< " " << bcgeom_data[itr->first] << " " 
		<< typeid(*(itr->second)).name() << " " << (itr->second)
		<< endl;

	bcgeom_data[itr->first]->addBC(itr->second);
      }
      for (bc_geom_itr = bcgeom_data.begin(); bc_geom_itr != bcgeom_data.end();
	   bc_geom_itr++) {
	d_BCReaderData[face_side].addBCData(bc_geom_itr->first,
				    bcgeom_data[bc_geom_itr->first]->clone());
	delete bc_geom_itr->second;
      }
						   
      BCR_dbg << "Printing out bcDataArray . . " << endl;
      //d_BCReaderData[face_side].print();


      delete bcGeom;

      // Delete stuff in bctype_data
      multimap<int, BoundCondBase*>::const_iterator m_itr;
      for (m_itr = bctype_data.begin(); m_itr != bctype_data.end(); ++m_itr) 
	delete m_itr->second;
  }

#if 1
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
#if 0
    for (bc_geom_itr = d_BCReaderData[face].d_BCDataArray.begin();
	 bc_geom_itr != d_BCReaderData[face].d_BCDataArray.end();
	 bc_geom_itr++) {
      cout << "mat_id = " << bc_geom_itr->first << endl;
      d_BCReaderData[face].combineBCGeometryTypes(bc_geom_itr->first);
    }
#endif
    
    BCR_dbg << "Printing out bcDataArray for face " << face 
	    << " after adding 'all' . . " << endl;
    //d_BCReaderData[face].print();
  }
#endif

  // Need to take the individual boundary conditions and combine them into
  // a single different (side and the union of any holes (circles or
  // rectangles.  This only happens if there are more than 1 bc_data per
  // face.
  

  BCR_dbg << endl << "Before combineBCS() . . ." << endl << endl;
  for (Patch::FaceType face = Patch::startFace; 
       face <= Patch::endFace; face=Patch::nextFace(face)) {
    BCR_dbg << endl << endl << "Before Face . . ." << face << endl;
    //    d_BCReaderData[face].print();
  } 
  

  combineBCS();


  BCR_dbg << endl << "After combineBCS() . . ." << endl << endl;
  for (Patch::FaceType face = Patch::startFace; 
       face <= Patch::endFace; face=Patch::nextFace(face)) {
    BCR_dbg << "After Face . . .  " << face << endl;
    //    d_BCReaderData[face].print();
  } 

}

const BCDataArray BoundCondReader::getBCDataArray(Patch::FaceType& face) const
{
  map<Patch::FaceType,BCDataArray > m = this->d_BCReaderData;
  return m[face];
}



void BoundCondReader::combineBCS()
{
  for (Patch::FaceType face = Patch::startFace; 
       face <= Patch::endFace; face=Patch::nextFace(face)) {
    BCR_dbg << endl << "Working on Face = " << face << endl;
    BCR_dbg << endl << "Original inputs" << endl;

    BCDataArray rearranged;
    BCDataArray& original = d_BCReaderData[face];

    //    original.print();
    BCR_dbg << endl;

    BCDataArray::bcDataArrayType::const_iterator mat_id_itr;
    for (mat_id_itr = original.d_BCDataArray.begin(); 
	 mat_id_itr != original.d_BCDataArray.end(); 
	 ++mat_id_itr) {
      int mat_id = mat_id_itr->first;

      BCR_dbg << "Mat ID = " << mat_id << endl;


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

	BCR_dbg << "Printing out the bcd types" << endl;
	//	bcd_itr->first.print();

	
	if (count_if(bcd_itr->second.begin(),
		     bcd_itr->second.end(),
		     cmp_type<SideBCData>()) == 1 && 
	    bcd_itr->second.size() == 1) {
	  BCGeomBase* bc = bcd_itr->second[0];
	  rearranged.addBCData(mat_id,bc->clone());
	} else {
	  // Find the child that is the "side" bc
	  BCGeomBase* side_bc;
	  vector<BCGeomBase*>::const_iterator index;
	  index = find_if(bcd_itr->second.begin(),
			  bcd_itr->second.end(),
			  cmp_type<SideBCData>());
	
	  
	  if (index != bcd_itr->second.end()) {
	    BCR_dbg << "Found the side bc data" << endl;
	    side_bc = (*index)->clone();
	  } else {
	    BCR_dbg << "Didnt' find the side bc data" << endl;

	    index = find_if(original.d_BCDataArray[-1].begin(),
			    original.d_BCDataArray[-1].end(),
			    cmp_type<SideBCData>());

	    if (index != d_BCReaderData[face].d_BCDataArray[-1].end()){
	      BCR_dbg << "Using the 'all' case" << endl;
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
		      cmp_type<SideBCData>());
	  
	  BCR_dbg << endl << "Before deleting" << endl;
	  for_each(union_bc->child.begin(),
		   union_bc->child.end(),
		   Uintah::print);

	  
	  for(itr = new_end; itr != union_bc->child.end(); ++itr)
	    delete *itr;
	  union_bc->child.erase(new_end,union_bc->child.end());

	  BCR_dbg << endl << "After deleting" << endl;

	  for_each(union_bc->child.begin(),union_bc->child.end(),
		   Uintah::print);

	  
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

	  BCR_dbg << endl << "Printing out BCGeomBase types in rearranged" 
		  << endl;

	  for_each(rearranged.d_BCDataArray[mat_id].begin(),
		   rearranged.d_BCDataArray[mat_id].end(),
		   Uintah::print);

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

    BCR_dbg << endl << "Printing out rearranged list" << endl;
    //    rearranged.print();

    // Reassign the rearranged data
    d_BCReaderData[face] = rearranged;
    
    BCR_dbg << endl << "Printing out rearranged from d_BCReaderData list" 
	    << endl;

    //    d_BCReaderData[face].print();

  }


  BCR_dbg << endl << "Printing out in combineBCS()" << endl;
  for (Patch::FaceType face = Patch::startFace; 
       face <= Patch::endFace; face=Patch::nextFace(face)) {
    BCR_dbg << "After Face . . .  " << face << endl;
    //    d_BCReaderData[face].print();
  } 


}


bool BoundCondReader::compareBCData(BCGeomBase* b1, BCGeomBase* b2)
{
  return false;
}

namespace Uintah {

  void print(BCGeomBase* p) {
    BCR_dbg << "type = " << typeid(*p).name() << endl;
  }
  
}



