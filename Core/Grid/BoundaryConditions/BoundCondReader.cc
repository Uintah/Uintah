/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#include <Core/Grid/BoundaryConditions/BoundCondReader.h>
#include <Core/Grid/BoundaryConditions/BoundCondBase.h>
#include <Core/Grid/BoundaryConditions/BCGeomBase.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/Grid/BoundaryConditions/BoundCondFactory.h>
#include <Core/Grid/BoundaryConditions/BCDataArray.h>
#include <Core/Grid/BoundaryConditions/SideBCData.h>
#include <Core/Grid/BoundaryConditions/CircleBCData.h>
#include <Core/Grid/BoundaryConditions/AnnulusBCData.h>
#include <Core/Grid/BoundaryConditions/EllipseBCData.h>
#include <Core/Grid/BoundaryConditions/RectangleBCData.h>
#include <Core/Grid/BoundaryConditions/UnionBCData.h>
#include <Core/Grid/BoundaryConditions/DifferenceBCData.h>
#include <Core/Grid/BoundaryConditions/BCData.h>
#include <Core/Grid/BoundaryConditions/BoundCond.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/DebugStream.h>

#include   <utility>
#include   <typeinfo>
#include   <sstream>
#include   <iostream>
#include   <algorithm>
#include   <string>
#include   <map>
#include   <iterator>
#include   <set>

using namespace std;

using namespace Uintah;

// export SCI_DEBUG="BCR_DBG:+,OLD_BC_DBG:+"
static DebugStream BCR_dbg ("BCR_DBG", false);


BoundCondReader::BoundCondReader() 
{
}

BoundCondReader::~BoundCondReader()
{
  //cout << "Calling BoundCondReader destructor" << endl;
}

// given a set of lower or upper bounds for multiple boxes (points)
// this function checks if Point p (usually center of circle, ellipse, or annulus)
// is on a given face on any of the boxes.
bool is_on_face(const int dir, 
                const Point p,
                const std::vector<Point>& points)
{
  std::vector<Point>::const_iterator iter = points.begin();
  while (iter != points.end()) {
    if ( p(dir) == (*iter)(dir) ) return true;
    ++iter;
  }
  return false;
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
  
  std::vector<Point> grid_LoPts; // store the lower bounds of all boxes
  std::vector<Point> grid_HiPts; // store the upper bounds of all boxes
  
  for(ProblemSpecP level_ps = grid_ps->findBlock("Level");
      level_ps != 0; level_ps = level_ps->findNextBlock("Level")){

     //find upper/lower corner
     for(ProblemSpecP box_ps = level_ps->findBlock("Box");
        box_ps != 0; box_ps = box_ps->findNextBlock("Box")){
       Point lower;
       Point upper;
       box_ps->require("lower", lower);
       box_ps->require("upper", upper);
       grid_LoPts.push_back(lower);
       grid_HiPts.push_back(upper);       
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
    std::stringstream origin_stream(origin);
    std::stringstream radius_stream(radius);
    double r,o[3];
    radius_stream >> r;
    origin_stream >> o[0] >> o[1] >> o[2];
    Point p(o[0],o[1],o[2]);

    if( !radius_stream || !origin_stream ) {
      std::cout <<  "WARNING: BoundCondReader.cc: stringstream failed..." << std::endl;
    }    
    
    //  bullet proofing-- origin must be on the same plane as the face

    bool isOnFace = false;
    
    if(plusMinusFaces == -1){    // x-, y-, z- faces
      isOnFace = is_on_face(p_dir,p,grid_LoPts);
    }
    
    if(plusMinusFaces == 1){     // x+, y+, z+ faces
      isOnFace = is_on_face(p_dir,p,grid_HiPts);      
    }    
    
    if(!isOnFace){
      ostringstream warn;
      warn<<"ERROR: Input file\n The Circle BC geometry is not correctly specified."
          << " The origin " << p << " must be on the same plane" 
          << " as face (" << fc <<"). Double check the origin and Level:box spec. \n\n";
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }
    
    
    if (origin == "" || radius == "") {
      ostringstream warn;
      warn<<"ERROR\n Circle BC geometry not correctly specified \n"
          << " you must specify origin [x,y,z] and radius [r] \n\n";
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }
    
    
    bcGeom = scinew CircleBCData(p,r);
  }
  else if (values.find("annulus") != values.end()) {
    fc = values["annulus"];
    whichPatchFace(fc, face_side, plusMinusFaces, p_dir);
    string origin = values["origin"];
    string in_radius = values["inner_radius"];
    string out_radius = values["outer_radius"];
    std::stringstream origin_stream(origin);
    std::stringstream in_radius_stream(in_radius);
    std::stringstream out_radius_stream(out_radius);
    double i_r,o_r,o[3];
    in_radius_stream >> i_r;
    out_radius_stream >> o_r;
    origin_stream >> o[0] >> o[1] >> o[2];
    Point p(o[0],o[1],o[2]);

    //  bullet proofing-- origin must be on the same plane as the face
    bool isOnFace = false;
    
    if(plusMinusFaces == -1){    // x-, y-, z- faces
      isOnFace = is_on_face(p_dir,p,grid_LoPts);
    }
    
    if(plusMinusFaces == 1){     // x+, y+, z+ faces
      isOnFace = is_on_face(p_dir,p,grid_HiPts);      
    }    
    
    if(!isOnFace){
      ostringstream warn;
      warn<<"ERROR: Input file\n The Annulus BC geometry is not correctly specified."
          << " The origin " << p << " must be on the same plane" 
          << " as face (" << fc <<"). Double check the origin and Level:box spec. \n\n";
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }
    
    if (origin == "" || in_radius == "" || out_radius == "" ) {
      ostringstream warn;
      warn<<"ERROR\n Annulus BC geometry not correctly specified \n"
          << " you must specify origin [x,y,z], inner_radius [r] outer_radius [r] \n\n";
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }
    bcGeom = scinew AnnulusBCData(p,i_r,o_r);
  }
  else if (values.find("ellipse") != values.end()) {
    fc = values["ellipse"];
    whichPatchFace(fc, face_side, plusMinusFaces, p_dir);
    string str_origin = values["origin"];
    string str_minor_radius = values["minor_radius"];    
    string str_major_radius = values["major_radius"];
    string str_angle = values["angle"];    
    std::stringstream origin_stream(str_origin);
    std::stringstream minor_radius_stream(str_minor_radius);
    std::stringstream major_radius_stream(str_major_radius);
    std::stringstream angle_stream(str_angle);
    double minor_r,major_r,origin[3], angle;
    minor_radius_stream >> minor_r;
    major_radius_stream >> major_r;
    origin_stream >> origin[0] >> origin[1] >> origin[2];
    Point p(origin[0],origin[1],origin[2]);
    angle_stream >> angle;
    
    //  bullet proofing-- origin must be on the same plane as the face
    bool isOnFace = false;
    
    if(plusMinusFaces == -1){    // x-, y-, z- faces
      isOnFace = is_on_face(p_dir,p,grid_LoPts);
    }
    
    if(plusMinusFaces == 1){     // x+, y+, z+ faces
      isOnFace = is_on_face(p_dir,p,grid_HiPts);      
    }    
    
    if(!isOnFace){
      ostringstream warn;
      warn<<"ERROR: Input file\n The Ellipse BC geometry is not correctly specified."
      << " The origin " << p << " must be on the same plane" 
      << " as face (" << fc <<"). Double check the origin and Level:box spec. \n\n";
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }
    
    if (major_r < minor_r) {
      ostringstream warn;
      warn<<"ERROR\n Ellipse BC geometry not correctly specified \n"
      << " Major radius must be larger than minor radius \n\n";
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }    
    
    if (str_origin == "" || str_minor_radius == "" || str_major_radius == "" ) {
      ostringstream warn;
      warn<<"ERROR\n Ellipse BC geometry not correctly specified \n"
      << " you must specify origin [x,y,z], inner_radius [r] outer_radius [r] \n\n";
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }
    
    bcGeom = scinew EllipseBCData(p,minor_r,major_r,fc,angle);
  }
  
  else if (values.find("rectangle") != values.end()) {
    fc = values["rectangle"];
    whichPatchFace(fc, face_side, plusMinusFaces, p_dir);
    string low = values["lower"];
    string up = values["upper"];
    std::stringstream low_stream(low), up_stream(up);
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
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }   
   
    if (low == "" || up == "") {
      ostringstream warn;
      warn<<"ERROR\n Rectangle BC geometry not correctly specified \n"
          << " you must specify lower [x,y,z] and upper[x,y,z] \n\n";
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }
    if ( (l.x() >  u.x() || l.y() >  u.y() || l.z() >  u.z()) ||
         (l.x() == u.x() && l.y() == u.y() && l.z() == u.z())){
      ostringstream warn;
      warn<<"ERROR\n Rectangle BC geometry not correctly specified \n"
          << " lower pt "<< l <<" upper pt " << u;
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }
    
    bcGeom = scinew RectangleBCData(l,u);
  }
  
  else {
    ostringstream warn;
    warn << "ERROR\n Boundary condition geometry not correctly specified "
      " Valid options (side, circle, rectangle, annulus";
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__);  
  }

  // name the boundary condition object:
  std::string bcname; 
  if (values.find("name") != values.end()){
    std::string name = values["name"];
    BCR_dbg << "Setting name to: " << name << endl;
    bcGeom->setBCName( name ); 
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

    std::multimap<int, BoundCondBase*> bctype_data;

    for (ProblemSpecP child = face_ps->findBlock("BCType"); child != 0;
        child = child->findNextBlock("BCType")) {
      int mat_id;
      BoundCondBase* bc;
      BoundCondFactory::create(child,bc,mat_id);
      BCR_dbg << "Inserting into mat_id = " << mat_id << " bc = " 
              <<  bc->getBCVariable() << " bctype = " 
              <<  bc->getBCType__NEW() 
              <<  " "  << bc  << endl;

      bctype_data.insert(pair<int,BoundCondBase*>(mat_id,bc->clone()));
      delete bc;
    }

    // Add the Auxillary boundary condition type
#if 1
    set<int> materials;
    for (multimap<int,BoundCondBase*>::const_iterator i = bctype_data.begin();
         i != bctype_data.end(); i++) {
      //      cout << "mat id = " << i->first << endl;
      materials.insert(i->first);
    }
    for (set<int>::const_iterator i = materials.begin(); i != materials.end();
         i++) {
      BoundCondBase* bc = scinew BoundCond<NoValue>("Auxiliary");
      bctype_data.insert(pair<int,BoundCondBase*>(*i,bc->clone()));
      delete bc;
    }
#endif

    // Print out all of the bcs just created
    multimap<int,BoundCondBase*>::const_iterator it;
    for (it = bctype_data.begin(); it != bctype_data.end(); it++) {
      BCR_dbg << "Getting out mat_id = " << it->first << " bc = " 
              << it->second->getBCVariable() <<  " bctype = " 
              << it->second->getBCType__NEW() << endl;
      //      cout << "mat = " << it -> first << " BoundCondBase address = " 
      //   << it->second << " bctype = " 
      //   << typeid(*(it->second)).name() << endl;
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
    d_BCReaderData[face_side].print();

    delete bcGeom;

    // Delete stuff in bctype_data
    multimap<int, BoundCondBase*>::const_iterator m_itr;
    for (m_itr = bctype_data.begin(); m_itr != bctype_data.end(); ++m_itr) {
      //      cout << "deleting BoundCondBase address = " << m_itr->second 
      //   << " bctype = " << typeid(*(m_itr->second)).name() << endl;
      delete m_itr->second;
    }
    bctype_data.clear();
    bcgeom_data.clear();
    
  }  // loop over faces

 

#if 1
  // Find the mat_id = "all" (-1) information and store it in each 
  // materials boundary condition section.
  BCR_dbg << "Add 'all' boundary condition information" << endl;
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
#if 1
    BCR_dbg << endl << "Combining BCGeometryTypes for face " << face << endl;
    for (bc_geom_itr = d_BCReaderData[face].d_BCDataArray.begin();
        bc_geom_itr != d_BCReaderData[face].d_BCDataArray.end();
        bc_geom_itr++) {
      BCR_dbg << "mat_id = " << bc_geom_itr->first << endl;
      d_BCReaderData[face].combineBCGeometryTypes_NEW(bc_geom_itr->first);
    }
#endif

    BCR_dbg << endl << "Printing out bcDataArray for face " << face 
      << " after adding 'all' . . " << endl;
    d_BCReaderData[face].print();
  }  // face loop
#endif

  // Need to take the individual boundary conditions and combine them into
  // a single different (side and the union of any holes (circles or
  // rectangles.  This only happens if there are more than 1 bc_data per
  // face.


  BCR_dbg << endl << "Before combineBCS() . . ." << endl << endl;
  for (Patch::FaceType face = Patch::startFace; 
      face <= Patch::endFace; face=Patch::nextFace(face)) {
    BCR_dbg << endl << endl << "Before Face . . ." << face << endl;
    d_BCReaderData[face].print();
  } 

  bulletProofing();

  combineBCS_NEW();

  BCR_dbg << endl << "After combineBCS() . . ." << endl << endl;
  for (Patch::FaceType face = Patch::startFace; 
      face <= Patch::endFace; face=Patch::nextFace(face)) {
    BCR_dbg << "After Face . . .  " << face << endl;
    d_BCReaderData[face].print();
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

    original.print();
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
        bcd_itr->first.print();

        
        if (count_if(bcd_itr->second.begin(),
                     bcd_itr->second.end(),
                     cmp_type<SideBCData>()) == 1 && 
            bcd_itr->second.size() == 1) {
          BCGeomBase* bc = bcd_itr->second[0];
          rearranged.addBCData(mat_id,bc->clone());
        } else {
          // Find the child that is the "side" bc
          BCGeomBase* side_bc = NULL;
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
          BCR_dbg << "mat_id = " << mat_id << endl;

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
    rearranged.print();

    // Reassign the rearranged data
    d_BCReaderData[face] = rearranged;
    
    BCR_dbg << endl << "Printing out rearranged from d_BCReaderData list" 
            << endl;

    d_BCReaderData[face].print();

  }


  BCR_dbg << endl << "Printing out in combineBCS()" << endl;
  for (Patch::FaceType face = Patch::startFace; 
       face <= Patch::endFace; face=Patch::nextFace(face)) {
    BCR_dbg << "After Face . . .  " << face << endl;
    d_BCReaderData[face].print();
  } 


}

void BoundCondReader::combineBCS_NEW()
{
  for (Patch::FaceType face = Patch::startFace; 
       face <= Patch::endFace; face=Patch::nextFace(face)) {
    BCR_dbg << endl << "Working on Face = " << face << endl;
    BCR_dbg << endl << "Original inputs" << endl;

    BCDataArray rearranged;
    BCDataArray& original = d_BCReaderData[face];

    original.print();
    BCR_dbg << endl;

    BCDataArray::bcDataArrayType::iterator mat_id_itr;
    for (mat_id_itr = original.d_BCDataArray.begin(); 
         mat_id_itr != original.d_BCDataArray.end(); 
         ++mat_id_itr) {
      int mat_id = mat_id_itr->first;

      BCR_dbg << "Mat ID = " << mat_id << endl;


      // Find all of the BCData types that are in a given BCDataArray
      vector<BCGeomBase*>::const_iterator vec_itr, side_index,other_index;

      typedef vector<BCGeomBase*> BCGeomBaseVec;
      BCGeomBaseVec& bcgeom_vec = mat_id_itr->second;

      // Don't do anything if the only BCGeomBase element is a SideBC
      if ( (bcgeom_vec.size() == 1) && 
           (count_if(bcgeom_vec.begin(),bcgeom_vec.end(),
                     cmp_type<SideBCData>()) == 1) ) {
        
        rearranged.addBCData(mat_id,bcgeom_vec[0]->clone());
      }

      // If there is more than one BCGeomBase element find the SideBC
      SideBCData* side_bc = NULL;
      DifferenceBCData* diff_bc = NULL;
      BCGeomBase* other_bc = NULL;

      if (bcgeom_vec.size() > 1) {

        int num_other = count_if(bcgeom_vec.begin(),bcgeom_vec.end(),
                                 not_type<SideBCData>()  );

        BCR_dbg << "num_other = " << num_other << endl << endl;

        if (num_other == 1) {

          side_index = find_if(bcgeom_vec.begin(),bcgeom_vec.end(),
                               cmp_type<SideBCData>());
          other_index = find_if(bcgeom_vec.begin(),bcgeom_vec.end(),
                                not_type<SideBCData>());
          
          side_bc = dynamic_cast<SideBCData*>((*side_index)->clone());
          other_bc = (*other_index)->clone();

          diff_bc = scinew DifferenceBCData(side_bc,other_bc);

          diff_bc->setBCName( side_bc->getBCName() ); //make sure the new piece has the right name

          rearranged.addBCData(mat_id,diff_bc->clone());
          rearranged.addBCData(mat_id,other_bc->clone());
          delete diff_bc;
          delete side_bc;
          delete other_bc;

        } else {

          UnionBCData* union_bc = scinew UnionBCData();
          remove_copy_if(bcgeom_vec.begin(),bcgeom_vec.end(),
                         back_inserter(union_bc->child),cmp_type<SideBCData>());
          
          side_index = find_if(bcgeom_vec.begin(),bcgeom_vec.end(),
                               cmp_type<SideBCData>());

          side_bc = dynamic_cast<SideBCData*>((*side_index)->clone());
          diff_bc = scinew DifferenceBCData(side_bc,union_bc->clone());

          diff_bc->setBCName( side_bc->getBCName() ); //make sure the new piece has the right name

          rearranged.addBCData(mat_id,diff_bc->clone());
          delete side_bc;
          delete diff_bc;
          for (vec_itr=union_bc->child.begin();vec_itr!=union_bc->child.end();
               ++vec_itr){
            rearranged.addBCData(mat_id,(*vec_itr)->clone());
          }

          //   delete union_bc;

        }

      }
      for_each(bcgeom_vec.begin(),bcgeom_vec.end(),delete_object<BCGeomBase>());
      bcgeom_vec.clear();
    }
    BCR_dbg << endl << "Printing out rearranged list" << endl;
    rearranged.print();


    d_BCReaderData[face] = rearranged;

    BCR_dbg << endl << "Printing out rearranged from d_BCReaderData list" 
            << endl;

    d_BCReaderData[face].print();
  }
        
  
  BCR_dbg << endl << "Printing out in combineBCS()" << endl;
  for (Patch::FaceType face = Patch::startFace; 
       face <= Patch::endFace; face=Patch::nextFace(face)) {
    BCR_dbg << "After Face . . .  " << face << endl;
    d_BCReaderData[face].print();
  } 

}

bool BoundCondReader::compareBCData(BCGeomBase* b1, BCGeomBase* b2)
{
  return false;
}

//______________________________________________________________________
//
void BoundCondReader::bulletProofing()
{
  for (Patch::FaceType face = Patch::startFace; 
       face <= Patch::endFace; face=Patch::nextFace(face)) {

    BCDataArray& original = d_BCReaderData[face];

    BCDataArray::bcDataArrayType::iterator mat_id_itr;
    for (mat_id_itr = original.d_BCDataArray.begin(); 
         mat_id_itr != original.d_BCDataArray.end(); 
         ++mat_id_itr) {

      typedef vector<BCGeomBase*> BCGeomBaseVec;
      BCGeomBaseVec& bcgeom_vec = mat_id_itr->second;

      //__________________________________
      // There must be 1 and only 1 side BC specified
      int nSides = count_if(bcgeom_vec.begin(),bcgeom_vec.end(), cmp_type<SideBCData>()  );
      
      if ( nSides != 1 ){
        ostringstream warn;
        warn<<"ERROR: <BoundaryConditions> <"<< Patch::getFaceName(face) << ">\n" 
            << "There must be 1 and only 1 side boundary condition specified \n\n";
        throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
      }
    }  // BCDataArray
  }  // patch faces
}




namespace Uintah {

  void print(BCGeomBase* p) {
    BCR_dbg << "type = " << typeid(*p).name() << endl;
  }
  
}



