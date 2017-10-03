/*
 * The MIT License
 *
 * Copyright (c) 1997-2017 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
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
#include <Core/Grid/BoundaryConditions/RectangulusBCData.h>
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

//-------------------------------------------------------------------------------------------------

BoundCondReader::BoundCondReader() 
{
}

//-------------------------------------------------------------------------------------------------

BoundCondReader::~BoundCondReader()
{
  //cout << "Calling BoundCondReader destructor" << endl;
}

//-------------------------------------------------------------------------------------------------

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

//-------------------------------------------------------------------------------------------------

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

//-------------------------------------------------------------------------------------------------

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
  
  for( ProblemSpecP level_ps = grid_ps->findBlock("Level"); level_ps != nullptr; level_ps = level_ps->findNextBlock("Level") ) {

     //find upper/lower corner
     for( ProblemSpecP box_ps = level_ps->findBlock("Box"); box_ps != nullptr; box_ps = box_ps->findNextBlock("Box") ) {
       Point lower;
       Point upper;
       box_ps->require( "lower", lower );
       box_ps->require( "upper", upper );
       grid_LoPts.push_back( lower );
       grid_HiPts.push_back( upper );       
       grid_LoPt = Min( lower, grid_LoPt );
       grid_HiPt = Max( upper, grid_HiPt );
     }
   }

  map<string,string> values;
  face_ps->getAttributes( values );
      
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
  else if (values.find("rectangulus") != values.end()) {
    fc = values["rectangulus"];
    whichPatchFace(fc, face_side, plusMinusFaces, p_dir);
    string in_low = values["inner_lower"];
    string in_up = values["inner_upper"];
    string out_low = values["outer_lower"];
    string out_up = values["outer_upper"];
    std::stringstream in_low_stream(in_low), in_up_stream(in_up);
    std::stringstream out_low_stream(out_low), out_up_stream(out_up);
    double in_lower[3],in_upper[3];
    double out_lower[3],out_upper[3];
    in_low_stream >> in_lower[0] >> in_lower[1] >> in_lower[2];
    out_low_stream >> out_lower[0] >> out_lower[1] >>out_lower[2];
    in_up_stream >> in_upper[0] >> in_upper[1] >> in_upper[2];
    out_up_stream >> out_upper[0] >> out_upper[1] >> out_upper[2];
    Point l(in_lower[0],in_lower[1],in_lower[2]),u(in_upper[0],in_upper[1],in_upper[2]);
    Point l2(out_lower[0],out_lower[1],out_lower[2]),u2(out_upper[0],out_upper[1],out_upper[2]);
   
    //  bullet proofing-- both rectangles must be on the same plane as the face
    bool isOnFace1 = false;
    bool isOnFace2 = false;

    if(plusMinusFaces == -1){    // x-, y-, z- faces
      isOnFace1 = is_on_face(p_dir,l,grid_LoPts) && is_on_face(p_dir,u,grid_LoPts);
      isOnFace2 = is_on_face(p_dir,l2,grid_LoPts) && is_on_face(p_dir,u2,grid_LoPts);
    }
    
    if(plusMinusFaces == 1){     // x+, y+, z+ faces
      isOnFace1 = is_on_face(p_dir,l,grid_HiPts) && is_on_face(p_dir,u,grid_HiPts);      
      isOnFace2 = is_on_face(p_dir,l2,grid_HiPts) && is_on_face(p_dir,u2,grid_HiPts);      
    }    
    
    if(!isOnFace1 || !isOnFace2){
      ostringstream warn;
      warn<<"ERROR: Input file\n The rectangle BC geometry is not correctly specified."
          << " The low " << l << " high " << u << " points must be on the same plane" 
          << " The low " << l2 << " high " << u2 << " points must be on the same plane" 
          << " as face (" << fc <<"). Double check against and Level:box spec. \n\n";
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }   
   
    if (in_low == "" || in_up == "") {
      ostringstream warn;
      warn<<"ERROR\n Rectangle BC geometry not correctly specified \n"
          << " you must specify in_lower [x,y,z] and in_upper[x,y,z] \n\n";
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }
    if (out_low == "" || out_up == "") {
      ostringstream warn;
      warn<<"ERROR\n Rectangle BC geometry not correctly specified \n"
          << " you must specify out_lower [x,y,z] and out_upper[x,y,z] \n\n";
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }
    if ( (l.x() >  u.x() || l.y() >  u.y() || l.z() >  u.z()) ||
         (l.x() == u.x() && l.y() == u.y() && l.z() == u.z())){
      ostringstream warn;
      warn<<"ERROR\n Rectangle BC geometry not correctly specified \n"
          << " lower pt "<< l <<" upper pt " << u;
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }
    if ( (l2.x() >  u2.x() || l2.y() >  u2.y() || l2.z() >  u2.z()) ||
         (l2.x() == u2.x() && l2.y() == u2.y() && l2.z() == u2.z())){
      ostringstream warn;
      warn<<"ERROR\n Rectangle BC geometry not correctly specified \n"
          << " lower pt "<< l <<" upper pt " << u;
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }
    bcGeom = scinew RectangulusBCData(l,u,l2,u2);
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
      << " you must specify origin [x,y,z], minor_radius [r] major_radius [r] \n\n";
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
    bool isOnFace = false;

    if(plusMinusFaces == -1){    // x-, y-, z- faces
      isOnFace = is_on_face(p_dir,l,grid_LoPts) && is_on_face(p_dir,u,grid_LoPts);
    }
    
    if(plusMinusFaces == 1){     // x+, y+, z+ faces
      isOnFace = is_on_face(p_dir,l,grid_HiPts) && is_on_face(p_dir,u,grid_HiPts);      
    }    
    
    if(!isOnFace){
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
  
  // get the bctype - mainly used by wasatch:
  if (values.find("type") != values.end()){
    std::string bndType = values["type"];
    BCR_dbg << "Setting bc type to: " << bndType << endl;
    bcGeom->setBndType( bndType );
  }

  if (face_ps->findBlock("ParticleBC")) {
    ProblemSpecP particleBCps = face_ps->findBlock("ParticleBC");
    ProblemSpecP pWallBC = particleBCps->findBlock("Wall");
    ProblemSpecP pInletBC= particleBCps->findBlock("Inlet");
    BCGeomBase::ParticleBndSpec pBndSpec;
    if (pWallBC) {
      pBndSpec.bndType = BCGeomBase::ParticleBndSpec::WALL;
      std::string wallType;
      pWallBC->getAttribute("walltype",wallType);
      if (wallType=="Elastic") {
        pBndSpec.wallType = BCGeomBase::ParticleBndSpec::ELASTIC;
        pBndSpec.restitutionCoef = 1.0;
      } else if (wallType=="Inelastic") {
        pBndSpec.wallType = BCGeomBase::ParticleBndSpec::INELASTIC;
        pBndSpec.restitutionCoef = 0.0;
      } else if (wallType=="PartiallyElastic") {
        pBndSpec.wallType = BCGeomBase::ParticleBndSpec::PARTIALLYELASTIC;
        pWallBC->get("Restitution", pBndSpec.restitutionCoef);
      }
    } else if (pInletBC) {
      pBndSpec.bndType = BCGeomBase::ParticleBndSpec::INLET;
      pInletBC->get("ParticlesPerSecond", pBndSpec.particlesPerSec);
    }
    bcGeom->setParticleBndSpec(pBndSpec);
  }

  BCR_dbg << "Face = " << fc << endl;
  return bcGeom;
}

//-------------------------------------------------------------------------------------------------

Uintah::Point moveToClosestNode(const Uintah::LevelP level, const int facedir, const int plusMinusFaces, const Uintah::Point& p0)
{
  using namespace Uintah;
  //
  // now find the closest node
  Vector halfdx = level->dCell()/2.0;
  Point newPos = p0;

  // find the closest cell
  IntVector cellIdx = level->getCellIndex(p0); // find the closest cell to the center of this circle
  Point closestCell = level->getCellPosition(cellIdx);
  
  // move the appropriate coordinate to the closest cell
  newPos(facedir) = closestCell(facedir);
  
  Point leftNode = newPos;
  Point rightNode = newPos;
  leftNode(facedir) -= halfdx[facedir];
  rightNode(facedir) +=halfdx[facedir];
  
  Vector diffLeft = p0 - leftNode;
  Vector diffRight = rightNode - p0;
  
  if (diffRight.length() > diffLeft.length()) { // object is closer to the left node
    newPos(facedir) -= halfdx[facedir]; // move the circle to the closest layer of nodes
  } else if (diffRight.length() < diffLeft.length()) {
    newPos(facedir) += halfdx[facedir]; // move the circle to the closest layer of nodes
  } else {
    newPos(facedir) += plusMinusFaces*halfdx[facedir]; // move the circle to the closest layer of nodes
  }
  return newPos;
}

//-------------------------------------------------------------------------------------------------

BCGeomBase* BoundCondReader::createInteriorBndBoundaryConditionFace(ProblemSpecP& face_ps,
                                                                  const ProblemSpecP& grid_ps,
                                                                  Patch::FaceType& face_side,
                                                                  const Uintah::LevelP level)
{
  
  // Determine the Level 0 grid high and low points, need by
  // the bullet proofing
  Point grid_LoPt(1e30, 1e30, 1e30);
  Point grid_HiPt(-1e30, -1e30, -1e30);
  
  std::vector<Point> grid_LoPts; // store the lower bounds of all boxes
  std::vector<Point> grid_HiPts; // store the upper bounds of all boxes
  
  for( ProblemSpecP level_ps = grid_ps->findBlock("Level"); level_ps != nullptr; level_ps = level_ps->findNextBlock("Level") ) {
    
    // Find upper/lower corner:
    for( ProblemSpecP box_ps = level_ps->findBlock("Box"); box_ps != nullptr; box_ps = box_ps->findNextBlock("Box") ) {
      Point lower;
      Point upper;
      box_ps->require( "lower", lower );
      box_ps->require( "upper", upper );
      grid_LoPts.push_back( lower );
      grid_HiPts.push_back( upper );
      grid_LoPt = Min( lower, grid_LoPt );
      grid_HiPt = Max( upper, grid_HiPt );
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
    ostringstream warn;
    warn<<"ERROR: You cannot specify an internal side boundary condition.";
    throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
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
    Point p0(o[0],o[1],o[2]);
    Point p = moveToClosestNode(level, p_dir, plusMinusFaces,p0);
    if( !radius_stream || !origin_stream ) {
      std::cout <<  "WARNING: BoundCondReader.cc: stringstream failed..." << std::endl;
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
    Point p0(o[0],o[1],o[2]);
    if (origin == "" || in_radius == "" || out_radius == "" ) {
      ostringstream warn;
      warn<<"ERROR\n Annulus BC geometry not correctly specified \n"
      << " you must specify origin [x,y,z], inner_radius [r] outer_radius [r] \n\n";
      throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
    }
    Point p = moveToClosestNode(level, p_dir, plusMinusFaces, p0);
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
    Point p0(origin[0],origin[1],origin[2]);
    Point p = moveToClosestNode(level, p_dir, plusMinusFaces,p0);
    angle_stream >> angle;
    
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
    Point l0(lower[0],lower[1],lower[2]),u0(upper[0],upper[1],upper[2]);
    Point l = moveToClosestNode(level, p_dir, plusMinusFaces,l0);
    Point u = moveToClosestNode(level, p_dir, plusMinusFaces,u0);
    
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
  
  // get the bctype - mainly used by wasatch:
  if (values.find("type") != values.end()){
    std::string bndType = values["type"];
    BCR_dbg << "Setting bc type to: " << bndType << endl;
    bcGeom->setBndType( bndType );
  }
  
  if (face_ps->findBlock("ParticleBC")) {
    ProblemSpecP particleBCps = face_ps->findBlock("ParticleBC");
    ProblemSpecP pWallBC = particleBCps->findBlock("Wall");
    ProblemSpecP pInletBC= particleBCps->findBlock("Inlet");
    BCGeomBase::ParticleBndSpec pBndSpec;
    if (pWallBC) {
      pBndSpec.bndType = BCGeomBase::ParticleBndSpec::WALL;
      std::string wallType;
      pWallBC->getAttribute("walltype",wallType);
      if (wallType=="Elastic") {
        pBndSpec.wallType = BCGeomBase::ParticleBndSpec::ELASTIC;
        pBndSpec.restitutionCoef = 1.0;
      } else if (wallType=="Inelastic") {
        pBndSpec.wallType = BCGeomBase::ParticleBndSpec::INELASTIC;
        pBndSpec.restitutionCoef = 0.0;
      } else if (wallType=="PartiallyElastic") {
        pBndSpec.wallType = BCGeomBase::ParticleBndSpec::PARTIALLYELASTIC;
        pWallBC->get("Restitution", pBndSpec.restitutionCoef);
      }
    } else if (pInletBC) {
      pBndSpec.bndType = BCGeomBase::ParticleBndSpec::INLET;
      pInletBC->get("ParticlesPerSecond", pBndSpec.particlesPerSec);
    }
    bcGeom->setParticleBndSpec(pBndSpec);
  }
  
  BCR_dbg << "Face = " << fc << endl;
  return bcGeom;
}

//-------------------------------------------------------------------------------------------------

void
BoundCondReader::read(ProblemSpecP& bc_ps, const ProblemSpecP& grid_ps, const Uintah::LevelP level)
{
  readDomainBCs(bc_ps,grid_ps);
  readInteriorBndBCs(bc_ps,grid_ps, level);
}

//-------------------------------------------------------------------------------------------------

void
BoundCondReader::readInteriorBndBCs(ProblemSpecP& bc_ps, const ProblemSpecP& grid_ps,const Uintah::LevelP level)
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
  
  string defaultMat="";
  ProblemSpecP defaultMatSpec = bc_ps->findBlock("DefaultMaterial");
  if(defaultMatSpec) bc_ps->get("DefaultMaterial", defaultMat);
  
  for( ProblemSpecP face_ps = bc_ps->findBlock("InteriorFace"); face_ps != nullptr; face_ps=face_ps->findNextBlock("InteriorFace") ) {
    
    Patch::FaceType face_side;
    BCGeomBase* bcGeom = createInteriorBndBoundaryConditionFace(face_ps,grid_ps,face_side, level);
    
    std::string face_label = "none";
    face_ps->getAttribute("name",face_label);
    BCR_dbg << "Face Label = " << face_label << "\n";
    
    BCR_dbg << endl << endl << "Face = " << face_side << " Geometry type = " << typeid(*bcGeom).name() << " " << bcGeom << "\n";
    
    std::multimap<int, BoundCondBase*> bctype_data;
    
    for( ProblemSpecP child = face_ps->findBlock("BCType"); child != nullptr; child = child->findNextBlock("BCType") ) {
      int mat_id;
      
      map<string,string> bc_attr;
      child->getAttributes( bc_attr );
      bool foundMatlID = ( bc_attr.find("id") != bc_attr.end() );
      
      if (!foundMatlID) {
        if (defaultMat == "") {
          SCI_THROW(ProblemSetupException("ERROR: No material id was specified in the BCType tag and I could not find a DefaulMaterial to use! Please revise your input file.", __FILE__, __LINE__));
        }
        else {
          mat_id = (defaultMat == "all") ? -1 : atoi(defaultMat.c_str());
        }
      }
      else {
        string id = bc_attr["id"];
        mat_id = (id == "all") ? -1 : atoi(id.c_str());
      }
      
      BoundCondBase* bc;
      BoundCondFactory::create(child, bc, mat_id, face_label);
      BCR_dbg << "Inserting into mat_id = " << mat_id << " bc = "
      <<  bc->getBCVariable() << " bctype = "
      <<  bc->getBCType()
      <<  " "  << bc  << endl;
      
      bctype_data.insert(pair<int,BoundCondBase*>(mat_id,bc->clone()));
      delete bc;
    }
    
    // Print out all of the bcs just created
    multimap<int,BoundCondBase*>::const_iterator it;
    for (it = bctype_data.begin(); it != bctype_data.end(); it++) {
      BCR_dbg << "Getting out mat_id = " << it->first << " bc = "
      << it->second->getBCVariable() <<  " bctype = "
      << it->second->getBCType() << endl;
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
    
    //____________________________________________________________________
    // CAUTION! tsaad: If NO BCs have been specified on this boundary, then NO iterators for that boundary
    // will be added. The next if-statement circumvents that problem for lack of a better design.
    // This is done to reduce input-file clutter. For example, for a constant density flow problem
    // a stationary-wall boundary is well defined and there's no reason for the user to input
    // any BCs there. To be able to set BCs through the code, we still need access to the iterator
    // for that boundary.
    if (bctype_data.size() == 0) {
      bcgeom_data[-1] = bcGeom->clone();
    }
    //-------------------------------------------------------------------
    
    for (bc_geom_itr = bcgeom_data.begin(); bc_geom_itr != bcgeom_data.end();
         bc_geom_itr++) {
      d_interiorBndBCReaderData[face_side].addBCData(bc_geom_itr->first,
                                          bcgeom_data[bc_geom_itr->first]->clone());
      delete bc_geom_itr->second;
    }
    
    BCR_dbg << "Printing out bcDataArray . . " << "for face_side = " << face_side << endl;
    d_interiorBndBCReaderData[face_side].print();
    
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
    
    mat_all_itr = d_interiorBndBCReaderData[face].d_BCDataArray.find(-1);
    if (mat_all_itr != d_interiorBndBCReaderData[face].d_BCDataArray.end())
      for (bc_geom_itr = d_interiorBndBCReaderData[face].d_BCDataArray.begin();
           bc_geom_itr != d_interiorBndBCReaderData[face].d_BCDataArray.end();
           bc_geom_itr++) {
        if (bc_geom_itr != mat_all_itr) {
          vector<BCGeomBase*>::const_iterator itr;
          for (itr = mat_all_itr->second.begin();
               itr != mat_all_itr->second.end(); ++itr)
            d_interiorBndBCReaderData[face].addBCData(bc_geom_itr->first,
                                           (*itr)->clone());
        }
      }
#if 1
    BCR_dbg << endl << "Combining BCGeometryTypes for face " << face << endl;
    for (bc_geom_itr = d_interiorBndBCReaderData[face].d_BCDataArray.begin();
         bc_geom_itr != d_interiorBndBCReaderData[face].d_BCDataArray.end();
         bc_geom_itr++) {
      BCR_dbg << "mat_id = " << bc_geom_itr->first << endl;
      //d_interiorBndBCReaderData[face].combineBCGeometryTypes_NEW(bc_geom_itr->first);
    }
#endif
    
    BCR_dbg << endl << "Printing out bcDataArray for face " << face
    << " after adding 'all' . . " << endl;
    d_interiorBndBCReaderData[face].print();
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
    d_interiorBndBCReaderData[face].print();
  }
  
  //bulletProofing();
  
  //combineBCS();
  
  BCR_dbg << endl << "After combineBCS() . . ." << endl << endl;
  for (Patch::FaceType face = Patch::startFace;
       face <= Patch::endFace; face=Patch::nextFace(face)) {
    BCR_dbg << "After Face . . .  " << face << endl;
    d_interiorBndBCReaderData[face].print();
  } 
  
}

//-------------------------------------------------------------------------------------------------

void
BoundCondReader::readDomainBCs(ProblemSpecP& bc_ps, const ProblemSpecP& grid_ps)
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

  string defaultMat="";
  ProblemSpecP defaultMatSpec = bc_ps->findBlock("DefaultMaterial");
  if(defaultMatSpec) bc_ps->get("DefaultMaterial", defaultMat);
  
  for( ProblemSpecP face_ps = bc_ps->findBlock("Face"); face_ps != nullptr; face_ps=face_ps->findNextBlock("Face") ) {

    Patch::FaceType face_side;
    BCGeomBase* bcGeom = createBoundaryConditionFace(face_ps,grid_ps,face_side);

    std::string face_label = "none";
    face_ps->getAttribute("name",face_label);
    BCR_dbg << "Face Label = " << face_label << std::endl;
    
    BCR_dbg << endl << endl << "Face = " << face_side << " Geometry type = " 
      << typeid(*bcGeom).name() << " " << bcGeom << endl;

    std::multimap<int, BoundCondBase*> bctype_data;

    for( ProblemSpecP child = face_ps->findBlock( "BCType" ); child != nullptr; child = child->findNextBlock( "BCType" ) ) {
      int mat_id;
      
      map<string,string> bc_attr;
      child->getAttributes(bc_attr);
      bool foundMatlID = ( bc_attr.find("id") != bc_attr.end() );
      
      if (!foundMatlID) {
        if (defaultMat == "") {
          SCI_THROW(ProblemSetupException("ERROR: No material id was specified in the BCType tag and I could not find a DefaulMaterial to use! Please revise your input file.", __FILE__, __LINE__));
        }
        else {
          mat_id = (defaultMat == "all") ? -1 : atoi(defaultMat.c_str());
        }
      }
      else {
        string id = bc_attr["id"];
        mat_id = (id == "all") ? -1 : atoi(id.c_str());
      }
      
      BoundCondBase* bc;
      BoundCondFactory::create(child, bc, mat_id, face_label);
      BCR_dbg << "Inserting into mat_id = " << mat_id << " bc = " 
              <<  bc->getBCVariable() << " bctype = " 
              <<  bc->getBCType()
              <<  " "  << bc  << endl;

      bctype_data.insert(pair<int,BoundCondBase*>(mat_id,bc->clone()));
      delete bc;
    }

//    // Add the Auxillary boundary condition type
//#if 1
//    set<int> materials;
//    for (multimap<int,BoundCondBase*>::const_iterator i = bctype_data.begin();
//         i != bctype_data.end(); i++) {
//      //      cout << "mat id = " << i->first << endl;
//      materials.insert(i->first);
//    }
//    for (set<int>::const_iterator i = materials.begin(); i != materials.end();
//         i++) {
//      BoundCondBase* bc = scinew BoundCond<NoValue>("Auxiliary");
//      bctype_data.insert(pair<int,BoundCondBase*>(*i,bc->clone()));
//      delete bc;
//    }
//#endif

    // Print out all of the bcs just created
    multimap<int,BoundCondBase*>::const_iterator it;
    for (it = bctype_data.begin(); it != bctype_data.end(); it++) {
      BCR_dbg << "Getting out mat_id = " << it->first << " bc = " 
              << it->second->getBCVariable() <<  " bctype = " 
              << it->second->getBCType() << endl;
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
    
    //____________________________________________________________________
    // CAUTION! tsaad: If NO BCs have been specified on this boundary, then NO iterators for that boundary
    // will be added. The next if-statement circumvents that problem for lack of a better design.
    // This is done to reduce input-file clutter. For example, for a constant density flow problem
    // a stationary-wall boundary is well defined and there's no reason for the user to input
    // any BCs there. To be able to set BCs through the code, we still need access to the iterator
    // for that boundary.
    if (bctype_data.size() == 0) {
      bcgeom_data[-1] = bcGeom->clone();
    }
    //-------------------------------------------------------------------
    
    for (bc_geom_itr = bcgeom_data.begin(); bc_geom_itr != bcgeom_data.end();
        bc_geom_itr++) {
      d_BCReaderData[face_side].addBCData(bc_geom_itr->first,
          bcgeom_data[bc_geom_itr->first]->clone());
      delete bc_geom_itr->second;
    }

    BCR_dbg << "Printing out bcDataArray . . " << "for face_side = " << face_side << endl;
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

  combineBCS();

  BCR_dbg << endl << "After combineBCS() . . ." << endl << endl;
  for (Patch::FaceType face = Patch::startFace; 
      face <= Patch::endFace; face=Patch::nextFace(face)) {
    BCR_dbg << "After Face . . .  " << face << endl;
    d_BCReaderData[face].print();
  } 
  
}

//-------------------------------------------------------------------------------------------------

const BCDataArray BoundCondReader::getBCDataArray(Patch::FaceType& face) const
{
  map<Patch::FaceType,BCDataArray > m = this->d_BCReaderData;
  return m[face];
}

//-------------------------------------------------------------------------------------------------

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
      SideBCData* side_bc = nullptr;
      DifferenceBCData* diff_bc = nullptr;
      BCGeomBase* other_bc = nullptr;
      UnionBCData* union_bc = nullptr;

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
          diff_bc->setBndType( side_bc->getBndType() ); //make sure the new piece has the correct boundary type
          diff_bc->setParticleBndSpec(side_bc->getParticleBndSpec());
          
          rearranged.addBCData(mat_id,diff_bc->clone());
          rearranged.addBCData(mat_id,other_bc->clone());
          delete diff_bc;
          delete side_bc;
          delete other_bc;

        } else {

          union_bc = scinew UnionBCData();
          // Need to clone the RectangleBC that are being inserted 
          // into the UnionBC 
          // remove_copy_if(bcgeom_vec.begin(),bcgeom_vec.end(),
          //            back_inserter(union_bc->child),cmp_type<SideBCData>());

          for (BCGeomBaseVec::iterator itr = bcgeom_vec.begin(); 
               itr != bcgeom_vec.end(); ++itr) {
            if (!cmp_type<SideBCData>()(*itr))
              union_bc->child.push_back((*itr)->clone());
          }
          
          side_index = find_if(bcgeom_vec.begin(),bcgeom_vec.end(),
                               cmp_type<SideBCData>());

          side_bc = dynamic_cast<SideBCData*>((*side_index)->clone());

          UnionBCData* union_bc_clone = union_bc->clone(); 

          diff_bc = scinew DifferenceBCData(side_bc,union_bc_clone);

          diff_bc->setBCName( side_bc->getBCName() ); //make sure the new piece has the right name
          diff_bc->setBndType( side_bc->getBndType() ); //make sure the new piece has the correct boundary type
          diff_bc->setParticleBndSpec(side_bc->getParticleBndSpec());
          
          rearranged.addBCData(mat_id,diff_bc->clone());
          delete side_bc;
          delete diff_bc;
          for (vec_itr=union_bc_clone->child.begin();vec_itr!=union_bc_clone->child.end();
               ++vec_itr){
            rearranged.addBCData(mat_id,(*vec_itr)->clone());
          }

          
          delete union_bc;
          delete union_bc_clone;
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

//-------------------------------------------------------------------------------------------------

bool BoundCondReader::compareBCData(BCGeomBase* b1, BCGeomBase* b2)
{
  return false;
}

//-------------------------------------------------------------------------------------------------

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
        warn<<"ERROR: <BoundaryConditions> <"<< Patch::getFaceName(face) << ">  nSides=" <<nSides<<  "but should equal 1!\n" 
            << "There must be 1 and only 1 side boundary condition specified \n\n";
        throw ProblemSetupException(warn.str(), __FILE__, __LINE__);
      }
    }  // BCDataArray
  }  // patch faces
}

//-------------------------------------------------------------------------------------------------

namespace Uintah {

  void print(BCGeomBase* p) {
    BCR_dbg << "type = " << typeid(*p).name() << endl;
  }
}

//-------------------------------------------------------------------------------------------------
