/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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

//////////////////////////////////////////////////////////////////

// export SCI_DEBUG="BCR_DBG:+"
static DebugStream BCR_dbg( "BCR_DBG", false ); // FIXME set back to false when done debugging

// Used for bullet proofing
static vector<Point> g_grid_LoPts; // store the lower bounds of all boxes
static vector<Point> g_grid_HiPts; // store the upper bounds of all boxes
static Point g_grid_LoPt(1e30, 1e30, 1e30);
static Point g_grid_HiPt(-1e30, -1e30, -1e30); 

//////////////////////////////////////////////////////////////////

typedef map< string, BCGeomBase* > m_s2bcgb;

//////////////////////////////////////////////////////////////////

BoundCondReader::BoundCondReader() 
{
}

BoundCondReader::~BoundCondReader()
{
  //cout << "Calling BoundCondReader destructor" << endl;
}

//////////////////////////////////////////////////////////////////

// Given a set of lower or upper bounds for multiple boxes (points),
// this function checks if Point p (usually center of circle, ellipse, or annulus)
// is on a given face on any of the boxes.
//
static
bool
is_on_face( const int             dir, 
            const Point         & p,
            const vector<Point> & points )
{
  vector<Point>::const_iterator iter = points.begin();
  while( iter != points.end() ) {

    //cout << "checking: " << p(dir) << " vs " << (*iter)(dir) << "\n";

    if ( p(dir) == (*iter)(dir) ) return true;
    ++iter;
  }
  return false;
}

//
// whichPatchFace()
//
//    Takes the string in 'fc' and returns enums/scalars that correspond to it.
//
//      fc             : "x-", "x+", "y-", etc
//      face_side      : [Output] The enumerated value that corresponds to the string "x-", etc.
//      plusMinusFaces : [Output] Either -1 or +1 depending on the -/+ in "x-", etc.
//      p_dir          : [Output] 1 for x faces, 2 for y faces, and 3 for z faces.
//
static
void
whichPatchFace( const string          & fc,
                      Patch::FaceType & face_side,
                      int             & plusMinusFaces,
                      int             & p_dir)
{
  if(      fc == "x-" ) {
    plusMinusFaces = -1;
    p_dir = 0;
    face_side = Patch::xminus;
  }
  else if( fc == "x+" ) {
    plusMinusFaces = 1;
    p_dir = 0;
    face_side = Patch::xplus;
  }
  else if( fc == "y-"){
    plusMinusFaces = -1;
    p_dir = 1;
    face_side = Patch::yminus;
  }
  else if( fc == "y+" ){
    plusMinusFaces = 1;
    p_dir = 1;
    face_side = Patch::yplus;
  }
  else if( fc == "z-" ){
    plusMinusFaces = -1;
    p_dir = 2;
    face_side = Patch::zminus;
  }
  else if( fc == "z+" ){
    plusMinusFaces = 1;
    p_dir = 2;
    face_side = Patch::zplus;
  }
  else {
    throw ProblemSetupException( "Bad face: " + fc, __FILE__, __LINE__ );
  }
}

static
BCGeomBase *
createCircle( map<string,string> values, const string & direction, const string & name )
{
  Patch::FaceType face_side;
  int             plusMinusFaces;
  int             p_dir;

  if( values[ "origin" ] == "" || values[ "radius" ] == "" ) {
    ostringstream err;
    err << "ERROR: Circle BC geometry not correctly specified.\n"
        << "You must specify origin [x,y,z] and radius [r].\n";
    throw ProblemSetupException( err.str(), __FILE__, __LINE__ );
  }

  whichPatchFace( direction, face_side, plusMinusFaces, p_dir );

  std::stringstream origin_stream( values[ "origin" ] );
  std::stringstream radius_stream( values[ "radius" ] );

  double radius, origin_pts[3];

  radius_stream >> radius;
  origin_stream >> origin_pts[0] >> origin_pts[1] >> origin_pts[2];

  Point origin( origin_pts[0], origin_pts[1], origin_pts[2] );

  if( !radius_stream || !origin_stream ) {
    throw ProblemSetupException( "BoundCondReader.cc: stringstream failed...\n", __FILE__, __LINE__ );
  }    
    
  // Bullet proofing -- origin must be on the same plane as the face.

  bool isOnFace = false;
    
  if( plusMinusFaces == -1 ) {    // x-, y-, z- faces
    isOnFace = is_on_face( p_dir, origin, g_grid_LoPts );
  }
    
  if( plusMinusFaces == 1 ){     // x+, y+, z+ faces
    isOnFace = is_on_face( p_dir, origin, g_grid_HiPts );      
  }    
    
  if( !isOnFace ) {
    ostringstream err;
    err << "ERROR: The Circle BC geometry is not correctly specified.\n"
        << "The origin " << origin << " must be on the same plane " 
        << "as face (" << direction << "). Double check the origin and Level:box spec.\n\n";
    throw ProblemSetupException( err.str(), __FILE__, __LINE__ );
  }
    
  return scinew CircleBCData( origin, radius, name, face_side );

} // end createCircle()

static
BCGeomBase *
createRectangle( map<string,string> values, const string & direction, const string & name )
{
  Patch::FaceType face_side;
  int             plusMinusFaces;
  int             p_dir;

  if( values[ "lower" ] == "" || values[ "upper" ] == "" ) {
    ostringstream err;
    err << "ERROR: Rectangle BC geometry not correctly specified.\n"
        << "You must specify both a 'lower' and 'upper' point [x,y,z].\n";
    throw ProblemSetupException( err.str(), __FILE__, __LINE__ );
  }

  whichPatchFace( direction, face_side, plusMinusFaces, p_dir );

  std::stringstream lower_stream( values[ "lower" ] );
  std::stringstream upper_stream( values[ "upper" ] );

  double lower_pts[3], upper_pts[3];

  lower_stream >> lower_pts[0] >> lower_pts[1] >> lower_pts[2];
  upper_stream >> upper_pts[0] >> upper_pts[1] >> upper_pts[2];

  Point lower( lower_pts[0], lower_pts[1], lower_pts[2] );
  Point upper( upper_pts[0], upper_pts[1], upper_pts[2] );

  if( !lower_stream || !upper_stream ) {
    throw ProblemSetupException( "BoundCondReader.cc: stringstream failed for lower or upper...\n", __FILE__, __LINE__ );
  }    
    
  // Bullet proofing -- origin must be on the same plane as the face.

  bool isOnFace = false;
    
  if( plusMinusFaces == -1 ) {    // x-, y-, z- faces
    isOnFace = is_on_face( p_dir, lower, g_grid_LoPts ) && is_on_face( p_dir, upper, g_grid_LoPts );
  }
    
  if( plusMinusFaces == 1 ){     // x+, y+, z+ faces
    isOnFace = is_on_face( p_dir, lower, g_grid_HiPts ) && is_on_face( p_dir, upper, g_grid_HiPts );
  }    
    
  if( !isOnFace ) {
    ostringstream err;
    err << "ERROR: The Rectangle BC geometry is not correctly specified.\n"
        << "The lower or upper point must be on the same plane " 
        << "as face (" << direction << ").\n\n";
    throw ProblemSetupException( err.str(), __FILE__, __LINE__ );
  }
    
  return scinew RectangleBCData( lower, upper, name, face_side );

} // end createRectangle()

static
BCGeomBase *
createAnnulus( map<string,string> values, const string & direction, const string & name )
{
  Patch::FaceType face_side;
  int             plusMinusFaces;
  int             p_dir;

  if( values[ "origin" ] == "" || values[ "inner_radius" ] == "" || values[ "outer_radius" ] == "" ) {
    ostringstream err;
    err << "ERROR: Annulus BC geometry not correctly specified.\n"
        << "You must specify origin [x,y,z], inner_radius, and outer_radius.\n";
    throw ProblemSetupException( err.str(), __FILE__, __LINE__ );
  }

  whichPatchFace( direction, face_side, plusMinusFaces, p_dir );

  std::stringstream origin_stream(       values[ "origin" ] );
  std::stringstream inner_radius_stream( values[ "inner_radius" ] );
  std::stringstream outer_radius_stream( values[ "outer_radius" ] );

  double inner_radius, outer_radius, origin_pts[3];

  inner_radius_stream >> inner_radius;
  outer_radius_stream >> outer_radius;
  origin_stream >> origin_pts[0] >> origin_pts[1] >> origin_pts[2];

  Point origin( origin_pts[0], origin_pts[1], origin_pts[2] );

  if( !inner_radius_stream || !outer_radius_stream || !origin_stream ) {
    throw ProblemSetupException( "BoundCondReader.cc: stringstream failed...\n", __FILE__, __LINE__ );
  }    
    
  // Bullet proofing -- origin must be on the same plane as the face.

  bool isOnFace = false;
    
  if( plusMinusFaces == -1 ) {    // x-, y-, z- faces
    isOnFace = is_on_face( p_dir, origin, g_grid_LoPts );
  }
    
  if( plusMinusFaces == 1 ){     // x+, y+, z+ faces
    isOnFace = is_on_face( p_dir, origin, g_grid_HiPts );      
  }    
    
  if( !isOnFace ) {
    ostringstream err;
    err << "ERROR: The Annulus BC geometry is not correctly specified.\n"
        << "The origin " << origin << " must be on the same plane " 
        << "as face (" << direction << "). Double check the origin and Level:box spec.\n\n";
    throw ProblemSetupException( err.str(), __FILE__, __LINE__ );
  }
    
  return scinew AnnulusBCData( origin, inner_radius, outer_radius, name, face_side );

} // end createAnnulus()

static
BCGeomBase *
createDifference( map<string,string>   values, 
                  const string       & direction,
                  const string       & name,
                  const m_s2bcgb     & fg_pieces )
{
  Patch::FaceType face_side;
  int             plusMinusFaces;
  int             p_dir;
  const string    part1 = values[ "part1" ];
  const string    part2 = values[ "part2" ];

  if( part1 == "" || part2 == "" ) {
    ostringstream err;
    err << "ERROR: Difference BC geometry requires a 'part1' and 'part2'...";
    throw ProblemSetupException( err.str(), __FILE__, __LINE__ );
  }

  m_s2bcgb::const_iterator it1 = fg_pieces.find( part1 );
  m_s2bcgb::const_iterator it2 = fg_pieces.find( part2 );

  if( it1 == fg_pieces.end() || it2 == fg_pieces.end() ) {
    throw ProblemSetupException( string( "While trying to create a 'difference' <FaceGeometry>," ) +
                                 "could not find one of these previous <FaceGeometry>s in .ups file: '" + 
                                 part1 + "', '" + part2 + "'.", __FILE__, __LINE__ );
  }

#if 0
FIXME this is Tony's face name stuff... 

    Patch::FaceType face_side;
    BCGeomBase* bcGeom = createBoundaryConditionFace(face_ps,grid_ps,face_side);

    std::string face_label = "none";
    face_ps->getAttribute("name",face_label);
    BCR_dbg << "Face Label = " << face_label << std::endl;
    
    BCR_dbg << endl << endl << "Face = " << face_side << " Geometry type = " 
      << typeid(*bcGeom).name() << " " << bcGeom << endl;

    HERE is where he passed the face_label into the create function...

    BoundCondFactory::create(child,bc,mat_id, face_label);

#endif

  BCGeomBase * p1 = it1->second;
  BCGeomBase * p2 = it2->second;

  if( p1 == NULL || p2 == NULL ) {
    throw ProblemSetupException( "Either part1 or part2 was not found.", __FILE__, __LINE__ );
  }

  whichPatchFace( direction, face_side, plusMinusFaces, p_dir );

  if( face_side != p1->getSide() || face_side != p2->getSide() ) {
    throw ProblemSetupException( "Part1 and/or part2 are not on the same side as " + name, __FILE__, __LINE__ );
  }
  return scinew DifferenceBCData( p1, p2, name, face_side );
}

//
// createBCGeomPiece()
//     face_geom_ps - problem spec to create a BC geom piece from.
//     fg_pieces    - all the the BC geom pieces that have already been created.
//
static
BCGeomBase *
createBCGeomPiece(       ProblemSpecP & face_geom_ps,
                   const m_s2bcgb     & fg_pieces )
{
  BCGeomBase * result = NULL;

  map<string,string> values;
  face_geom_ps->getAttributes( values );

#if 0  
  cout << "size of fg_pieces is " << fg_pieces.size() << "\n";
  cout << "Values:\n";
  for( map<string,string>::iterator iter = values.begin(); iter != values.end(); ++iter ) {
    cout << iter->first << " -- " << iter->second << "\n";
  }
  cout << "--------------------------------------\n";
#endif

  string name      = values[ "name" ];
  string type      = values[ "type" ];
  string direction = values[ "direction" ];

  cout << "createBCGeomPiece(): " << name << ", " << type << "\n";

  if( type == "circle" ) {
    result = createCircle( values, direction, name );
  }
  else if( type == "annulus" ) {
    result = createAnnulus( values, direction, name );
  }
  else if( type == "rectangle" ) {
    result = createRectangle( values, direction, name );
  }
  else if( type == "side" ) {
    Patch::FaceType face_side;
    int             plusMinusFaces;
    int             p_dir;
    whichPatchFace( direction, face_side, plusMinusFaces, p_dir );
    result = scinew SideBCData( name, face_side );
  }
  else if( type == "difference" ) {
    Patch::FaceType face_side;
    int             plusMinusFaces;
    int             p_dir;
    whichPatchFace( direction, face_side, plusMinusFaces, p_dir );
    result = createDifference( values, direction, name, fg_pieces );
  }
  else {
    throw ProblemSetupException( "Type '" + type + "' not valid." , __FILE__, __LINE__ );
  }

  cout << "Done createBCGeomPiece for " << name << "\n";
  return result;
}

//
// createFace()
//
//      Parses the 'face_ps' to create a new BCFace.  A BCFace knows its 'side' and has a list of geometries (BCGeomBases) 
//      that are associated with that face.  Multiple <Face>s can be specified for a given side (eg, "x+"), however
//      the first specified <Face> must have an associated <FaceGeometry> with type 'side' (see ups_spec.xml).  Then
//      all further <Face>s must have associated <FaceGeometry>s that do not have type 'side' (they can be 'circle', etc).
//
//      face_ps    : Problem Spec <Face> that will be parsed.
//      face_geoms : Set of all the Face Geoms that were already created.
//      sides      : [Input/Output] All the sides that have already been created.  If this is a new <Face> it is added to 'sides'.
//      faceGeom   : [Output] Reference to <FaceGeometry> that was just added to this BCFace. */
//      side       : [Output] Side the new Face is on.
//
static
BCFace *
createFace( ProblemSpecP                       face_ps, 
            m_s2bcgb                         & face_geoms,
            map< Patch::FaceType, BCFace* >  & sides,
            BCGeomBase                      *& faceGeom,
            Patch::FaceType                  & side )
{
  map<string,string> values;
  face_ps->getAttributes( values );

  string side_string = values[ "side" ];
  string geom_string = values[ "geometry" ];

  side     = Patch::stringToFaceType( side_string );
  faceGeom = face_geoms[ geom_string ];

  if( faceGeom == NULL ) {
    throw ProblemSetupException( "<FaceGeometry> piece named '" + geom_string +
                                 "' not found. It is needed for a specified <Face>.", __FILE__, __LINE__ );
  }

  BCFace * face = sides[ side ];

  if( face == NULL ) {

    // This is the first time that a <Face> for this 'side' (x-, x+, etc) has been parsed.
    // Make sure the corresponding <FaceGeometry> is of type="side".

    cout << "no main face exists yet for side " << side_string << "\n";
    cout << "going to create one out of " << geom_string << "\n";

    SideBCData * sideFaceGeom = dynamic_cast< SideBCData *>( faceGeom );

    if( sideFaceGeom == NULL ) {
      throw ProblemSetupException( "<FaceGeometry> piece named '" + geom_string +
                                   "' is not of type 'side'.\n" +
                                   "The first <FaceGeometry> applied to a <Face> must be of type='side'.",
                                   __FILE__, __LINE__ );
    }

    face = scinew BCFace( side );
    face->addGeometry( sideFaceGeom );
    sides[ side ] = face;

  }
  else { // Adding a new BC to the already existing <Face>.

    BCGeomBase * bcgeom = dynamic_cast< SideBCData *>( faceGeom );

    if( bcgeom != NULL ) { // faceGeom's type must not be 'side'!
      throw ProblemSetupException( "<FaceGeometry> piece named '" + geom_string +
                                   "' is of type 'side'.\n" +
                                   "The <Face>s may only have one <FaceGeometry> of type='side'.",
                                   __FILE__, __LINE__ );
    }
    face->addGeometry( faceGeom );
  }
  return face;
}

void
BoundCondReader::read( ProblemSpecP& bc_ps, const ProblemSpecP & grid_ps )
{  
  BCR_dbg << "here testing here\n";

  // FIXME: VERIFY THE FOLLOWING COMMENT:
  //
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

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // Determine the Level 0 grid high and low points, need by the bullet proofing.
  // [FIXME: NOTE, this originally was called every time a <Face> was processed...
  //         I think it is ok to just do once...???]
  //
  for( ProblemSpecP level_ps = grid_ps->findBlock("Level"); level_ps != 0; level_ps = level_ps->findNextBlock("Level") ) {
 
    //find upper/lower corner
    for( ProblemSpecP box_ps = level_ps->findBlock("Box"); box_ps != 0; box_ps = box_ps->findNextBlock("Box") ) {

      Point lower, upper;

      box_ps->require("lower", lower);
      box_ps->require("upper", upper);

      g_grid_LoPts.push_back(lower);
      g_grid_HiPts.push_back(upper);       

      g_grid_LoPt = Min( lower, g_grid_LoPt );
      g_grid_HiPt = Max( upper, g_grid_HiPt );
    }
  }

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // Read in the <FaceGeometry> pieces:
  //
  m_s2bcgb faceGeomPieces;

  for( ProblemSpecP face_geom_ps = bc_ps->findBlock("FaceGeometry"); face_geom_ps != 0; face_geom_ps = face_geom_ps->findNextBlock("FaceGeometry")) {

    BCGeomBase * bcGeom = createBCGeomPiece( face_geom_ps, faceGeomPieces );

    const string & name = bcGeom->getName();

    cout << "debug: created FaceGeometry piece '" << name << "' for side: " << bcGeom->getSide() << "\n";

    BCGeomBase * temp = faceGeomPieces[ name ];
    if( temp != NULL ) {
      throw ProblemSetupException( "Two <FaceGeometry> pieces have the same name: " + name, __FILE__, __LINE__ );
    }
    faceGeomPieces[ name ] = bcGeom;

  } // for face_geom_ps

  //////////////////////////////////////////////////////////////////////////////////////////////////
  // Read in the <Face>s and assign the above <FaceGeometry> to them:
  //

  // There should be six (and only six) <Face>s with type='side'.  However, more than one FaceGeometry can be added to a <Face>,
  // but only if the FaceGeometry is not of type='side'.
  //
  map< Patch::FaceType, BCFace* > sides;

  for( ProblemSpecP face_ps = bc_ps->findBlock("Face"); face_ps != 0; face_ps=face_ps->findNextBlock("Face") ) {

    BCGeomBase      * bcGeom;    // Set in following createFace() call...
    Patch::FaceType   face_side; // Set in following createFace() call...

    createFace( face_ps, faceGeomPieces, sides, bcGeom, face_side );

    cout << "Just added " << bcGeom->getName() << "\n";

    typedef multimap<int, BoundCondBase*> mmI2BCB;
    typedef pair    <int, BoundCondBase*> pI2BCB;

    typedef map     <int, BCGeomBase*>    mI2BCGB;

    string face_label = "none";
    face_ps->getAttribute( "name", face_label );  // FIXME... should this be inside the following BCType loop?

    for( ProblemSpecP child = face_ps->findBlock("BCType"); child != 0; child = child->findNextBlock("BCType") ) {

      BoundCondBase * bc = BoundCondFactory::create( child, face_label );
      cout << "Inserting into bcGeom (" << bcGeom->getName() << ") [mat_id = " << bc->getMatl()
           << "], bc name = " << bc->getVariable() 
           << ", bc type = " << bc->getType() 
           << ", bc matl = " << bc->getMatl() 
           << " ("  << bc  << ")\n";

      bcGeom->addBC( bc );
    }
    bcGeom->print(0); // DEBUG FIXME - remove

    // FIXME - old stuff follows that most likely can be removed:
    //
    // Add the Auxillary boundary condition type
    // set<int> materials;
    //
    //    const vector<BoundCondBase*> the_bcs = bcGeom->getBCData().getBCs();
    //
    //    for( vector<BoundCondBase*>::const_iterator iter = the_bcs.begin(); iter != the_bcs.end(); ++iter ) {
    //      materials.insert( (*iter)->getMatl() );
    //    }
    //
    //    for( set<int>::const_iterator iter = materials.begin(); iter != materials.end(); iter++ ) {
    //      int matl_id = *iter;
    //

    BoundCondBase* bc = scinew BoundCond<string>( "Auxiliary", "no type", "no value", face_label, "no_functor", -1 );

    bcGeom->addBC( bc );
    //    }

    // Search through the newly created boundary conditions and create
    // new BCGeomBase* clones if there are multi materials specified 
    // in the give <Face>.  This is usually a problem when Pressure is
    // specified for material id = 0, and other bcs such as velocity,
    // temperature, etc. for material_id != 0.
    //
    // FIXME... what do do below for the above comment?

  }  // end for( <Face> )

  // Take the individual boundary conditions and combine them into
  // a single difference (side and the union of any holes (circles or
  // rectangles)).  This only happens if there are more than 1 bc_data per
  // face.

  combineBCs( sides );

  cout << "After combineBCS() . . .\n\n";
  for( Patch::FaceType face = Patch::startFace; face <= Patch::endFace; face=Patch::nextFace(face) ) {
    cout << "Face " << face << ":\n";
    d_BCReaderData[face].print();
  } 

} // end read()

void
BoundCondReader::combineBCs( map<Patch::FaceType, BCFace*> & sides )
{
  for( Patch::FaceType faceIdx = Patch::startFace; faceIdx <= Patch::endFace; faceIdx++ ) {

    BCFace * face = sides[ faceIdx ];

    face->combineBCs();

    // Now take the 'face' information and stick it in the d_BCReaderData (in order
    // to match the previous way of doing things.

    const vector<BCGeomBase*> & geoms = face->getGeoms();

    cout << "Here: size of geoms: " << geoms.size() << "\n";

    set<int> materials;

    for( unsigned int pos = 0; pos < geoms.size(); pos++ ) {

      geoms[pos]->print(0); // DEBUG FIXME - remove

      /////////////////////////////////////////////////////////////////////////
      // 
      // If there is only one one material, then eliminate the '-1' category.
      // Note, the 'BCDataArray::getBoundCondData()' must also be updated so
      // that a request for material '-1' will return the real material.
      // So record with materials are actually used here, so that after this
      // loop we can take the appropriate action.

      const set<int> geom_matls = geoms[ pos ]->getMaterials();
      materials.insert( geom_matls.begin(), geom_matls.end() );
      
      d_BCReaderData[ faceIdx ].addBCGeomBase( geoms[ pos ] );
    }

    // FIXME: debug statement:
    cout << "Number of materials is " << materials.size() << "\n";

    if( materials.size() == 2 ) {
      // If size is 2 then there is -1 and one other #.  Remove the -1.  If size
      // is 1, then there is only -1, so we are good.  If size is greater then 2,
      // then we need -1 and the other materials.
      
      // FIXME: debug statement:
      cout << "Removing -1 material.\n";
      d_BCReaderData[ faceIdx ].removeGenericMaterial();
    }

    cout << "---- begin --------------------------------------------------\n";
    cout << "d_BCReaderData[ " << faceIdx << " ] is:\n";
    d_BCReaderData[faceIdx].print();
    cout << "---- end --------------------------------------------------\n";

  }
} // end combineBCS

