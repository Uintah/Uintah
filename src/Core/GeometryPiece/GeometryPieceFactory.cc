/*
 * The MIT License
 *
 * Copyright (c) 1997-2024 The University of Utah
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

#include <Core/GeometryPiece/GeometryPieceFactory.h>

#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/GeometryPiece/BoxGeometryPiece.h>
#include <Core/GeometryPiece/ConeGeometryPiece.h>
#include <Core/GeometryPiece/CylinderGeometryPiece.h>
#include <Core/GeometryPiece/DifferenceGeometryPiece.h>
#include <Core/GeometryPiece/EllipsoidGeometryPiece.h>
#include <Core/GeometryPiece/FileGeometryPiece.h>
#include <Core/GeometryPiece/IntersectionGeometryPiece.h>
#include <Core/GeometryPiece/NaaBoxGeometryPiece.h>
#include <Core/GeometryPiece/NullGeometryPiece.h>
#include <Core/GeometryPiece/ShellGeometryFactory.h>
#include <Core/GeometryPiece/ShellGeometryPiece.h>
#include <Core/GeometryPiece/SmoothCylGeomPiece.h>
#include <Core/GeometryPiece/SphereGeometryPiece.h>
#include <Core/GeometryPiece/SphereMembraneGeometryPiece.h>
#include <Core/GeometryPiece/TorusGeometryPiece.h>
#include <Core/GeometryPiece/TriGeometryPiece.h>
#include <Core/GeometryPiece/LineSegGeometryPiece.h>
#include <Core/GeometryPiece/UnionGeometryPiece.h>
#include <Core/GeometryPiece/ConvexPolyhedronGeometryPiece.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Parallel/Parallel.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Util/DOUT.hpp>
#include <Core/Util/RWS.h>

#include <iostream>
#include <string>

//#include <libxml/tree.h>

using namespace std;
using namespace Uintah;

namespace {
  Dout dout_gpf( "GeometryPieceFactory", "GeometryPieceFactory", "GeometryPieceFactory debug stream", false );
}

// Static class variable definition:
map<string,GeometryPieceP>             GeometryPieceFactory::m_namedPieces;
vector<GeometryPieceP>                 GeometryPieceFactory::m_unnamedPieces;
map<string, map<int, vector<Point> > > GeometryPieceFactory::m_insidePointsMap;
map< int, vector<Point> >              GeometryPieceFactory::m_allInsidePointsMap;

//------------------------------------------------------------------

bool
GeometryPieceFactory::foundInsidePoints(const std::string geomPieceName,
                                        const int patchID)
{
  typedef std::map<int,vector<Point> > PatchIDInsidePointsMapT;
//typedef std::map<std::string, PatchIDInsidePointsMapT > GeomNameInsidePtsMapT;

  if ( m_insidePointsMap.find(geomPieceName) != m_insidePointsMap.end() ) {
    // we found this geometry, lets see if we find this patch
    PatchIDInsidePointsMapT& thisPatchIDInsidePoints = m_insidePointsMap[geomPieceName];

    if ( thisPatchIDInsidePoints.find(patchID) != thisPatchIDInsidePoints.end() ) {
      // we found this patch ID
      return true;
    }
  }
  return false;
}
//------------------------------------------------------------------

const std::vector<Point>&
GeometryPieceFactory::getInsidePoints(const Uintah::Patch* const patch)
{
  typedef std::map<std::string,GeometryPieceP> NameGeomPiecesMapT;
//  typedef std::map<int,vector<Point> > PatchIDInsidePointsMapT;
  const int patchID = patch->getID();

  if ( m_allInsidePointsMap.find(patchID) != m_allInsidePointsMap.end() ){
    return m_allInsidePointsMap[patchID];
  }

  // loop over all geometry objects
  vector<Point> allInsidePoints;
  NameGeomPiecesMapT::iterator geomIter;

  for( geomIter = m_namedPieces.begin(); geomIter != m_namedPieces.end(); ++geomIter ){
    GeometryPieceP geomPiece = geomIter->second;

    const string geomName    = geomPiece->getName();
    const vector<Point>& thisGeomInsidePoints = getInsidePoints(geomName, patch);

    allInsidePoints.insert(allInsidePoints.end(), thisGeomInsidePoints.begin(), thisGeomInsidePoints.end());
  }

  m_allInsidePointsMap.insert(pair<int, vector<Point> >(patchID, allInsidePoints));
  return m_allInsidePointsMap[patchID];
}

//------------------------------------------------------------------

const std::vector<Point>&
GeometryPieceFactory::getInsidePoints(const std::string geomPieceName,
                                      const Uintah::Patch* const patch)
{
//typedef std::map<std::string,GeometryPieceP> NameGeomPiecesMapT;
  typedef std::map<int,vector<Point> > PatchIDInsidePointsMapT;
//typedef std::map<std::string, PatchIDInsidePointsMapT > GeomNameInsidePtsMapT;

  const int patchID = patch->getID();
  DOUTR( dout_gpf,  "computing points for patch " << patchID );

  if ( m_insidePointsMap.find(geomPieceName) != m_insidePointsMap.end()  ) {
    // we found this geometry, lets see if we find this patch
    PatchIDInsidePointsMapT& patchIDInsidePoints = m_insidePointsMap[geomPieceName];

    if ( patchIDInsidePoints.find(patchID) == patchIDInsidePoints.end() ) {
      // we did not find this patch. check if there are any points in this patch.
      GeometryPieceP geomPiece = m_namedPieces[ geomPieceName ];
      vector<Point> insidePoints;

      for(Uintah::CellIterator iter(patch->getCellIterator()); !iter.done(); iter++){
        IntVector iCell = *iter;

        Point p = patch->getCellPosition(iCell);
        const bool isInside = geomPiece->inside(p, false);

        if ( isInside ){
          insidePoints.push_back(p);
        }
      }
      patchIDInsidePoints.insert(pair<int, vector<Point> >(patch->getID(), insidePoints) );
    }
  }
  else {
    // if we did not find this geometry piece
    vector<Point> insidePoints;
    map<int, vector<Point> > patchIDInsidePoints;
    GeometryPieceP geomPiece = m_namedPieces[ geomPieceName ];

    for(Uintah::CellIterator iter(patch->getCellIterator()); !iter.done(); iter++){
      IntVector iCell = *iter;

      Point p = patch->getCellPosition(iCell);
      const bool isInside = geomPiece->inside(p,false);

      if ( isInside ){
        insidePoints.push_back(p);
      }
    }
    patchIDInsidePoints.insert(pair<int, vector<Point> >(patch->getID(), insidePoints) );
    m_insidePointsMap.insert(pair<string,PatchIDInsidePointsMapT>(geomPieceName, patchIDInsidePoints) );
  }
  // at this point, we can GUARANTEE that there is a vector of points associated with this geometry
  // and patch. This vector could be empty.
  return m_insidePointsMap[geomPieceName][patchID];
}

//------------------------------------------------------------------

void
GeometryPieceFactory::findInsidePoints(const Uintah::Patch* const patch)
{
  typedef std::map<std::string,GeometryPieceP> NameGeomPiecesMapT;
  typedef std::map<int,vector<Point> >         PatchIDInsidePointsMapT;

  const int patchID = patch->getID();

  // loop over all geometry objects
  NameGeomPiecesMapT::iterator geomIter;
  for( geomIter = m_namedPieces.begin(); geomIter != m_namedPieces.end(); ++geomIter ){

    vector<Point> insidePoints;
    map<int, vector<Point> > patchIDInsidePoints;

    GeometryPieceP geomPiece = geomIter->second;
    const string geomName    = geomPiece->getName();

    // check if we already found the inside points for this patch
    if (foundInsidePoints(geomName,patchID)){
      continue;
    }

    for(Uintah::CellIterator iter(patch->getCellIterator()); !iter.done(); iter++) {
      IntVector iCell = *iter;

      Point p = patch->getCellPosition(iCell);
      const bool isInside = geomPiece->inside(p,false);

      if ( isInside ){
        insidePoints.push_back(p);
      }
    }

    patchIDInsidePoints.insert(pair<int, vector<Point> >(patch->getID(), insidePoints)  );
    m_insidePointsMap.insert(pair<string,PatchIDInsidePointsMapT>(geomName, patchIDInsidePoints) );
  }
}

//------------------------------------------------------------------

void
GeometryPieceFactory::create( const ProblemSpecP           & ps,
                                    vector<GeometryPieceP> & objs )
{
  for( ProblemSpecP child = ps->findBlock(); child != nullptr; child = child->findNextBlock() ) {

    string gp_type = child->getNodeName();
    string gp_label;

    if( !child->getAttribute( "label", gp_label ) ) {
      // "label" and "name" are both used... so check for "label"
      // first, and if it isn't found, then check for "name".
      child->getAttribute( "name", gp_label );
    }

    DOUTR( dout_gpf, "---------------------------------------------------------------: geometryPiece label: " << gp_label );

    if( gp_label != "" ) {

      ProblemSpecP childBlock = child->findBlock();

      // See if there is any data for this node (that is not in a sub-block)
      string data = child->getNodeValue();
      remove_lt_white_space(data);

      // Lookup in table to see if this piece has already been named...
      GeometryPieceP referencedPiece = m_namedPieces[ gp_label ];

      // If it has a childBlock or data, then it is not just a reference.
      bool goHasInfo = (childBlock || data != "");

      if( referencedPiece.get_rep() != nullptr && goHasInfo ) {
       cout << "Error: GeometryPiece (" << gp_type << ")"
            << " labeled: '" << gp_label
            << "' has already been specified...  You can't change its values.\n"
            << "Please just reference the original by only "
            << "using the label (no values)\n";
       throw ProblemSetupException("Duplicate GeomPiece definition not allowed",
                                    __FILE__, __LINE__);
      }

      if( goHasInfo ) {
        DOUTR( dout_gpf, "Creating new GeometryPiece: " << gp_label
                         <<  " (of type: " << gp_type << ")");
      } else {

        if( referencedPiece.get_rep() != nullptr ) {
          DOUTR( dout_gpf, "Referencing GeometryPiece: " << gp_label
                            << " (of type: " << gp_type << ")");

          objs.push_back( referencedPiece );
        } else {
          cout << "Error... couldn't find the referenced GeomPiece: "
               << gp_label << " (" << gp_type << ")\n";
          throw ProblemSetupException("Referenced GeomPiece does not exist",
                                      __FILE__, __LINE__);
        }

        // Verify that the referenced piece is of the same type as
        // the originally created piece.
        if( referencedPiece->getType() != gp_type ) {
          cout << "Error... the referenced GeomPiece: " << gp_label
               << " (" << referencedPiece->getType() << "), "
               << "is not of the same type as this new object: '"
               << gp_type << "'!\n";
          throw ProblemSetupException("Referenced GeomPiece is not of the same type as original",__FILE__, __LINE__);
        }
        continue;
      }

    } else {
      DOUTR( dout_gpf, "Creating non-labeled GeometryPiece of type (" << gp_type << ")");
    }

    GeometryPiece * newGeomPiece = nullptr;

    if ( gp_type == BoxGeometryPiece::TYPE_NAME ) {
      newGeomPiece = scinew BoxGeometryPiece(child);
    }
    else if ( gp_type == NaaBoxGeometryPiece::TYPE_NAME ){
      newGeomPiece = scinew NaaBoxGeometryPiece(child);
    }
    else if ( gp_type == SphereGeometryPiece::TYPE_NAME ) {
      newGeomPiece = scinew SphereGeometryPiece(child);
    }
    else if ( gp_type == SphereMembraneGeometryPiece::TYPE_NAME ) {
      newGeomPiece = scinew SphereMembraneGeometryPiece(child);
    }
    else if ( gp_type == CylinderGeometryPiece::TYPE_NAME ) {
      newGeomPiece = scinew CylinderGeometryPiece(child);
    }
    else if ( gp_type == TorusGeometryPiece::TYPE_NAME ) {
      newGeomPiece = scinew TorusGeometryPiece(child);
    }
    else if ( gp_type ==  SmoothCylGeomPiece::TYPE_NAME ) {
      newGeomPiece = scinew SmoothCylGeomPiece(child);
    }
    else if ( gp_type == EllipsoidGeometryPiece::TYPE_NAME ) {
      newGeomPiece = scinew EllipsoidGeometryPiece(child);
    }
    else if ( gp_type == ConeGeometryPiece::TYPE_NAME ) {
      newGeomPiece = scinew ConeGeometryPiece(child);
    }
    else if ( gp_type == TriGeometryPiece::TYPE_NAME ) {
      newGeomPiece = scinew TriGeometryPiece(child);
    }
    else if ( gp_type == LineSegGeometryPiece::TYPE_NAME ) {
      newGeomPiece = scinew LineSegGeometryPiece(child);
    }
    else if ( gp_type == UnionGeometryPiece::TYPE_NAME ) {
      newGeomPiece = scinew UnionGeometryPiece(child);
    }
    else if ( gp_type == DifferenceGeometryPiece::TYPE_NAME ) {
      newGeomPiece = scinew DifferenceGeometryPiece(child);
    }
    else if ( gp_type == FileGeometryPiece::TYPE_NAME ) {
      newGeomPiece = scinew FileGeometryPiece(child);
    }
    else if ( gp_type == IntersectionGeometryPiece::TYPE_NAME ) {
      newGeomPiece = scinew IntersectionGeometryPiece(child);
    }
    else if ( gp_type == NullGeometryPiece::TYPE_NAME ) {
      newGeomPiece = scinew NullGeometryPiece(child);
    }
    else if ( gp_type == ConvexPolyhedronGeometryPiece::TYPE_NAME ) {
      newGeomPiece = scinew ConvexPolyhedronGeometryPiece(child);
    }
    else if (gp_type == "res"         || //gp_type == "velocity" ||
             gp_type == "temperature" || gp_type == "comment"  ||
             gp_type == "density"     || gp_type == "pressure" ||
             gp_type == "scalar"      || gp_type == "color"    ||
             gp_type == "concentration" ||
             gp_type == "conductivity"  ||
             gp_type == "neg_charge_density" ||         // these are not geometry pieces.
             gp_type == "pos_charge_density" ||
             gp_type == "permittivity" ||
             gp_type == "affineTransformation_A0" ||
             gp_type == "affineTransformation_A1" ||
             gp_type == "affineTransformation_A2" ||
             gp_type == "affineTransformation_b"  ||
             gp_type == "numLevelsParticleFilling"  ||
             gp_type == "volumeFraction" )  {
      // Ignoring.
      continue;    // restart loop to avoid accessing name of empty object

    } 
    else {
      // Perhaps it is a shell piece... let's find out:
      newGeomPiece = ShellGeometryFactory::create(child);

      if( newGeomPiece == nullptr ) {
        if( Parallel::getMPIRank() == 0 ) {
          cerr << "WARNING: Unknown Geometry Piece Type " << "(" << gp_type << ")\n" ;
        }
        continue;    // restart loop to avoid accessing name of empty object
      }
    }

    //__________________________________
    // Look for the "name" or "label" of the object.
    string name;
    if(child->getAttribute("name", name)){
      newGeomPiece->setName(name);
    }
    else if(child->getAttribute("label", name)){
      newGeomPiece->setName(name);
    }

    if( name != "" ) {
      m_namedPieces[ name ] = newGeomPiece;
      DOUTR( dout_gpf,  "  Adding to m_namedPieces");
    }
    else {
      m_unnamedPieces.push_back( newGeomPiece );
      DOUTR( dout_gpf,  "  Adding to m_unnamedPieces");
    }

    objs.push_back( newGeomPiece );

  } // end for( child )
  DOUTR( dout_gpf,  "  Done creating geometry objects");
}

//------------------------------------------------------------------

const std::map<std::string,GeometryPieceP>&
GeometryPieceFactory::getNamedGeometryPieces()
{
  return m_namedPieces;
}

//------------------------------------------------------------------

void
GeometryPieceFactory::resetFactory()
{
  DOUTR( dout_gpf, "GeometryPieceFactory::resetFactory()" );
  m_unnamedPieces.clear();
  m_namedPieces.clear();
  m_insidePointsMap.clear();
  m_allInsidePointsMap.clear();
}

//______________________________________________________________________
//  Recursively search through the problem spec for geometry pieces
//  that have already been created.  This is tricky and confusing code.  Consider two cases:
//  Case A is easy but with Case B the geom_object must be recursively searched.
//
//   CASE A
//         <geom_object>
//           <box label="mpm_box">            << childBlock and  gp label
//             <min>[1, 1, 1]</min>
//             <max>[1.5, 1.5, 1.5]</max>
//           </box>
//         </geom_object>
//   CASE B
//         <geom_object>
//           <difference>                    << start recursive search 
//             <box label="domain">          << childBlock and gp label
//               <min>[-1, -1, -1]</min>
//               <max>[4, 4, 4]</max>
//             </box>
//             <box label="mpm_box"/>         << no childBlock or gpLabel
//           </difference>
//         </geom_object>
//
//   returns a negative INT_MAX if any of the geom pieces were not found.
//   returns a positive integer if all of the geom pieces were found.
//
//______________________________________________________________________

int
GeometryPieceFactory::geometryPieceExists(const ProblemSpecP & ps,
                                          const bool isTopLevel   /* true */)
{

  int nFoundPieces = 0;
  for( ProblemSpecP child_ps = ps->findBlock(); child_ps != nullptr; child_ps = child_ps->findNextBlock() ) {
    

//    xmlNode* a_node = child_ps.get_rep()->getNode();          // debuggin output commented out -Todd
//    cout << "  nodeName " << a_node->name << endl;
//    child.get_rep()->printElementNames( a_node ); 
     
    bool hasChildBlock = false;
    if( child_ps->findBlock() ){
      hasChildBlock = true;
    }

    string gp_label = "";                                     // geometry piece label

    // search the node for either a label or name.
    if( !child_ps->getAttribute( "label", gp_label ) ) {
      child_ps->getAttribute( "name", gp_label );
    }

    //__________________________________
    //  possibly do a recursive search
    if( gp_label == "" )  {
      if( hasChildBlock ){                                // This could be either a <union> or<difference> or <intersection > node, dig deeper
        //cout << "   digging deeper "  << endl; 
        nFoundPieces += geometryPieceExists( child_ps, false );
      }
      continue;
    }

    // Is this geometry piece already a known?
    GeometryPieceP referencedPiece = m_namedPieces[ gp_label ];

    if( referencedPiece.get_rep() != nullptr  ) {
//    cout <<     " this child is known " << nFoundPieces << endl; 
      nFoundPieces += 1;
      continue;                       // keep searching
    }

    //__________________________________
    // Does the child have the spec of a geom_piece?
    // See if there is any data for this node (that is not in a sub-block)
    // If the spec exists then the geom_piece doesn't exist
    string nodeValue = child_ps->getNodeValue();
 
    remove_lt_white_space(nodeValue);

    bool has_gp_spec = ( hasChildBlock || nodeValue != "");
    if( has_gp_spec ){
      nFoundPieces = -INT_MAX;
    }

//    cout << "     geometryPieceExists: hasChildBlock: " << hasChildBlock  
//         << " nodeValue: (" << nodeValue << ") gp_label " << gp_label 
//         <<  " nodeName: " << child_ps->getNodeName() << " nFoundPieces: " << nFoundPieces << endl; 

    if( isTopLevel ){
      break;
    }
  }  // loop over geometry pieces

//  cout << "     exit isTopLevel: " << isTopLevel << " nFoundPieces: " << nFoundPieces << endl; 

  return nFoundPieces;
}

//------------------------------------------------------------------

void
GeometryPieceFactory::resetGeometryPiecesOutput()
{
  DOUTR( dout_gpf, "resetGeometryPiecesOutput() unnamedPieces.size()"
                   << m_unnamedPieces.size() << " namedPieces.end() " << m_namedPieces.size() );

  for( unsigned int pos = 0; pos < m_unnamedPieces.size(); pos++ ) {
    m_unnamedPieces[pos]->resetOutput();
    DOUTR( dout_gpf, "  - Reset: " << m_unnamedPieces[pos]->getName() );
  }

  map<std::string,GeometryPieceP>::const_iterator iter = m_namedPieces.begin();

  while( iter != m_namedPieces.end() ) {
    DOUTR( dout_gpf, "  - Reset: " << iter->second->getName() );
    iter->second->resetOutput();
    iter++;
  }
}

//------------------------------------------------------------------
