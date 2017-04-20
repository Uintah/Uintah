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
#include <Core/GeometryPiece/UnionGeometryPiece.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Parallel/Parallel.h>
#include <Core/ProblemSpec/ProblemSpec.h>
#include <Core/ProblemSpec/ProblemSpecP.h>
#include <Core/Util/DebugStream.h>
#include <Core/Util/RWS.h>

#include <iostream>
#include <string>

using namespace std;
using namespace Uintah;

static DebugStream dbg( "GeometryPieceFactory", false );

// Static class variable definition:
map<string,GeometryPieceP>             GeometryPieceFactory::namedPieces_;
vector<GeometryPieceP>                 GeometryPieceFactory::unnamedPieces_;
map<string, map<int, vector<Point> > > GeometryPieceFactory::insidePointsMap_;
map< int, vector<Point> >              GeometryPieceFactory::allInsidePointsMap_;

//------------------------------------------------------------------

bool
GeometryPieceFactory::foundInsidePoints(const std::string geomPieceName,
                                        const int patchID)
{
  typedef std::map<int,vector<Point> > PatchIDInsidePointsMapT;
//typedef std::map<std::string, PatchIDInsidePointsMapT > GeomNameInsidePtsMapT;
  if (insidePointsMap_.find(geomPieceName) != insidePointsMap_.end()  ) {
    // we found this geometry, lets see if we find this patch
    PatchIDInsidePointsMapT& thisPatchIDInsidePoints = insidePointsMap_[geomPieceName];
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
  if (allInsidePointsMap_.find(patchID) != allInsidePointsMap_.end()) return allInsidePointsMap_[patchID];
  // loop over all geometry objects
  vector<Point> allInsidePoints;
  NameGeomPiecesMapT::iterator geomIter;
  for( geomIter = namedPieces_.begin(); geomIter != namedPieces_.end(); ++geomIter )
  {
    GeometryPieceP geomPiece = geomIter->second;
    const string geomName = geomPiece->getName();
    const vector<Point>& thisGeomInsidePoints = getInsidePoints(geomName, patch);
    allInsidePoints.insert(allInsidePoints.end(), thisGeomInsidePoints.begin(), thisGeomInsidePoints.end());
  }
  allInsidePointsMap_.insert(pair<int, vector<Point> >(patchID, allInsidePoints));
  return allInsidePointsMap_[patchID];
}

//------------------------------------------------------------------

const std::vector<Point>&
GeometryPieceFactory::getInsidePoints(const std::string geomPieceName, const Uintah::Patch* const patch)
{
//typedef std::map<std::string,GeometryPieceP> NameGeomPiecesMapT;
  typedef std::map<int,vector<Point> > PatchIDInsidePointsMapT;
//typedef std::map<std::string, PatchIDInsidePointsMapT > GeomNameInsidePtsMapT;

  const int patchID = patch->getID();
  dbg << "computing points for patch " << patchID << std::endl;
  if (insidePointsMap_.find(geomPieceName) != insidePointsMap_.end()  ) {
    // we found this geometry, lets see if we find this patch
    PatchIDInsidePointsMapT& patchIDInsidePoints = insidePointsMap_[geomPieceName];
    if ( patchIDInsidePoints.find(patchID) == patchIDInsidePoints.end() ) {
      // we did not find this patch. check if there are any points in this patch.
      GeometryPieceP geomPiece = namedPieces_[ geomPieceName ];
      vector<Point> insidePoints;
      for(Uintah::CellIterator iter(patch->getCellIterator()); !iter.done(); iter++)
      {
        IntVector iCell = *iter;
        Point p = patch->getCellPosition(iCell);
        const bool isInside = geomPiece->inside(p);
        if ( isInside )
        {
          insidePoints.push_back(p);
        }
      }
      patchIDInsidePoints.insert(pair<int, vector<Point> >(patch->getID(), insidePoints) );
    }
  } else {
    // if we did not find this geometry piece
    vector<Point> insidePoints;
    map<int, vector<Point> > patchIDInsidePoints;
    GeometryPieceP geomPiece = namedPieces_[ geomPieceName ];
    
    for(Uintah::CellIterator iter(patch->getCellIterator()); !iter.done(); iter++)
    {
      IntVector iCell = *iter;
      Point p = patch->getCellPosition(iCell);
      const bool isInside = geomPiece->inside(p);
      if ( isInside )
      {
        insidePoints.push_back(p);
      }
    }
    patchIDInsidePoints.insert(pair<int, vector<Point> >(patch->getID(), insidePoints) );
    insidePointsMap_.insert(pair<string,PatchIDInsidePointsMapT>(geomPieceName, patchIDInsidePoints) );
  }
  // at this point, we can GUARANTEE that there is a vector of points associated with this geometry
  // and patch. This vector could be empty.
  return insidePointsMap_[geomPieceName][patchID];
}

//------------------------------------------------------------------

void
GeometryPieceFactory::findInsidePoints(const Uintah::Patch* const patch)
{
  typedef std::map<std::string,GeometryPieceP> NameGeomPiecesMapT;
  typedef std::map<int,vector<Point> > PatchIDInsidePointsMapT;
  const int patchID = patch->getID();
  // loop over all geometry objects
  NameGeomPiecesMapT::iterator geomIter;
  for( geomIter = namedPieces_.begin(); geomIter != namedPieces_.end(); ++geomIter )
  {
    vector<Point> insidePoints;
    map<int, vector<Point> > patchIDInsidePoints;

    GeometryPieceP geomPiece = geomIter->second;
    const string geomName = geomPiece->getName();
   
    // check if we already found the inside points for this patch
    if (foundInsidePoints(geomName,patchID)) continue;
    
    for(Uintah::CellIterator iter(patch->getCellIterator()); !iter.done(); iter++)
    {
      IntVector iCell = *iter;
      Point p = patch->getCellPosition(iCell);
      const bool isInside = geomPiece->inside(p);
      if ( isInside )
      {
        insidePoints.push_back(p);
      }
    }
    patchIDInsidePoints.insert(pair<int, vector<Point> >(patch->getID(), insidePoints)  );
    insidePointsMap_.insert(pair<string,PatchIDInsidePointsMapT>(geomName, patchIDInsidePoints) );
  }
}

//------------------------------------------------------------------

void
GeometryPieceFactory::create( const ProblemSpecP           & ps,
                                    vector<GeometryPieceP> & objs )
{
  for( ProblemSpecP child = ps->findBlock(); child != nullptr; child = child->findNextBlock() ) {

    string go_type = child->getNodeName();
    string go_label;

    if( !child->getAttribute( "label", go_label ) ) {
      // "label" and "name" are both used... so check for "label"
      // first, and if it isn't found, then check for "name".
      child->getAttribute( "name", go_label );
    }

    dbg << "---------------------------------------------------------------: go_label: " << go_label << "\n";
    
    if( go_label != "" ) {

      ProblemSpecP   childBlock = child->findBlock();

      // See if there is any data for this node (that is not in a sub-block)
      string data = child->getNodeValue();
      remove_lt_white_space(data);

      // Lookup in table to see if this piece has already be named...
      GeometryPieceP referencedPiece = namedPieces_[ go_label ];

      // If it has a childBlock or data, then it is not just a reference.
      bool goHasInfo = childBlock || data != "";

      if( referencedPiece.get_rep() != nullptr && goHasInfo ) {
       cout << "Error: GeometryPiece (" << go_type << ")"
            << " labeled: '" << go_label 
            << "' has already been specified...  You can't change its values.\n"
            << "Please just reference the original by only "
            << "using the label (no values)\n";
       throw ProblemSetupException("Duplicate GeomPiece definition not allowed",
                                    __FILE__, __LINE__);
      }

      if( goHasInfo ) {
        dbg << "Creating new GeometryPiece: " << go_label 
            <<  " (of type: " << go_type << ")\n";
      } else {

        if( referencedPiece.get_rep() != nullptr ) {
          dbg << "Referencing GeometryPiece: " << go_label 
              << " (of type: " << go_type << ")\n";
          objs.push_back( referencedPiece );
        } else {
          cout << "Error... couldn't find the referenced GeomPiece: " 
               << go_label << " (" << go_type << ")\n";
          throw ProblemSetupException("Referenced GeomPiece does not exist",
                                      __FILE__, __LINE__);
        }

        // Verify that the referenced piece is of the same type as
        // the originally created piece.
        if( referencedPiece->getType() != go_type ) {
          cout << "Error... the referenced GeomPiece: " << go_label 
               << " (" << referencedPiece->getType() << "), "
               << "is not of the same type as this new object: '" 
               << go_type << "'!\n";
          throw ProblemSetupException("Referenced GeomPiece is not of the same type as original",__FILE__, __LINE__);
        }
        continue;
      }

    } else {
      dbg << "Creating non-labeled GeometryPiece of type '" << go_type << "'\n";
    }

    GeometryPiece * newGeomPiece = nullptr;

    if ( go_type == BoxGeometryPiece::TYPE_NAME ) {
      newGeomPiece = scinew BoxGeometryPiece(child);
    }
    else if ( go_type == NaaBoxGeometryPiece::TYPE_NAME ){
      newGeomPiece = scinew NaaBoxGeometryPiece(child);
    }
    else if ( go_type == SphereGeometryPiece::TYPE_NAME ) {
      newGeomPiece = scinew SphereGeometryPiece(child);
    }
    else if ( go_type == SphereMembraneGeometryPiece::TYPE_NAME ) {
      newGeomPiece = scinew SphereMembraneGeometryPiece(child);
    }
    else if ( go_type == CylinderGeometryPiece::TYPE_NAME ) {
      newGeomPiece = scinew CylinderGeometryPiece(child);
    }
    else if ( go_type == TorusGeometryPiece::TYPE_NAME ) {
      newGeomPiece = scinew TorusGeometryPiece(child);
    }
    else if ( go_type ==  SmoothCylGeomPiece::TYPE_NAME ) {
      newGeomPiece = scinew SmoothCylGeomPiece(child);
    }
    else if ( go_type == EllipsoidGeometryPiece::TYPE_NAME ) {
      newGeomPiece = scinew EllipsoidGeometryPiece(child);
    }
    else if ( go_type == ConeGeometryPiece::TYPE_NAME ) {
      newGeomPiece = scinew ConeGeometryPiece(child);
    }
    else if ( go_type == TriGeometryPiece::TYPE_NAME ) {
      newGeomPiece = scinew TriGeometryPiece(child);
    }
    else if ( go_type == UnionGeometryPiece::TYPE_NAME ) {
      newGeomPiece = scinew UnionGeometryPiece(child);
    }
    else if ( go_type == DifferenceGeometryPiece::TYPE_NAME ) {
      newGeomPiece = scinew DifferenceGeometryPiece(child);
    }
    else if ( go_type == FileGeometryPiece::TYPE_NAME ) {
      newGeomPiece = scinew FileGeometryPiece(child);
    }
    else if ( go_type == IntersectionGeometryPiece::TYPE_NAME ) {
      newGeomPiece = scinew IntersectionGeometryPiece(child);
    }
    else if ( go_type == NullGeometryPiece::TYPE_NAME ) {
      newGeomPiece = scinew NullGeometryPiece(child);
    }
    else if (go_type == "res"         || go_type == "velocity" || 
             go_type == "temperature" || go_type == "comment"  ||
             go_type == "density"     || go_type == "pressure" ||
             go_type == "scalar"      || go_type == "color"    ||
             go_type == "concentration" ||
             go_type == "conductivity"  ||
             go_type == "neg_charge_density" ||
             go_type == "pos_charge_density" ||
             go_type == "permittivity" ||
             go_type == "affineTransformation_A0" || 
             go_type == "affineTransformation_A1" ||
             go_type == "affineTransformation_A2" ||
             go_type == "affineTransformation_b"  ||
             go_type == "volumeFraction" )  {
      // Ignoring. 
      continue;    // restart loop to avoid accessing name of empty object
      
    } else {
      // Perhaps it is a shell piece... let's find out:
      newGeomPiece = ShellGeometryFactory::create(child);

      if( newGeomPiece == nullptr ) {
        if( Parallel::getMPIRank() == 0 ) {
          cerr << "WARNING: Unknown Geometry Piece Type " << "(" << go_type << ")\n" ;
        }
        continue;    // restart loop to avoid accessing name of empty object
      }
    }
    // Look for the "name" of the object.  (Can also be referenced as "label").
    string name;
    if(child->getAttribute("name", name)){
      newGeomPiece->setName(name);
    } else if(child->getAttribute("label", name)){
      newGeomPiece->setName(name);
    }
    if( name != "" ) {
      namedPieces_[ name ] = newGeomPiece;
    }
    else {
      unnamedPieces_.push_back( newGeomPiece );
    }

    objs.push_back( newGeomPiece );

  } // end for( child )
  dbg << "Done creating geometry objects\n";
}

//------------------------------------------------------------------

const std::map<std::string,GeometryPieceP>&
GeometryPieceFactory::getNamedGeometryPieces()
{
  return namedPieces_;
}

//------------------------------------------------------------------

void
GeometryPieceFactory::resetFactory()
{
  unnamedPieces_.clear();
  namedPieces_.clear();
  insidePointsMap_.clear();
  allInsidePointsMap_.clear();
}

//------------------------------------------------------------------

void
GeometryPieceFactory::resetGeometryPiecesOutput()
{
  dbg << "resetGeometryPiecesOutput()\n";

  for( unsigned int pos = 0; pos < unnamedPieces_.size(); pos++ ) {
    unnamedPieces_[pos]->resetOutput();
    dbg << "  - Reset: " << unnamedPieces_[pos]->getName() << "\n";
  }

  map<std::string,GeometryPieceP>::const_iterator iter = namedPieces_.begin();

  while( iter != namedPieces_.end() ) {
    dbg << "  - Reset: " << iter->second->getName() << "\n";
    iter->second->resetOutput();
    iter++;
  }
}

//------------------------------------------------------------------
