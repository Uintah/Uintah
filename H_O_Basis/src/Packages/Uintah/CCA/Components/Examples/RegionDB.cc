
#include <Packages/Uintah/CCA/Components/Examples/RegionDB.h>
#include <Packages/Uintah/Core/Exceptions/ProblemSetupException.h>
#include <Packages/Uintah/Core/Grid/GeomPiece/GeometryPiece.h>
#include <Packages/Uintah/Core/Grid/GeomPiece/GeometryPieceFactory.h>
#include <Packages/Uintah/Core/Grid/GeomPiece/BoxGeometryPiece.h>
#include <Packages/Uintah/Core/Grid/GeomPiece/DifferenceGeometryPiece.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <iostream>

using namespace Uintah;
using namespace std;

RegionDB::RegionDB()
{
}

RegionDB::~RegionDB()
{
  for(MapType::iterator iter = db.begin(); iter != db.end(); iter++)
    delete iter->second;
}

void RegionDB::problemSetup(ProblemSpecP& ps, const GridP& grid)
{
  ProblemSpecP regions = ps->findBlock("Regions");
  if(!regions)
    throw ProblemSetupException("Regions block not found");
  vector<GeometryPiece*> pieces;
  GeometryPieceFactory::create(regions, pieces);
  for(vector<GeometryPiece*>::iterator iter = pieces.begin();
      iter != pieces.end(); iter++){
    addRegion(*iter);
    
  }

  // Create regions for x+, x-, y+, ...
  const LevelP& level = grid->getLevel(0);
  IntVector low, high;
  level->findCellIndexRange(low, high);
  IntVector ec(level->getExtraCells());
  Vector dx = level->dCell();
  Vector dx4 = dx*0.25;
  Point inner_lower = level->getNodePosition(low)+dx4;
  Point outer_lower = level->getNodePosition(low)-dx4;
  Point inner_upper = level->getNodePosition(high)-dx4;
  Point outer_upper = level->getNodePosition(high)+dx4;
  addRegion(new BoxGeometryPiece(outer_lower, outer_upper), "entire_domain");
  addRegion(new BoxGeometryPiece(inner_lower, inner_upper), "interior");
  addRegion(new DifferenceGeometryPiece(new BoxGeometryPiece(outer_lower, outer_upper),
					new BoxGeometryPiece(inner_lower, inner_upper)),
	    "allfaces");
  addRegion(new BoxGeometryPiece(outer_lower,
				 Point(inner_lower.x(), outer_upper.y(), outer_upper.z())),
	    "x-");
  addRegion(new BoxGeometryPiece(Point(inner_upper.x(), outer_lower.y(), outer_lower.z()),
				 Point(outer_upper)),
	    "x+");
  addRegion(new BoxGeometryPiece(outer_lower,
				 Point(outer_upper.x(), inner_lower.y(), outer_upper.z())),
	    "y-");
  addRegion(new BoxGeometryPiece(Point(outer_lower.x(), inner_upper.y(), outer_lower.z()),
				 Point(outer_upper)),
	    "y+");
  addRegion(new BoxGeometryPiece(outer_lower,
				 Point(inner_lower.x(), outer_upper.y(), inner_lower.z())),
	    "z-");
  addRegion(new BoxGeometryPiece(Point(outer_lower.x(), outer_lower.y(), inner_upper.z()),
				 Point(outer_upper)),
	    "z+");
  addRegion(new DifferenceGeometryPiece(new BoxGeometryPiece(outer_lower, outer_upper),
					new BoxGeometryPiece(Point(inner_lower.x(), outer_lower.y(), outer_lower.z()),
							     Point(inner_upper.x(), outer_upper.y(), outer_upper.z()))),
	    "xfaces");
  addRegion(new DifferenceGeometryPiece(new BoxGeometryPiece(outer_lower, outer_upper),
					new BoxGeometryPiece(Point(outer_lower.x(), inner_lower.y(), outer_lower.z()),
						Point(outer_upper.x(), inner_upper.y(), outer_upper.z()))),
	    "yfaces");
  addRegion(new DifferenceGeometryPiece(new BoxGeometryPiece(outer_lower, outer_upper),
					new BoxGeometryPiece(Point(outer_lower.x(), outer_lower.y(), inner_lower.z()),
						Point(outer_upper.x(), outer_upper.y(), inner_upper.z()))),
	    "zfaces");
}

void RegionDB::addRegion(GeometryPiece* piece, const string& name)
{
  piece->setName(name);
  addRegion(piece);
}

void RegionDB::addRegion(GeometryPiece* piece)
{
  if(piece->getName().length() == 0)
    throw ProblemSetupException("Geometry pieces in <Region> must be named");
  if(db.find(piece->getName()) != db.end())
    throw ProblemSetupException("Duplicate name of geometry piece: "+piece->getName());
  db[piece->getName()]=piece;
}

const GeometryPiece* RegionDB::getObject(const string& name) const
{
  MapType::const_iterator iter = db.find(name);
  if(iter == db.end())
    return 0;
  else
    return iter->second;
}
