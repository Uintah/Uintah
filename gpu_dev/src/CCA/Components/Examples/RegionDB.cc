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



#include <CCA/Components/Examples/RegionDB.h>
#include <Core/Exceptions/ProblemSetupException.h>
#include <Core/GeometryPiece/GeometryPiece.h>
#include <Core/GeometryPiece/GeometryPieceFactory.h>
#include <Core/GeometryPiece/BoxGeometryPiece.h>
#include <Core/GeometryPiece/DifferenceGeometryPiece.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <iostream>

using namespace Uintah;
using namespace std;

RegionDB::RegionDB()
{
}

void
RegionDB::problemSetup(ProblemSpecP& ps, const GridP& grid)
{
  ProblemSpecP regions = ps->findBlock("Regions");
  if(!regions)
    throw ProblemSetupException("Regions block not found", __FILE__, __LINE__);
  vector<GeometryPieceP> pieces;
  GeometryPieceFactory::create(regions, pieces);
  for(vector<GeometryPieceP>::iterator iter = pieces.begin();
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

void
RegionDB::addRegion(GeometryPieceP piece, const string& name)
{
  piece->setName(name);
  addRegion(piece);
}

void
RegionDB::addRegion(GeometryPieceP piece)
{
  if(piece->getName().length() == 0)
    throw ProblemSetupException("Geometry pieces in <Region> must be named", __FILE__, __LINE__);

  if(db.find(piece->getName()) != db.end())
    throw ProblemSetupException("Duplicate name of geometry piece: "+piece->getName(), __FILE__, __LINE__);

  db[piece->getName()] = piece;
}

GeometryPieceP
RegionDB::getObject(const string& name) const
{
  MapType::const_iterator iter = db.find(name);
  if(iter == db.end())
    return 0;
  else
    return iter->second;
}
