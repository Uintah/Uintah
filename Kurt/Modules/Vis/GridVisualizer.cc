/*
 *  GridVisualizer.cc:  Displays Patch boundaries
 *
 *  Written by:
 *   James Bigler
 *   Department of Computer Science
 *   University of Utah
 *   June 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/GeometryPort.h>
#include <SCICore/Geom/GeomLine.h>
#include <SCICore/Geom/Material.h>
#include <SCICore/Malloc/Allocator.h>
#include <Kurt/DataArchive/ArchivePort.h>
#include <Kurt/DataArchive/Archive.h>
#include <Uintah/Grid/GridP.h>
#include <Uintah/Grid/Grid.h>
#include <Uintah/Grid/Level.h>
#include <vector>

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace SCICore::GeomSpace;
using namespace SCICore::Geometry;
using namespace Uintah;
using namespace Kurt::Datatypes;
using namespace PSECore::Datatypes;
using namespace std;

class GridVisualizer : public Module {
private:
  void addBoxGeometry(GeomLines* edges, const Box& box);
  GridP getGrid();

  ArchiveIPort* in;
  GeometryOPort* ogeom;
  MaterialHandle level_color[6];
  
public:
  GridVisualizer(const clString& id);
  virtual ~GridVisualizer();
  virtual void execute();

};

extern "C" Module* make_GridVisualizer(const clString& id) {
  return new GridVisualizer(id);
}

GridVisualizer::GridVisualizer(const clString& id)
: Module("GridVisualizer", id, Filter)
{

  // Create the input port
  in=new ArchiveIPort(this, "Data Archive",
		      ArchiveIPort::Atomic);
  add_iport(in);

  // Create the output port
  ogeom=new GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
  add_oport(ogeom);
  
  ////////////////////////////////
  // Set up the colors used

  // define some colors
  MaterialHandle dk_red;
  MaterialHandle dk_green;
  MaterialHandle dk_blue;
  MaterialHandle lt_red;
  MaterialHandle lt_green;
  MaterialHandle lt_blue;
  MaterialHandle gray;

  dk_red = scinew Material(Color(0,0,0), Color(.3,0,0),
			   Color(.5,.5,.5), 20);
  dk_green = scinew Material(Color(0,0,0), Color(0,.3,0),
			     Color(.5,.5,.5), 20);
  dk_blue = scinew Material(Color(0,0,0), Color(0,0,.3),
			    Color(.5,.5,.5), 20);
  lt_red = scinew Material(Color(0,0,0), Color(.8,0,0),
			   Color(.5,.5,.5), 20);
  lt_green = scinew Material(Color(0,0,0), Color(0,.8,0),
			     Color(.5,.5,.5), 20);
  lt_blue = scinew Material(Color(0,0,0), Color(0,0,.8),
			    Color(.5,.5,.5), 20);
  gray = scinew Material(Color(0,0,0), Color(.4,.4,.4),
			 Color(.5,.5,.5), 20);
  
  // assign some colors to the different levels
  level_color[0] = lt_red;
  level_color[1] = lt_green;
  level_color[2] = lt_blue;
  level_color[3] = dk_red;
  level_color[4] = dk_green;
  level_color[5] = dk_blue;
}

GridVisualizer::~GridVisualizer()
{
}

GridP GridVisualizer::getGrid()
{
  ArchiveHandle handle;
  if(!in->get(handle)){
    std::cerr<<"Didn't get a handle\n";
    return 0;
  }

  //    setVars( handle );
  //-------------------------
  // in setVars
  DataArchive& dataArchive = *((*(handle.get_rep()))());
  vector< double > times;
  vector< int > indices;
  dataArchive.queryTimesteps( indices, times );
  GridP grid = dataArchive.queryGrid(times[0]);

  return grid;
}

void GridVisualizer::addBoxGeometry(GeomLines* edges, const Box& box)
{
  Point min = box.lower();
  Point max = box.upper();
  
  edges->add(Point(min.x(), min.y(), min.z()), Point(min.x(), min.y(), max.z()));
  edges->add(Point(min.x(), min.y(), min.z()), Point(min.x(), max.y(), min.z()));
  edges->add(Point(min.x(), min.y(), min.z()), Point(max.x(), min.y(), min.z()));
  edges->add(Point(max.x(), min.y(), min.z()), Point(max.x(), max.y(), min.z()));
  edges->add(Point(max.x(), min.y(), min.z()), Point(max.x(), min.y(), max.z()));
  edges->add(Point(min.x(), max.y(), min.z()), Point(max.x(), max.y(), min.z()));
  edges->add(Point(min.x(), max.y(), min.z()), Point(min.x(), max.y(), max.z()));
  edges->add(Point(min.x(), min.y(), min.z()), Point(min.x(), min.y(), max.z()));
  edges->add(Point(min.x(), min.y(), max.z()), Point(max.x(), min.y(), max.z()));
  edges->add(Point(min.x(), min.y(), max.z()), Point(min.x(), max.y(), max.z()));
  edges->add(Point(max.x(), max.y(), min.z()), Point(max.x(), max.y(), max.z()));
  edges->add(Point(max.x(), min.y(), max.z()), Point(max.x(), max.y(), max.z()));
  edges->add(Point(min.x(), max.y(), max.z()), Point(max.x(), max.y(), max.z()));
}

void GridVisualizer::execute()
{

  // clean out ogeom
  ogeom->delAll();

  // Get the handle on the grid and the number of levels
  GridP grid = getGrid();
  if(!grid)
    return;
  int numLevels = grid->numLevels();

  //-----------------------------------------
  // for each level in the grid
  for(int l = 0;l<numLevels;l++){
    LevelP level = grid->getLevel(l);

    // edges is all the edges made up all the patches in the level
    GeomLines* edges = new GeomLines();
    GeomObj* top = new GeomMaterial(edges, level_color[l]);
    
    Level::const_patchIterator iter;
    //---------------------------------------
    // for each patch in the level
    for(iter=level->patchesBegin(); iter != level->patchesEnd(); iter++){
      const Patch* patch=*iter;
      Box box = patch->getBox();
      addBoxGeometry(edges, box);
    }
    // add all the edges for the level
    ogeom->addObj(top, "Grid Visualizer");
  }
}

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.1  2000/06/05 21:10:30  bigler
// Added new module to visualize UINTAH grid
//
// Revision 1.0  2000/06/02 09:27:30  bigler
// Created initialversion
//


