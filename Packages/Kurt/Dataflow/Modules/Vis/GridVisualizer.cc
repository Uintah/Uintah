/*
 *  GridVisualizer.cc:  Displays Patch boundaries
 *
 *  This module is used display the patch boundaries and node locations.
 *  Each level has a unique color.  It is hard coded for up to six
 *  different colors (or levels), and after the sixth level it reuses
 *  the last color.  The node locations have the same color as the patch
 *  boundaries, but are darker.  The colors are as follows:
 *    Level 1: Red
 *    Level 2: Green
 *    Level 3: Blue
 *    Level 4: Yellow
 *    Level 5: Cyan
 *    Level 6: Magenta
 *
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
#include <SCICore/Geom/Pt.h>
#include <SCICore/Geom/Material.h>
#include <SCICore/Malloc/Allocator.h>
#include <Kurt/DataArchive/ArchivePort.h>
#include <Kurt/DataArchive/Archive.h>
#include <Uintah/Grid/GridP.h>
#include <Uintah/Grid/Grid.h>
#include <Uintah/Grid/Level.h>
#include <Uintah/Grid/Patch.h>
#include <Uintah/Grid/NodeIterator.h> // Must be included after Patch.h
#include <vector>
#include <sstream>

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
public:
  GridVisualizer(const clString& id);
  virtual ~GridVisualizer();
  virtual void execute();

private:
  void addBoxGeometry(GeomLines* edges, const Box& box);
  GridP getGrid();

  ArchiveIPort* in;
  GeometryOPort* ogeom;
  MaterialHandle level_color[6];
  MaterialHandle node_color[6];
  
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
  MaterialHandle lt_red;
  MaterialHandle lt_green;
  MaterialHandle lt_blue;
  MaterialHandle lt_yellow;
  MaterialHandle lt_cyan;
  MaterialHandle lt_magenta;

  MaterialHandle dk_red;
  MaterialHandle dk_green;
  MaterialHandle dk_blue;
  MaterialHandle dk_yellow;
  MaterialHandle dk_cyan;
  MaterialHandle dk_magenta;

  lt_red = scinew Material(Color(0,0,0), Color(.8,0,0),
			   Color(.5,.5,.5), 20);
  lt_green = scinew Material(Color(0,0,0), Color(0,.8,0),
			     Color(.5,.5,.5), 20);
  lt_blue = scinew Material(Color(0,0,0), Color(0,0,.8),
			    Color(.5,.5,.5), 20);
  lt_yellow = scinew Material(Color(0,0,0), Color(0.8,0.8,0),
			    Color(.5,.5,.5), 20);
  lt_cyan = scinew Material(Color(0,0,0), Color(0,0.8,0.8),
			    Color(.5,.5,.5), 20);
  lt_magenta = scinew Material(Color(0,0,0), Color(0.8,0,0.8),
			    Color(.5,.5,.5), 20);
  
  dk_red = scinew Material(Color(0,0,0), Color(.3,0,0),
			   Color(.5,.5,.5), 20);
  dk_green = scinew Material(Color(0,0,0), Color(0,.3,0),
			     Color(.5,.5,.5), 20);
  dk_blue = scinew Material(Color(0,0,0), Color(0,0,.3),
			    Color(.5,.5,.5), 20);
  dk_yellow = scinew Material(Color(0,0,0), Color(0.3,0.3,0),
			    Color(.5,.5,.5), 20);
  dk_cyan = scinew Material(Color(0,0,0), Color(0,0.3,0.3),
			    Color(.5,.5,.5), 20);
  dk_magenta = scinew Material(Color(0,0,0), Color(0.3,0,0.3),
			    Color(.5,.5,.5), 20);
  
  // assign some colors to the different levels
  level_color[0] = lt_red;
  level_color[1] = lt_green;
  level_color[2] = lt_blue;
  level_color[3] = lt_yellow;
  level_color[4] = lt_cyan;
  level_color[5] = lt_magenta;

  // assign some colors to the different nodes in the levels
  node_color[0] = dk_red;
  node_color[1] = dk_green;
  node_color[2] = dk_blue;
  node_color[3] = dk_yellow;
  node_color[4] = dk_cyan;
  node_color[5] = dk_magenta;
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

  // access the grid through the handle and dataArchive
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

    // there can be up to 6 colors only
    int color_index = l;
    if (color_index >= 6)
      color_index = 5;
    
    // edges is all the edges made up all the patches in the level
    GeomLines* edges = new GeomLines();
    GeomObj* top_edges = new GeomMaterial(edges, level_color[color_index]);

    // nodes consists of the nodes in all the patches in the level
    GeomPts* nodes = new GeomPts(1); // 1 is the size
    GeomObj* top_nodes = new GeomMaterial(nodes, node_color[color_index]);
    
    
    Level::const_patchIterator iter;
    //---------------------------------------
    // for each patch in the level
    for(iter=level->patchesBegin(); iter != level->patchesEnd(); iter++){
      const Patch* patch=*iter;
      Box box = patch->getBox();
      addBoxGeometry(edges, box);

      //------------------------------------
      // for each node in the patch
      for(NodeIterator iter = patch->getNodeIterator(); !iter.done(); iter++){
	nodes->add(patch->nodePosition(*iter));
      }

    }

    // add all the edges for the level
    ostringstream name_edges;
    name_edges << "Patches - level " << l;
    ogeom->addObj(top_edges, name_edges.str().c_str());

    // add all the nodes for the level
    ostringstream name_nodes;
    name_nodes << "Nodes - level " << l;
    ogeom->addObj(top_nodes, name_nodes.str().c_str());
  }
}

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.2  2000/06/06 18:31:56  bigler
// Added support for displaying nodes as well as support for more than 6 levels (it uses the last color for levels beyond 6).
//
// Revision 1.1  2000/06/05 21:10:30  bigler
// Added new module to visualize UINTAH grid
//
// Revision 1.0  2000/06/02 09:27:30  bigler
// Created initial version
//


