/*
 *  GridVisualizer.cc:  Displays Patch boundaries
 *
 *  This module is used display the patch boundaries and node locations.
 *  Each level has a color chosen by the user from the UI in the PSE.
 *  It is hard coded for up to six
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
#include <PSECore/Widgets/FrameWidget.h>
#include <SCICore/Geom/GeomLine.h>
#include <SCICore/Geom/GeomPick.h>
#include <SCICore/Geom/GeomSphere.h>
#include <SCICore/Geom/Pt.h>
#include <SCICore/Geom/Material.h>
#include <SCICore/Geometry/BBox.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <SCICore/Thread/CrowdMonitor.h>
#include <Uintah/Components/MPM/Util/Matrix3.h>
#include <Uintah/Datatypes/ArchivePort.h>
#include <Uintah/Datatypes/Archive.h>
#include <Uintah/Grid/GridP.h>
#include <Uintah/Grid/Grid.h>
#include <Uintah/Grid/Level.h>
#include <Uintah/Grid/Patch.h>
#include <Uintah/Grid/NodeIterator.h> // Must be included after Patch.h
#include <Uintah/Grid/CellIterator.h> // Must be included after Patch.h
//#include <Uintah/Grid/FaceIterator.h> // Must be included after Patch.h
#include <Uintah/Grid/TypeDescription.h>
#include <vector>
#include <sstream>
#include <iostream>
//#include <string>

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace PSECore::Widgets;
using namespace SCICore::GeomSpace;
using namespace SCICore::Geometry;
using namespace SCICore::TclInterface;
using namespace Uintah;
using namespace Uintah::Datatypes;
using namespace std;

#define GRID_COLOR 1
#define NODE_COLOR 2
  // should match the values in the tcl code
#define NC_VAR 0
#define CC_VAR 1
#define FC_VAR 2
  
class GridVisualizer : public Module {
public:
  GridVisualizer(const clString& id);
  virtual ~GridVisualizer();
  virtual void execute();
  virtual void widget_moved(int last);
  void tcl_command(TCLArgs& args, void* userdata);
  virtual void geom_pick(GeomPick* pick, void* userdata, GeomObj* picked);

private:
  void addBoxGeometry(GeomLines* edges, const Box& box);
  GridP getGrid();
  void setupColors();
  MaterialHandle getColor(clString color, int type);
  void setVars(GridP grid);
  void graph(string varname, string material, string index);
  string vector_to_string(vector< double > data);
  string vector_to_string(vector< Vector > data);
  string vector_to_string(vector< Matrix3 > data);
  
  ArchiveIPort* in;
  GeometryOPort* ogeom;
  MaterialHandle level_color[6];
  MaterialHandle node_color[6];

  TCLstring level1_grid_color;
  TCLstring level2_grid_color;
  TCLstring level3_grid_color;
  TCLstring level4_grid_color;
  TCLstring level5_grid_color;
  TCLstring level6_grid_color;
  TCLstring level1_node_color;
  TCLstring level2_node_color;
  TCLstring level3_node_color;
  TCLstring level4_node_color;
  TCLstring level5_node_color;
  TCLstring level6_node_color;
  TCLint plane_on; // the selection plane
  TCLint node_select_on; // the nodes
  TCLint var_orientation; // whether node center or cell centered
  TCLdouble radius;
  TCLint polygons;
  TCLint nl;
  TCLstring curr_var;
  
  CrowdMonitor widget_lock;
  FrameWidget *widget2d;
  int init;
  int widget_id;
  bool widget_on;
  bool nodes_on;
  int need_2d;
  vector<int> old_id_list;
  vector<int> id_list;

  IntVector currentNode;
  vector< string > names;
  vector< double > times;
  vector< const TypeDescription *> types;
  double time;
  DataArchive* archive;
};

static clString widget_name("GridVisualizer Widget");
 
extern "C" Module* make_GridVisualizer(const clString& id) {
  return scinew GridVisualizer(id);
}

GridVisualizer::GridVisualizer(const clString& id)
: Module("GridVisualizer", id, Filter),
  level1_grid_color("level1_grid_color",id, this),
  level2_grid_color("level2_grid_color",id, this),
  level3_grid_color("level3_grid_color",id, this),
  level4_grid_color("level4_grid_color",id, this),
  level5_grid_color("level5_grid_color",id, this),
  level6_grid_color("level6_grid_color",id, this),
  level1_node_color("level1_node_color",id, this),
  level2_node_color("level2_node_color",id, this),
  level3_node_color("level3_node_color",id, this),
  level4_node_color("level4_node_color",id, this),
  level5_node_color("level5_node_color",id, this),
  level6_node_color("level6_node_color",id, this),
  plane_on("plane_on",id,this),
  node_select_on("node_select_on",id,this),
  var_orientation("var_orientation",id,this),
  radius("radius",id,this),
  polygons("polygons",id,this),
  nl("nl",id,this),
  curr_var("curr_var",id,this),
  widget_lock("GridVusualizer widget lock"),
  init(1), need_2d(1)
{

  // Create the input port
  in=scinew ArchiveIPort(this, "Data Archive",
		      ArchiveIPort::Atomic);
  add_iport(in);

  // Create the output port
  ogeom=scinew GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
  add_oport(ogeom);

  float INIT(0.1);
  widget2d = scinew FrameWidget(this, &widget_lock, INIT);
}

GridVisualizer::~GridVisualizer()
{
}

// returns a pointer to the grid
GridP GridVisualizer::getGrid()
{
  ArchiveHandle handle;
  if(!in->get(handle)){
    std::cerr<<"Didn't get a handle\n";
    return 0;
  }

  // access the grid through the handle and dataArchive
  archive = (*(handle.get_rep()))();
  vector< int > indices;
  times.clear();
  archive->queryTimesteps( indices, times );
  int timestep = (*(handle.get_rep())).timestep();
  time = times[timestep];
  GridP grid = archive->queryGrid(time);

  return grid;
}

// adds the lines to edges that make up the box defined by box 
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

// grabs the colors form the UI and assigns them to the local colors
void GridVisualizer::setupColors() {
  ////////////////////////////////
  // Set up the colors used

  // define some colors
  // assign some colors to the different levels
  level_color[0] = getColor(level1_grid_color.get(),GRID_COLOR);
  level_color[1] = getColor(level2_grid_color.get(),GRID_COLOR);
  level_color[2] = getColor(level3_grid_color.get(),GRID_COLOR);
  level_color[3] = getColor(level4_grid_color.get(),GRID_COLOR);
  level_color[4] = getColor(level5_grid_color.get(),GRID_COLOR);
  level_color[5] = getColor(level6_grid_color.get(),GRID_COLOR);

  // assign some colors to the different nodes in the levels
  node_color[0] = getColor(level1_node_color.get(),NODE_COLOR);
  node_color[1] = getColor(level2_node_color.get(),NODE_COLOR);
  node_color[2] = getColor(level3_node_color.get(),NODE_COLOR);
  node_color[3] = getColor(level4_node_color.get(),NODE_COLOR);
  node_color[4] = getColor(level5_node_color.get(),NODE_COLOR);
  node_color[5] = getColor(level6_node_color.get(),NODE_COLOR);
}

// based on the color expressed by color returns the color
MaterialHandle GridVisualizer::getColor(clString color, int type) {
  float i;
  if (type == GRID_COLOR)
    i = 1.0;
  else
    i = 0.7;
  if (color == "red")
    return scinew Material(Color(0,0,0), Color(i,0,0),
			   Color(.5,.5,.5), 20);
  else if (color == "green")
    return scinew Material(Color(0,0,0), Color(0,i,0),
			   Color(.5,.5,.5), 20);
  else if (color == "yellow")
    return scinew Material(Color(0,0,0), Color(i,i,0),
			   Color(.5,.5,.5), 20);
  else if (color == "magenta")
    return scinew Material(Color(0,0,0), Color(i,0,i),
			   Color(.5,.5,.5), 20);
  else if (color == "cyan")
    return scinew Material(Color(0,0,0), Color(0,i,i),
			   Color(.5,.5,.5), 20);
  else if (color == "blue")
    return scinew Material(Color(0,0,0), Color(0,0,i),
			   Color(.5,.5,.5), 20);
  else
    return scinew Material(Color(0,0,0), Color(i,i,i),
			   Color(.5,.5,.5), 20);
}

void GridVisualizer::setVars(GridP grid) {
  names.clear();
  types.clear();
  archive->queryVariables(names, types);

  string varNames("");
  string matls("");
  const Patch* patch = *(grid->getLevel(0)->patchesBegin());
  
  for(int i = 0; i< names.size(); i++) {
    switch (types[i]->getType()) {
    case TypeDescription::NCVariable:
      if (var_orientation.get() == NC_VAR) {
	varNames += " ";
	varNames += names[i];
	ostringstream mat;
	mat << " " << archive->queryNumMaterials(names[i], patch, time);
	matls += mat.str();
      }
      break;
    case TypeDescription::CCVariable:
      if (var_orientation.get() == CC_VAR) {
	varNames += " ";
	varNames += names[i];
	ostringstream mat;
	mat << " " << archive->queryNumMaterials(names[i], patch, time);
	matls += mat.str();
      }
      break;
    case TypeDescription::FCVariable:
      if (var_orientation.get() == FC_VAR) {
	varNames += " ";
	varNames += names[i];
	ostringstream mat;
	mat << " " << archive->queryNumMaterials(names[i], patch, time);
	matls += mat.str();
      }
      break;
    }
  }

  cerr << "varNames = " << varNames << endl;
  TCL::execute(id + " setVar_list " + varNames.c_str());
  TCL::execute(id + " setMat_list " + matls.c_str());  
}

void GridVisualizer::execute()
{

  cerr << "\t\tEntering execute.\n";

  old_id_list = id_list;
  // clean out ogeom
  if (old_id_list.size() != 0)
    for (int i = 0; i < old_id_list.size(); i++)
      ogeom->delObj(old_id_list[i]);
  id_list.clear();

  // Get the handle on the grid and the number of levels
  GridP grid = getGrid();
  if(!grid)
    return;
  int numLevels = grid->numLevels();

  // setup the tickle stuff
  setupColors();
  nl.set(numLevels);
  setVars(grid);
  clString visible;
  TCL::eval(id + " isVisible", visible);
  if ( visible == "1") {
    TCL::execute(id + " destroyFrames");
    TCL::execute(id + " build");

    TCL::execute("update idletasks");
    reset_vars();
  }
  
  // setup widget
  cerr << "\t\tStarting widget stuff\n";
  if (init == 1) {
    init = 0;
    GeomObj *w2d = widget2d->GetWidget();
    widget_id = ogeom->addObj(w2d, widget_name, &widget_lock);

    widget2d->Connect (ogeom);
  }
  widget_on = plane_on.get() != 0;
  widget2d->SetState(widget_on);
  widget2d->SetCurrentMode(1);

  cerr << "\t\tStarting locator\n";
  if (need_2d != 0){
    Point min, max;
    BBox gridBB;
    grid->getSpatialRange(gridBB);
    min = gridBB.min();
    max = gridBB.max();
    
    Point center = min + (max-min)/2.0;
    double max_scale;
    if (need_2d == 1) {
      // Find the boundary and put in optimal place
      // in xy plane with reasonable frame thickness
      Point right( max.x(), center.y(), center.z());
      Point down( center.x(), min.y(), center.z());
      widget2d->SetPosition( center, right, down);
      max_scale = Max( (max.x() - min.x()), (max.y() - min.y()) );
    } else if (need_2d == 2) {
      // Find the boundary and put in optimal place
      // in yz plane with reasonable frame thickness
      Point right( center.x(), center.y(), max.z());
      Point down( center.x(), min.y(), center.z());	    
      widget2d->SetPosition( center, right, down);
      max_scale = Max( (max.z() - min.z()), (max.y() - min.y()) );
    } else {
      // Find the boundary and put in optimal place
      // in xz plane with reasonable frame thickness
      Point right( max.x(), center.y(), center.z());
      Point down( center.x(), center.y(), min.z());	    
      widget2d->SetPosition( center, right, down);
      max_scale = Max( (max.x() - min.x()), (max.z() - min.z()) );
    }
    widget2d->SetScale( max_scale/30. );
    need_2d = 0;
  }
  
  cerr << "\t\tFinished the widget stuff\n";

  GeomGroup* pick_nodes = scinew GeomGroup;
  int nu,nv;
  double rad = radius.get();
  bool node_on = node_select_on.get() != 0;
  Box widget_box;
#define MIN_POLYS 8
#define MAX_POLYS 400
#define MIN_NU 4
#define MAX_NU 20
#define MIN_NV 2
#define MAX_NV 20
  
  if (node_on) {
    // calculate the spheres nu,nv based on the number of polygons
    float t = (polygons.get() - MIN_POLYS)/float(MAX_POLYS - MIN_POLYS);
    nu = int(MIN_NU + t*(MAX_NU - MIN_NU)); 
    nv = int(MIN_NV + t*(MAX_NV - MIN_NV));

    if (widget_on) {
      // get the position of the frame widget and determine
      // the boundaries
      Point center, R, D, I;
      widget2d->GetPosition( center, R, D);
      I = center;
      Vector v1 = R - center;
      Vector v2 = D - center;
      
      // calculate the edge points
      Point upper = center + v1 + v2;
      Point lower = center - v1 - v2;
      
      // need to determine extents of lower/upper
      Point temp1 = upper;
      upper = Max(temp1,lower);
      lower = Min(temp1,lower);
      widget_box = Box(lower,upper);
    }
    else {
      BBox gridBB;
      grid->getSpatialRange(gridBB);
      widget_box = Box(gridBB.min(),gridBB.max());
    }
  }
  
  //-----------------------------------------
  // for each level in the grid
  for(int l = 0;l<numLevels;l++){
    LevelP level = grid->getLevel(l);

    // there can be up to 6 colors only
    int color_index = l;
    if (color_index >= 6)
      color_index = 5;
    
    // edges is all the edges made up all the patches in the level
    GeomLines* edges = scinew GeomLines();

    // nodes consists of the nodes in all the patches in the level
    GeomPts* nodes = scinew GeomPts(1); // 1 is the number of points

    // spheres that will be the selectable nodes
    GeomGroup* spheres = scinew GeomGroup;
    
    Level::const_patchIterator iter;
    //---------------------------------------
    // for each patch in the level
    for(iter=level->patchesBegin(); iter != level->patchesEnd(); iter++){
      const Patch* patch=*iter;
      Box box = patch->getBox();
      addBoxGeometry(edges, box);

      switch (var_orientation.get()) {
      case NC_VAR:
	//------------------------------------
	// for each node in the patch
	for(NodeIterator iter = patch->getNodeIterator();!iter.done(); iter++){
	  nodes->add(patch->nodePosition(*iter),
		     node_color[color_index]->diffuse);
	}
	
	//------------------------------------
	// for each node in the patch that intersects the widget space
	if(node_on) {
	  for(NodeIterator iter = patch->getNodeIterator(widget_box); !iter.done(); iter++){
	    spheres->add(scinew GeomSphere(patch->nodePosition(*iter),
					   rad,nu,nv,*iter));
	  }
	}
	break;

      case CC_VAR:
	//------------------------------------
	// for each node in the patch
	for(CellIterator iter = patch->getCellIterator();!iter.done(); iter++){
	  nodes->add(patch->cellPosition(*iter),
		     node_color[color_index]->diffuse);
	}
	
	//------------------------------------
	// for each node in the patch that intersects the widget space
	if(node_on) {
	  for(CellIterator iter = patch->getCellIterator(widget_box); !iter.done(); iter++){
	    spheres->add(scinew GeomSphere(patch->cellPosition(*iter),
					   rad,nu,nv,*iter));
	  }
	}
	break;
	
      case FC_VAR:
	// not implemented
	break;
      }
    }

    // add all the nodes for the level
    ostringstream name_nodes;
    name_nodes << "Nodes - level " << l;
    id_list.push_back(ogeom->addObj(nodes, name_nodes.str().c_str()));

    // add all the edges for the level
    ostringstream name_edges;
    name_edges << "Patches - level " << l;
    id_list.push_back(ogeom->addObj(scinew GeomMaterial(edges, level_color[color_index]), name_edges.str().c_str()));

    // add the spheres for the nodes
    pick_nodes->add(scinew GeomMaterial(spheres, node_color[color_index]));
  }

  if (node_on) {
    GeomPick* pick = scinew GeomPick(pick_nodes,this);
    id_list.push_back(ogeom->addObj(pick,"Selectable Nodes"));
  }
  cerr << "\t\tFinished execute\n";
}

void GridVisualizer::widget_moved(int last)
{
  if(last && !abort_flag) {
    abort_flag=1;
    want_to_execute();
  }
}

void GridVisualizer::tcl_command(TCLArgs& args, void* userdata)
{
  if(args.count() < 2) {
    args.error("Streamline needs a minor command");
    return;
  }
  if(args[1] == "findxy") {
    need_2d=1;
    want_to_execute();
  }
  else if(args[1] == "findyz") {
    need_2d=2;
    want_to_execute();
  }
  else if(args[1] == "findxz") {
    need_2d=3;
    want_to_execute();
  }
  else if(args[1] == "graph") {
    string varname(args[2]());
    string matl(args[3]());
    string index(args[4]());
    cerr << "Graphing " << varname << " with material " << matl << endl;
    graph(varname,matl,index);
  }
  else {
    Module::tcl_command(args, userdata);
  }
}

string GridVisualizer::vector_to_string(vector< double > data) {
  ostringstream ostr;
  for(int i = 0; i < data.size(); i++) {
      ostr << i << " " << data[i]  << " ";
    }
  return ostr.str();
}

string GridVisualizer::vector_to_string(vector< Vector > data) {
  ostringstream ostr;
  for(int i = 0; i < data.size(); i++) {
      ostr << i << " " << data[i].length2() << " ";
    }
  return ostr.str();
}

string GridVisualizer::vector_to_string(vector< Matrix3 > data) {
  ostringstream ostr;
  for(int i = 0; i < data.size(); i++) {
      ostr << i << " " << data[i].Norm() << " ";
    }
  return ostr.str();
}

void GridVisualizer::graph(string varname, string material, string index) {

  /*
    template<class T>
    void query(std::vector<T>& values, const std::string& name, int matlIndex,
               IntVector loc, double startTime, double endTime);
  */
  // archive->query(value, varname, material, currentNode, times[0], times[times.size()-1]);

  // determine type
  const TypeDescription *td;
  for(int i = 0; i < names.size() ; i++)
    if (names[i] == varname)
      td = types[i];
  
  vector< double > valuesd;
  vector< Vector > valuesv;
  vector< Matrix3 > valuesm;
  vector< Point > valuesp;
  int matl = atoi(material.c_str());
  const TypeDescription* subtype = td->getSubType();
  switch ( subtype->getType() ) {
  case TypeDescription::double_type:
    cerr << "Graphing a double\n";
    archive->query(valuesd, varname, matl, currentNode, times[0], times[times.size()-1]);
    cerr << "Received data.  Size of data = " << valuesd.size() << endl;
    TCL::execute(id + " graph_data " + index.c_str() + material.c_str() + " " +
		 varname.c_str()+" "+vector_to_string(valuesd).c_str());
    break;
  case TypeDescription::Vector:
    cerr << "Graphing a Vector\n";
    archive->query(valuesv, varname, matl, currentNode, times[0], times[times.size()-1]);
    cerr << "Received data.  Size of data = " << valuesv.size() << endl;
    TCL::execute(id + " graph_data " + index.c_str() + material.c_str() + " " +
		 varname.c_str()+" "+vector_to_string(valuesv).c_str());
    break;
  case TypeDescription::Matrix3:
    cerr << "Graphing a Matrix3\n";
    archive->query(valuesm, varname, matl, currentNode, times[0], times[times.size()-1]);
    cerr << "Received data.  Size of data = " << valuesm.size() << endl;
    TCL::execute(id + " graph_data " + index.c_str() + material.c_str() + " " +
		 varname.c_str()+" "+vector_to_string(valuesm).c_str());
    break;
  case TypeDescription::Point:
    cerr << "Error trying to graph a Point.  No valid representation for Points for 2d graph.\n";
    //archive->query(valuesp, varname, matl, currentNode, times[0], times[times.size()-1]);
    //cerr << "Received data.  Size of data = " << valuesp.size() << endl;
    break;
  default:
    cerr<<"Unknown var type\n";
  }// else { Tensor,Other}
  
}



// if a pick event was received extract the id from the picked
void GridVisualizer::geom_pick(GeomPick* pick, void* userdata, GeomObj* picked) {
#if DEBUG
  cerr << "Caught pick event in GridVisualizer!\n";
  cerr << "this = " << this << ", pick = " << pick << endl;
  cerr << "User data = " << userdata << endl;
#endif
  IntVector id;
  if ( picked->getId( id ) ) {
    cerr<<"Id = "<< id <<endl;
    currentNode = id;
  }
  else
    cerr<<"Not getting the correct data\n";
}

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.4  2000/08/22 19:14:47  bigler
// Added graphing of NCvar and CCvar variables accross time for selected node.
// Can now also visualize cell centers.
//
// Revision 1.3  2000/08/14 17:29:04  bigler
// Added node selectability
//
// Revision 1.2  2000/08/09 03:18:09  jas
// Changed new to scinew and added deletes to some of the destructors.
//
// Revision 1.1  2000/06/20 17:57:19  kuzimmer
// Moved GridVisualizer to Uintah
//
// Revision 1.2  2000/06/06 18:31:56  bigler
// Added support for displaying nodes as well as support for more
// than 6 levels (it uses the last color for levels beyond 6).
//
// Revision 1.1  2000/06/05 21:10:30  bigler
// Added new module to visualize UINTAH grid
//
// Revision 1.0  2000/06/02 09:27:30  bigler
// Created initial version
//


