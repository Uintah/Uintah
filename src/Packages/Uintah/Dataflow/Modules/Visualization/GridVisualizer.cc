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

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Widgets/FrameWidget.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Geom/GeomLine.h>
#include <Core/Geom/GeomPick.h>
#include <Core/Geom/GeomSphere.h>
#include <Core/Geom/Pt.h>
#include <Core/Geom/Material.h>
#include <Core/Geometry/BBox.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Thread/CrowdMonitor.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Dataflow/Ports/ArchivePort.h>
#include <Packages/Uintah/Core/Datatypes/Archive.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/NodeIterator.h> // Includ after Patch.h
#include <Packages/Uintah/Core/Grid/CellIterator.h> // Includ after Patch.h
//#include <Packages/Uintah/Core/Grid/FaceIterator.h> // Includ after Patch.h
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include <vector>
#include <sstream>
#include <iostream>
//#include <string>

using namespace SCIRun;
using namespace std;


namespace Uintah {

#define GRID_COLOR 1
#define NODE_COLOR 2
  // should match the values in the tcl code
#define NC_VAR 0
#define CC_VAR 1

struct ID {
  IntVector id;
  int level;
};
  
class GridVisualizer : public Module {
public:
  GridVisualizer(const string& id);
  virtual ~GridVisualizer();
  virtual void execute();
  virtual void widget_moved(int last);
  void tcl_command(TCLArgs& args, void* userdata);
  virtual void geom_pick(GeomPick* pick, void* userdata, GeomObj* picked);
  virtual void geom_pick(GeomPick* pick, void* userdata);
  virtual void geom_pick(GeomPick* pick, ViewWindow* window,
			 int data, const BState& bs);
  void pick();
  
private:
  void addBoxGeometry(GeomLines* edges, const Box& box);
  bool getGrid();
  void setupColors();
  MaterialHandle getColor(string color, int type);
  void add_type(string &type_list,const TypeDescription *subtype);
  void setVars(GridP grid);
  void getnunv(int* nu, int* nv);
  void extract_data(string display_mode, string varname,
		    vector<string> mat_list, vector<string> type_list,
		    string index);
  bool is_cached(string name, string& data);
  void cache_value(string where, vector<double>& values, string &data);
  void cache_value(string where, vector<Vector>& values);
  void cache_value(string where, vector<Matrix3>& values);
  string vector_to_string(vector< int > data);
  string vector_to_string(vector< string > data);
  string vector_to_string(vector< double > data);
  string vector_to_string(vector< Vector > data, string type);
  string vector_to_string(vector< Matrix3 > data, string type);
  string currentNode_str();
  
  ArchiveIPort* in;
  GeometryOPort* ogeom;
  MaterialHandle level_color[6];
  MaterialHandle node_color[6];

  GuiString level1_grid_color;
  GuiString level2_grid_color;
  GuiString level3_grid_color;
  GuiString level4_grid_color;
  GuiString level5_grid_color;
  GuiString level6_grid_color;
  GuiString level1_node_color;
  GuiString level2_node_color;
  GuiString level3_node_color;
  GuiString level4_node_color;
  GuiString level5_node_color;
  GuiString level6_node_color;
  GuiInt plane_on; // the selection plane
  GuiInt node_select_on; // the nodes
  GuiInt var_orientation; // whether node center or cell centered
  GuiDouble radius;
  GuiInt polygons;
  GuiInt nl;
  GuiInt index_x;
  GuiInt index_y;
  GuiInt index_z;
  GuiInt index_l;
  GuiString curr_var;
  
  CrowdMonitor widget_lock;
  FrameWidget *widget2d;
  int init;
  int widget_id;
  bool widget_on;
  bool nodes_on;
  int need_2d;
  vector<int> old_id_list;
  vector<int> id_list;
  GeomSphere* selected_sphere;
  bool node_selected;
  
  ID currentNode;
  vector< string > names;
  vector< double > times;
  vector< const TypeDescription *> types;
  double time;
  DataArchive* archive;
  int old_generation;
  int old_timestep;
  GridP grid;
  map< string, string > material_data_list;
  // call material_data_list.clear() to erase all entries
};

static string widget_name("GridVisualizer Widget");
 
extern "C" Module* make_GridVisualizer(const string& id) {
  return scinew GridVisualizer(id);
}

GridVisualizer::GridVisualizer(const string& id)
: Module("GridVisualizer", id, Filter, "Visualization", "Uintah"),
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
  index_x("index_x",id,this),
  index_y("index_y",id,this),
  index_z("index_z",id,this),
  index_l("index_l",id,this),
  curr_var("curr_var",id,this),
  widget_lock("GridVusualizer widget lock"),
  init(1), need_2d(1), node_selected(false),
  old_generation(-1), old_timestep(0), grid(NULL)
{

  float INIT(0.1);
  widget2d = scinew FrameWidget(this, &widget_lock, INIT);
}

GridVisualizer::~GridVisualizer()
{
}

// assigns a grid based on the archive and the timestep to grid
// return true if there was a new grid, false otherwise
bool GridVisualizer::getGrid()
{
  ArchiveHandle handle;
  if(!in->get(handle)){
    std::cerr<<"Didn't get a handle\n";
    grid = NULL;
    return false;
  }

  // access the grid through the handle and dataArchive
  archive = (*(handle.get_rep()))();
  int new_generation = (*(handle.get_rep())).generation;
  bool archive_dirty =  new_generation != old_generation;
  int timestep = (*(handle.get_rep())).timestep();
  if (archive_dirty) {
    old_generation = new_generation;
    vector< int > indices;
    times.clear();
    archive->queryTimesteps( indices, times );
    // set old_timestep to something that will cause a new grid
    // to be queried.
    old_timestep = -1;
    // clean out the cached information if the grid has changed
    material_data_list.clear();
  }
  if (timestep != old_timestep) {
    time = times[timestep];
    grid = archive->queryGrid(time);
    old_timestep = timestep;
    return true;
  }
  return false;
}

#if 0
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
  TCL::execute(id + " set_time " + vector_to_string(indices).c_str());
  int timestep = (*(handle.get_rep())).timestep();
  time = times[timestep];
  GridP grid = archive->queryGrid(time);

  return grid;
}
#endif

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
MaterialHandle GridVisualizer::getColor(string color, int type) {
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

void GridVisualizer::add_type(string &type_list,const TypeDescription *subtype)
{
  switch ( subtype->getType() ) {
  case TypeDescription::double_type:
    type_list += " scaler";
    break;
  case TypeDescription::Vector:
    type_list += " vector";
    break;
  case TypeDescription::Matrix3:
    type_list += " matrix3";
    break;
  default:
    cerr<<"Error in GridVisualizer::setVars(): Vartype not implemented.  Aborting process.\n";
    abort();
  }
}  

void GridVisualizer::setVars(GridP grid) {
  string varNames("");
  string type_list("");
  const Patch* patch = *(grid->getLevel(0)->patchesBegin());

  cerr << "Calling clearMat_list\n";
  TCL::execute(id + " clearMat_list ");
  
  for(int i = 0; i< (int)names.size(); i++) {
    switch (types[i]->getType()) {
    case TypeDescription::NCVariable:
      if (var_orientation.get() == NC_VAR) {
	varNames += " ";
	varNames += names[i];
	cerr << "Calling appendMat_list\n";
	TCL::execute(id + " appendMat_list " + archive->queryMaterials(names[i], patch, time).expandedString().c_str());
	add_type(type_list,types[i]->getSubType());
      }
      break;
    case TypeDescription::CCVariable:
      if (var_orientation.get() == CC_VAR) {
	varNames += " ";
	varNames += names[i];
	cerr << "Calling appendMat_list\n";
	TCL::execute(id + " appendMat_list " + archive->queryMaterials(names[i], patch, time).expandedString().c_str());
	add_type(type_list,types[i]->getSubType());
      }
      break;
    default:
      cerr << "GridVisualizer::setVars: Warning!  Ignoring unknown type.\n";
      break;
    }

    
  }

  cerr << "varNames = " << varNames << endl;
  TCL::execute(id + " setVar_list " + varNames.c_str());
  TCL::execute(id + " setType_list " + type_list.c_str());  
}

void GridVisualizer::getnunv(int* nu, int* nv) {
#define MIN_POLYS 8
#define MAX_POLYS 400
#define MIN_NU 4
#define MAX_NU 20
#define MIN_NV 2
#define MAX_NV 20
  // calculate the spheres nu,nv based on the number of polygons
  float t = (polygons.get() - MIN_POLYS)/float(MAX_POLYS - MIN_POLYS);
  *nu = int(MIN_NU + t*(MAX_NU - MIN_NU)); 
  *nv = int(MIN_NV + t*(MAX_NV - MIN_NV));
}

void GridVisualizer::execute()
{

  cerr << "\t\tEntering execute.\n";
  // Create the input port
  in= (ArchiveIPort *) get_iport("Data Archive");
  // Create the output port
  ogeom= (GeometryOPort *) get_oport("Geometry");

  old_id_list = id_list;
  // clean out ogeom
  if (old_id_list.size() != 0)
    for (int i = 0; i < (int)old_id_list.size(); i++)
      ogeom->delObj(old_id_list[i]);
  id_list.clear();

  // Get the handle on the grid and the number of levels
  bool new_grid = getGrid();
  if(!grid)
    return;
  int numLevels = grid->numLevels();

  // setup the tickle stuff
  setupColors();
  if (new_grid) {
    nl.set(numLevels);
    names.clear();
    types.clear();
    archive->queryVariables(names, types);
  }
  // make sure the variables are properly displayed
  setVars(grid);
  string visible;
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
  
  if (node_on) {
    getnunv(&nu,&nv);
    
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
					   rad,nu,nv,l,*iter));
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
					   rad,nu,nv,l,*iter));
	  }
	}
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
#if 0
  // may implement this some day
  if (node_selected) {
    //Point p;
    switch (var_orientation.get()) {
    case NC_VAR:
      //p = patch->nodePosition(currentNode);
      break;
    case CC_VAR:
      //p = patch->cellPosition(currentNode);
      break;
    }
    //seleted_sphere->move(p);
  }
#endif
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
  else if(args[1] == "extract_data") {
    int i = 2;
    string displaymode(args[i++]);
    string varname(args[i++]);
    string index(args[i++]);
    int num_mat;
    string_to_int(args[i++], num_mat);
    cerr << "Extracting " << num_mat << " materals:";
    vector< string > mat_list;
    vector< string > type_list;
    for (int j = i; j < i+(num_mat*2); j++) {
      string mat(args[j]);
      mat_list.push_back(mat);
      j++;
      string type(args[j]);
      type_list.push_back(type);
    }
    cerr << endl;
    cerr << "Graphing " << varname << " with materials: " << vector_to_string(mat_list) << endl;
    extract_data(displaymode,varname,mat_list,type_list,index);
  }
  else {
    Module::tcl_command(args, userdata);
  }
}

bool GridVisualizer::is_cached(string name, string& data) {
  map< string, string >::iterator iter;
  iter = material_data_list.find(name);
  if (iter == material_data_list.end()) {
    return false;
  }
  else {
    data = iter->second;
    return true;
  }
}

void GridVisualizer::cache_value(string where, vector<double>& values,
				 string &data) {
  data = vector_to_string(values);
  material_data_list[where] = data;
}

void GridVisualizer::cache_value(string where, vector<Vector>& values) {
  string data = vector_to_string(values,"length");
  material_data_list[where+" length"] = data;
  data = vector_to_string(values,"length2");
  material_data_list[where+" length2"] = data;
  data = vector_to_string(values,"x");
  material_data_list[where+" x"] = data;
  data = vector_to_string(values,"y");
  material_data_list[where+" y"] = data;
  data = vector_to_string(values,"z");
  material_data_list[where+" z"] = data;
}

void GridVisualizer::cache_value(string where, vector<Matrix3>& values) {
  string data = vector_to_string(values,"Determinant");
  material_data_list[where+" Determinant"] = data;
  data = vector_to_string(values,"Trace");
  material_data_list[where+" Trace"] = data;
  data = vector_to_string(values,"Norm");
  material_data_list[where+" Norm"] = data;
}

string GridVisualizer::vector_to_string(vector< int > data) {
  ostringstream ostr;
  for(int i = 0; i < (int)data.size(); i++) {
      ostr << data[i]  << " ";
    }
  return ostr.str();
}

string GridVisualizer::vector_to_string(vector< string > data) {
  string result;
  for(int i = 0; i < (int)data.size(); i++) {
      result+= (data[i] + " ");
    }
  return result;
}

string GridVisualizer::vector_to_string(vector< double > data) {
  ostringstream ostr;
  for(int i = 0; i < (int)data.size(); i++) {
      ostr << data[i]  << " ";
    }
  return ostr.str();
}

string GridVisualizer::vector_to_string(vector< Vector > data, string type) {
  ostringstream ostr;
  if (type == "length") {
    for(int i = 0; i < (int)data.size(); i++) {
      ostr << data[i].length() << " ";
    }
  } else if (type == "length2") {
    for(int i = 0; i < (int)data.size(); i++) {
      ostr << data[i].length2() << " ";
    }
  } else if (type == "x") {
    for(int i = 0; i < (int)data.size(); i++) {
      ostr << data[i].x() << " ";
    }
  } else if (type == "y") {
    for(int i = 0; i < (int)data.size(); i++) {
      ostr << data[i].y() << " ";
    }
  } else if (type == "z") {
    for(int i = 0; i < (int)data.size(); i++) {
      ostr << data[i].z() << " ";
    }
  }

  return ostr.str();
}

string GridVisualizer::vector_to_string(vector< Matrix3 > data, string type) {
  ostringstream ostr;
  if (type == "Determinant") {
    for(int i = 0; i < (int)data.size(); i++) {
      ostr << data[i].Determinant() << " ";
    }
  } else if (type == "Trace") {
    for(int i = 0; i < (int)data.size(); i++) {
      ostr << data[i].Trace() << " ";
    }
  } else if (type == "Norm") {
    for(int i = 0; i < (int)data.size(); i++) {
      ostr << data[i].Norm() << " ";
    } 
 }

  return ostr.str();
}

string GridVisualizer::currentNode_str() {
  ostringstream ostr;
  ostr << "Level-" << currentNode.level << "-(";
  ostr << currentNode.id.x()  << ",";
  ostr << currentNode.id.y()  << ",";
  ostr << currentNode.id.z() << ")";
  return ostr.str();
}

void GridVisualizer::extract_data(string display_mode, string varname,
				  vector <string> mat_list,
				  vector <string> type_list, string index) {

  pick();
  
  /*
    template<class T>
    void query(std::vector<T>& values, const std::string& name, int matlIndex,
               IntVector loc, double startTime, double endTime);
  */

  // clear the current contents of the ticles's material data list
  TCL::execute(id + " reset_var_val");

  // determine type
  const TypeDescription *td;
  for(int i = 0; i < (int)names.size() ; i++)
    if (names[i] == varname)
      td = types[i];
  
  string name_list("");
  const TypeDescription* subtype = td->getSubType();
  switch ( subtype->getType() ) {
  case TypeDescription::double_type:
    cerr << "Graphing a variable of type double\n";
    // loop over all the materials in the mat_list
    for(int i = 0; i < (int)mat_list.size(); i++) {
      string data;
      if (!is_cached(currentNode_str()+" "+varname+" "+mat_list[i],data)) {
	// query the value and then cache it
	vector< double > values;
	int matl = atoi(mat_list[i].c_str());
	try {
	  archive->query(values, varname, matl, currentNode.id, times[0], times[times.size()-1]);
	} catch (const VariableNotFoundInGrid& exception) {
	  cerr << "Caught VariableNotFoundInGrid Exception: " << exception.message() << endl;
	  return;
	} 
	cerr << "Received data.  Size of data = " << values.size() << endl;
	cache_value(currentNode_str()+" "+varname+" "+mat_list[i],values,data);
      } else {
	cerr << "Cache hit\n";
      }
      TCL::execute(id+" set_var_val "+data.c_str());
      name_list = name_list + mat_list[i] + " " + type_list[i] + " ";
    }
    break;
  case TypeDescription::Vector:
    cerr << "Graphing a variable of type Vector\n";
    // loop over all the materials in the mat_list
    for(int i = 0; i < (int)mat_list.size(); i++) {
      string data;
      if (!is_cached(currentNode_str()+" "+varname+" "+mat_list[i]+" "
		     +type_list[i],data)) {
	// query the value and then cache it
	vector< Vector > values;
	int matl = atoi(mat_list[i].c_str());
	try {
	  archive->query(values, varname, matl, currentNode.id, times[0], times[times.size()-1]);
	} catch (const VariableNotFoundInGrid& exception) {
	  cerr << "Caught VariableNotFoundInGrid Exception: " << exception.message() << endl;
	  return;
	} 
	cerr << "Received data.  Size of data = " << values.size() << endl;
	// could integrate this in with the cache_value somehow
	data = vector_to_string(values,type_list[i]);
	cache_value(currentNode_str()+" "+varname+" "+mat_list[i],values);
      } else {
	cerr << "Cache hit\n";
      }
      TCL::execute(id+" set_var_val "+data.c_str());
      name_list = name_list + mat_list[i] + " " + type_list[i] + " ";
    }
    break;
  case TypeDescription::Matrix3:
    cerr << "Graphing a variable of type Matrix3\n";
    // loop over all the materials in the mat_list
    for(int i = 0; i < (int)mat_list.size(); i++) {
      string data;
      if (!is_cached(currentNode_str()+" "+varname+" "+mat_list[i]+" "
		     +type_list[i],data)) {
	// query the value and then cache it
	vector< Matrix3 > values;
	int matl = atoi(mat_list[i].c_str());
	try {
	  archive->query(values, varname, matl, currentNode.id, times[0], times[times.size()-1]);
	} catch (const VariableNotFoundInGrid& exception) {
	  cerr << "Caught VariableNotFoundInGrid Exception: " << exception.message() << endl;
	  return;
	} 
	cerr << "Received data.  Size of data = " << values.size() << endl;
	// could integrate this in with the cache_value somehow
	data = vector_to_string(values,type_list[i]);
	cache_value(currentNode_str()+" "+varname+" "+mat_list[i],values);
      }
      else {
	// use cached value that was put into data by is_cached
	cerr << "Cache hit\n";
      }
      TCL::execute(id+" set_var_val "+data.c_str());
      name_list = name_list + mat_list[i] + " " + type_list[i] + " ";
    }
    break;
  case TypeDescription::Point:
    cerr << "Error trying to graph a Point.  No valid representation for Points for 2d graph.\n";
    break;
  default:
    cerr<<"Unknown var type\n";
    }// else { Tensor,Other}
  TCL::execute(id+" "+display_mode.c_str()+"_data "+index.c_str()+" "
	       +varname.c_str()+" "+currentNode_str().c_str()+" "
	       +name_list.c_str());
  
}



// if a pick event was received extract the id from the picked
void GridVisualizer::geom_pick(GeomPick* /*pick*/, void* /*userdata*/,
			       GeomObj* picked) {
#if DEBUG
  cerr << "Caught pick event in GridVisualizer!\n";
  cerr << "this = " << this << ", pick = " << pick << endl;
  cerr << "User data = " << userdata << endl;
#endif
  IntVector id;
  int level;
  if ( picked->getId( id ) && picked->getId(level)) {
    cerr<<"Id = "<< id << " Level = " << level << endl;
    currentNode.id = id;
    index_l.set(level);
    index_x.set(currentNode.id.x());
    index_y.set(currentNode.id.y());
    index_z.set(currentNode.id.z());
#if 0  // may implement this some day
    // add the selected sphere to the geometry
    Point p;
    int nu,nv;
    double rad = radius.get() * 1.5;
    getnunv(&nu,&nv);
    switch (var_orientation.get()) {
    case NC_VAR:
      //p = patch->nodePosition(currentNode);
      break;
    case CC_VAR:
      //p = patch->cellPosition(currentNode);
      break;
    }
    if (!node_selected) {
      selected_sphere = scinew GeomSphere(p,rad,nu,nv,
					  currentNode.level,currentNode.id);
      GeomPick* pick = scinew GeomPick(scinew GeomMaterial(selected_sphere, getColor("green",GRID_COLOR)),this);
      ogeom->addObj(pick,"Selected Node");
      node_selected = true;
    }
    else {
      //seleted_sphere->move(p);
    }
#endif
  }
  else
    cerr<<"Not getting the correct data\n";
}

// this doesn't do anything.  They are only here to eliminate compiler warnings
void GridVisualizer::geom_pick(GeomPick* /*pick*/, void* /*userdata*/) {
}

// this doesn't do anything.  They are only here to eliminate compiler warnings
void GridVisualizer::geom_pick(GeomPick* /*pick*/, ViewWindow* /*window*/,
			       int /*data*/, const BState& /*bs*/) {
}

// if a pick event was received extract the id from the picked
void GridVisualizer::pick() {
  reset_vars();
  currentNode.id.x(index_x.get());
  currentNode.id.y(index_y.get());
  currentNode.id.z(index_z.get());
  currentNode.level = index_l.get();
  cerr << "Extracting values for " << currentNode.id << ", level " << currentNode.level << endl;
}

} // End namespace Uintah
