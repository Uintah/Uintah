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
#include <Packages/Uintah/Dataflow/Modules/Visualization/VariablePlotter.h>

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Widgets/FrameWidget.h>
#include <Core/Geometry/IntVector.h>
#include <Core/Geom/GeomLine.h>
#include <Core/Geom/GeomPick.h>
#include <Core/Geom/GeomSphere.h>
#include <Core/Geom/GeomPoint.h>
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

namespace Uintah {
  
#define GRID_COLOR 1
#define NODE_COLOR 2
  // should match the values in the tcl code
#define NC_VAR 0
#define CC_VAR 1

class GridVisualizer : public VariablePlotter {
public:
  GridVisualizer(GuiContext* ctx);

  virtual ~GridVisualizer() {}

  virtual void execute();
  
  virtual void widget_moved(bool last);
  void tcl_command(GuiArgs& args, void* userdata);
  virtual void geom_pick(GeomPickHandle pick, void* userdata, GeomHandle picked);
  
protected:
  void addBoxGeometry(GeomLines* edges, const Box& box);
  void setupColors();
  MaterialHandle getColor(string color, int type);

  void initialize_ports();
  void setup_widget();
  void update_widget();
  Box get_widget_boundary();
  // Gets a physical point for the location of currentNode
  int get_current_node_position(Point &location);
  int update_selected_node();


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
  GuiDouble radius;
  GuiInt polygons;

  CrowdMonitor widget_lock;
  FrameWidget *widget2d;
  int init;
  int widget_id;
  bool widget_on;
  bool nodes_on;
  int need_2d;
  vector<int> old_id_list;
  vector<int> id_list;

  // These are variables used to show the location of the selected node
  int selected_sphere_geom_id;
  GuiInt show_selected_node;
  MaterialHandle selected_sphere_color;  
};

static string widget_name("GridVisualizer Widget");

  DECLARE_MAKER(GridVisualizer) 
} // end namespace Uintah

using namespace Uintah;

GridVisualizer::GridVisualizer(GuiContext* ctx):
  VariablePlotter("GridVisualizer", ctx),
  level1_grid_color(ctx->subVar("level1_grid_color")),
  level2_grid_color(ctx->subVar("level2_grid_color")),
  level3_grid_color(ctx->subVar("level3_grid_color")),
  level4_grid_color(ctx->subVar("level4_grid_color")),
  level5_grid_color(ctx->subVar("level5_grid_color")),
  level6_grid_color(ctx->subVar("level6_grid_color")),
  level1_node_color(ctx->subVar("level1_node_color")),
  level2_node_color(ctx->subVar("level2_node_color")),
  level3_node_color(ctx->subVar("level3_node_color")),
  level4_node_color(ctx->subVar("level4_node_color")),
  level5_node_color(ctx->subVar("level5_node_color")),
  level6_node_color(ctx->subVar("level6_node_color")),
  plane_on(ctx->subVar("plane_on")),
  node_select_on(ctx->subVar("node_select_on")),
  radius(ctx->subVar("radius")),
  polygons(ctx->subVar("polygons")),
  widget_lock("GridVusualizer widget lock"),
  need_2d(1),
  init(1),
  selected_sphere_geom_id(0),
  show_selected_node(ctx->subVar("show_selected_node"))
{
  float INIT(0.1);
  widget2d = scinew FrameWidget(this, &widget_lock, INIT, false);

  selected_sphere_color = new Material(Color(0,0,0), Color(1,0.6,0.3),
				       Color(.5,.5,.5), 20);
  
}

void GridVisualizer::initialize_ports() {
  // Create the input port
  in= (ArchiveIPort *) get_iport("Data Archive");
  // Create the output port
  ogeom= (GeometryOPort *) get_oport("Geometry");
}

void GridVisualizer::setup_widget() {
  // setup widget
  cerr << "\t\tStarting widget stuff\n";
  if (init == 1) {
    init = 0;
    GeomHandle w2d = widget2d->GetWidget();
    widget_id = ogeom->addObj(w2d, widget_name, &widget_lock);

    widget2d->Connect (ogeom);
  }

  setupColors();
}

void GridVisualizer::update_widget() {
  cerr << "\t\tStarting locator\n";
  widget_on = plane_on.get() != 0;
  widget2d->SetState(widget_on);
  // set thier mode to resize/translate only
  widget2d->SetCurrentMode(3);

  if (need_2d != 0){
    Point min, max;
    BBox gridBB;
    grid->getSpatialRange(gridBB);
    // Need to extend the BBox just a smidgen to correct for floating
    // point error.
    Vector offset(1e-12, 1e-12, 1e-12);
    min = gridBB.min()-offset;
    max = gridBB.max()+offset;
    
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
}

Box GridVisualizer::get_widget_boundary() {
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
    return Box(lower,upper);
  }
  else {
    BBox gridBB;
    grid->getSpatialRange(gridBB);
    // Need to extend the BBox just a smidgen to correct for floating
    // point error.
    Point offset(1e-12, 1e-12, 1e-12);
    return Box((gridBB.min()-offset).asPoint(),
	       (gridBB.max()+offset).asPoint());
  }
}

// Returns 0 for success, 1 otherwise
int GridVisualizer::get_current_node_position(Point &location) {
  // This call makes sure that currentNode is updated.
  pick();
  
  // We need to make sure that the grid pointer is valid.
  if (grid.get_rep() == NULL)
    return 1;
  // Now we need to get the level that corresponds to currentNode.
  if (grid->numLevels() <= currentNode.level) {
    error("GridVisualizer::get_current_node_position: "
	  "currentNode.level exceeds number of levels for the grid.");
    return 1;
  }
  LevelP level = grid->getLevel(currentNode.level);
  // From here we can query the location of the index from the level.
  switch (var_orientation.get()) {
  case NC_VAR:
    location = level->getNodePosition(currentNode.id);
    break;
  case CC_VAR:
    location = level->getCellPosition(currentNode.id);
    break;
  default:
    error("GridVisualizer::get_current_node_position: "
	  "unknown variable orientation");
  }
  return 0;
}

int GridVisualizer::update_selected_node() {
  // Since this function could be called from anywhere we need to make sure
  // that our ports are valid and that we have a grid.
  //  initialize_ports();
  
  // We need to determine if we should get rid of the last selected node.
  // By default we should get rid of it if is exists always.

  // These geom ids start at 1, so 0 is an unintialized value
  if (selected_sphere_geom_id != 0) {
    ogeom->delObj(selected_sphere_geom_id);
    selected_sphere_geom_id = 0;
  }

  // Now we can try using the grid after we have removed the geometry.
  //  if (initialize_grid() == 2)
  //    return 1;

  // Now add the node back if we are supposed to.
  reset_vars();
  if (show_selected_node.get() != 0) {
    int nu,nv;
    double rad;
    Point location;

    GeomSphere::getnunv(polygons.get(), nu, nv);
    rad = radius.get();

    if (!get_current_node_position(location)) {
      // add the sphere to the data
      selected_sphere_geom_id =
	ogeom->addObj(scinew GeomMaterial(scinew GeomSphere(location,
							    rad*1.5,nu,nv),
					  selected_sphere_color),
		      "Current Node");
    } else {
      // there was a problem, so don't add the sphere
      error("Can't add selected node.");
    }
  }
  ogeom->flush();
  return 0;
}
  
void GridVisualizer::execute()
{
  cerr << "GridVisualizer::execute:start\n";

  initialize_ports();
  
  // clean out ogeom
  for (unsigned int i = 0; i < id_list.size(); i++)
    ogeom->delObj(id_list[i]);
  id_list.clear();

  if (initialize_grid() == 2)
    return;
  update_tcl_window();
  
  setup_widget();
  update_widget();
  
  cerr << "GridVisualizer::execute:Finished the widget stuff\n";

  GeomGroup* pick_nodes = scinew GeomGroup;
  int nu,nv;
  double rad;
  bool node_on = node_select_on.get() != 0;
  Box widget_box;

  if (node_on) {
    GeomSphere::getnunv(polygons.get(), nu, nv);
    rad = radius.get();
    widget_box = get_widget_boundary();
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
    GeomPoints* nodes = scinew GeomPoints();

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
		     node_color[color_index].get_rep());
	}
	
	//------------------------------------
	// for each node in the patch that intersects the widget space
	if(node_on) {
	  for(NodeIterator iter = patch->getNodeIterator(widget_box); !iter.done(); iter++){
	    GeomSphere *s = scinew GeomSphere(patch->nodePosition(*iter),
					      rad,nu,nv);
	    s->setId(l);
	    s->setId(*iter);
	    spheres->add(s);
	  }
	}
	break;

      case CC_VAR:
	//------------------------------------
	// for each node in the patch
	for(CellIterator iter = patch->getCellIterator();!iter.done(); iter++){
	  nodes->add(patch->cellPosition(*iter),
		     node_color[color_index].get_rep());
	}
	
	//------------------------------------
	// for each node in the patch that intersects the widget space
	if(node_on) {
	  for(CellIterator iter = patch->getCellIterator(widget_box); !iter.done(); iter++){
	    GeomSphere *s = scinew GeomSphere(patch->cellPosition(*iter),
					      rad,nu,nv);
	    s->setId(l);
	    s->setId(*iter);
	    spheres->add(s);
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

  update_selected_node();
  
  cerr << "GridVisualizer::execute:end\n";
}

void GridVisualizer::widget_moved(bool last)
{
  if(last && !abort_flag) {
    abort_flag=1;
    want_to_execute();
  }
}

void GridVisualizer::tcl_command(GuiArgs& args, void* userdata)
{
  if(args.count() < 2) {
    args.error("GridVisualizer needs a minor command");
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
  else if(args[1] == "update_sn") {
    // we need to update the location of the selected node
    update_selected_node();
  }
  else {
    VariablePlotter::tcl_command(args, userdata);
  }
}

// if a pick event was received extract the id from the picked
void GridVisualizer::geom_pick(GeomPickHandle /*pick*/, void* /*userdata*/,
			       GeomHandle picked) {
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

    update_selected_node();
  }
  else
    cerr<<"Not getting the correct data\n";
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

















#if 0 // old code

//#include <string>

using namespace SCIRun;
using namespace std;


namespace Uintah {


struct ID {
  IntVector id;
  int level;
};
  
class GridVisualizer : public Module {
public:
  GridVisualizer(const string& id);
  virtual ~GridVisualizer();
  void pick();
  
private:
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
  GuiInt var_orientation; // whether node center or cell centered
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
    std::cerr<<"GridVisualizer::getGrid() Didn't get a handle\n";
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
    TCL::execute(id + " set_time " + vector_to_string(indices).c_str());
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
void GridVisualizer::pick() {
  reset_vars();
  currentNode.id.x(index_x.get());
  currentNode.id.y(index_y.get());
  currentNode.id.z(index_z.get());
  currentNode.level = index_l.get();
  cerr << "Extracting values for " << currentNode.id << ", level " << currentNode.level << endl;
}

#endif
