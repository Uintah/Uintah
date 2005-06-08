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
#include <Packages/Uintah/Core/Grid/Variables/NodeIterator.h> // Includ after Patch.h
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h> // Includ after Patch.h
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
  
  virtual void widget_moved(bool last, BaseWidget*);
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
  // Using the bounding box of the grid, creates a default radius
  // size.  Only sets it if it is different.
  void update_default_radius();
  // Gets a physical point for the location of currentNode
  int get_current_node_position(Point &location);
  int update_selected_node(bool flush_position=false);


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
  GuiInt use_default_radius;
  GuiDouble default_radius;
  GuiDouble radius;
  GuiInt polygons;

  CrowdMonitor widget_lock;
  FrameWidget *widget2d;
  int init;
  int widget_id;
  bool widget_on;
  bool nodes_on;
  Point iPoint_;
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
  use_default_radius(ctx->subVar("use_default_radius")),
  default_radius(ctx->subVar("default_radius")),
  radius(ctx->subVar("radius")),
  polygons(ctx->subVar("polygons")),
  widget_lock("GridVusualizer widget lock"),
  init(1),
  iPoint_(Point(0,0,0)),
  need_2d(1),
  selected_sphere_geom_id(0),
  show_selected_node(ctx->subVar("show_selected_node"))
{
  float INIT(0.1);
  widget2d = scinew FrameWidget(this, &widget_lock, INIT, false);

  selected_sphere_color = new Material(Color(0,0,0), Color(1,0.6,0.3),
                                       Color(.5,.5,.5), 20);
  
}

void
GridVisualizer::initialize_ports()
{
  // Create the input port
  in_ = (ArchiveIPort *) get_iport("Data Archive");
  // Create the output port
  ogeom= (GeometryOPort *) get_oport("Geometry");
}

void
GridVisualizer::setup_widget()
{
  // setup widget
  remark("Starting widget stuff");
  if (init == 1) {
    init = 0;
    GeomHandle w2d = widget2d->GetWidget();
    widget_id = ogeom->addObj(w2d, widget_name, &widget_lock);
    

    widget2d->Connect (ogeom);

    // set thier mode to resize/translate only
    widget2d->SetCurrentMode(3);
  }

  setupColors();
}

void
GridVisualizer::update_widget() 
{
  remark("Starting locator");
  widget_on = plane_on.get() != 0;
  widget2d->SetState(widget_on);

  Point min, max;
  BBox gridBB;
  grid_->getSpatialRange(gridBB);
  // Need to extend the BBox just a smidgen to correct for floating
  // point error.
  Vector offset(1e-12, 1e-12, 1e-12);
  min = gridBB.min()-offset;
  max = gridBB.max()+offset;
  
  if( iPoint_ != min ){
    iPoint_ = min;
    need_2d = 1;
  }
  
  if (need_2d != 0){
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

// default radius =  radius of a sphere that would fill 
// the volume of a cell.
void
GridVisualizer::update_default_radius()
{
  LevelP level = grid_->getLevel(0);
  Vector dCell = level->dCell();
  double new_default_radius = Min( dCell.x()/2,dCell.y()/2,dCell.z()/2);
  if (new_default_radius != default_radius.get()) {
    default_radius.set(new_default_radius);
  }
}

Box
GridVisualizer::get_widget_boundary() 
{
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
    grid_->getSpatialRange(gridBB);
    // Need to extend the BBox just a smidgen to correct for floating
    // point error.
    Point offset(1e-12, 1e-12, 1e-12);
    return Box((gridBB.min()-offset).asPoint(),
               (gridBB.max()+offset).asPoint());
  }
}

// Returns 0 for success, 1 otherwise
int
GridVisualizer::get_current_node_position(Point &location)
{
  // This call makes sure that currentNode is updated.
  pick();
  
  // We need to make sure that the grid pointer is valid.
  if (grid_.get_rep() == NULL)
    return 1;
  // Now we need to get the level that corresponds to currentNode.
  if (grid_->numLevels() <= currentNode_.level_) {
    error("GridVisualizer::get_current_node_position: "
          "currentNode.level exceeds number of levels for the grid.");
    return 1;
  }
  LevelP level = grid_->getLevel(currentNode_.level_);
  // From here we can query the location of the index from the level.
  switch (var_orientation_.get()) {
  case NC_VAR:
    location = level->getNodePosition(currentNode_.id_);
    break;
  case CC_VAR:
    location = level->getCellPosition(currentNode_.id_);
    break;
  default:
    error("GridVisualizer::get_current_node_position: "
          "unknown variable orientation");
  }
  return 0;
}

int
GridVisualizer::update_selected_node(bool flush_position) 
{
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
    if (use_default_radius.get() == 1)
      rad = default_radius.get();
    else
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

  if (flush_position)
    ogeom->flush();
  
  return 0;
}
  
void
GridVisualizer::execute()
{
  remark("GridVisualizer::execute:start");

  initialize_ports();
  
  // clean out ogeom
  for (unsigned int i = 0; i < id_list.size(); i++)
    ogeom->delObj(id_list[i]);
  id_list.clear();

  switch (initialize_grid()) {
  case 0:
    // Grid didn't change, so don't update gui
    break;
  case 1:
    // Grid changed so update the gui
    update_tcl_window();
    break;
  case 2:
    // Didn't get a grid handle
    return;
  }
  
  setup_widget();
  update_widget();
  update_default_radius();
  
  remark("GridVisualizer::execute:Finished the widget stuff");

  GeomGroup* pick_nodes = scinew GeomGroup;
  int nu,nv;
  double rad = 1;
  bool node_on = node_select_on.get() != 0;
  Box widget_box;

  if (node_on) {
    GeomSphere::getnunv(polygons.get(), nu, nv);
    if (use_default_radius.get() == 1)
      rad = default_radius.get();
    else
      rad = radius.get();
    widget_box = get_widget_boundary();
  }

  //-----------------------------------------
  // for each level in the grid
  for(int l = 0;l<numLevels_;l++){
    LevelP level = grid_->getLevel(l);

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

      switch (var_orientation_.get()) {
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
  
  remark("GridVisualizer::execute:end");
}

void
GridVisualizer::widget_moved(bool last, BaseWidget*)
{
  if(last && !abort_flag) {
    abort_flag=1;
    want_to_execute();
  }
}

void
GridVisualizer::tcl_command(GuiArgs& args, void* userdata)
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
    update_selected_node(true);
  }
  else {
    VariablePlotter::tcl_command(args, userdata);
  }
}

// if a pick event was received extract the id from the picked
void
GridVisualizer::geom_pick(GeomPickHandle /*pick*/, void* /*userdata*/,
                          GeomHandle picked) {
  ostringstream str;
#if DEBUG
  str << "Caught pick event in GridVisualizer!\n";
  str << "this = " << this << ", pick = " << pick << "\n";
  str << "User data = " << userdata;
  remark( str.str() );
#endif
  IntVector id;
  int level;
  if ( picked->getId( id ) && picked->getId(level)) {

    str.clear();
    str << "Id = " << id << " Level = " << level << endl;
    remark( str.str() );
    
    currentNode_.id_ = id;
    index_l_.set(level);
    index_x_.set(currentNode_.id_.x());
    index_y_.set(currentNode_.id_.y());
    index_z_.set(currentNode_.id_.z());

    update_selected_node(true);
  }
  else
    error("GridVisualizer::geom_pick:Not getting the correct data");
}

// adds the lines to edges that make up the box defined by box 
void
GridVisualizer::addBoxGeometry(GeomLines* edges, const Box& box)
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
MaterialHandle
GridVisualizer::getColor(string color, int type)
{
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
