/*
 *  PatchVisualizer.cc:  Displays Patch boundaries
 *
 *  This module is used display the patch boundaries.  There will be
 *  different methods of visualizing the boundaries (solid color, by x,
 *  y, and z, and random).  The coloring by x,y,z,random is based off of
 *  a color map passed in.  If no colormap is passed in then solid color
 *  coloring will be used.
 *
 *  One can change values for up to 6 different levels.  After that, levels
 *  6 and above will use the settings for level 5.
 *  
 *
 *  Written by:
 *   James Bigler
 *   Department of Computer Science
 *   University of Utah
 *   January 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/ColorMapPort.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/GeomLine.h>
#include <Core/Geom/Material.h>
#include <Core/Geometry/BBox.h>
#include <Core/Geometry/Point.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Dataflow/Ports/ArchivePort.h>
#include <Packages/Uintah/Core/Datatypes/Archive.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <vector>
#include <sstream>
#include <iostream>
#include <float.h>
#include <time.h>
#include <stdlib.h>

namespace Uintah {

using namespace SCIRun;
using namespace std;

#define SOLID 0
#define X_DIM 1
#define Y_DIM 2
#define Z_DIM 3
#define RANDOM 4
  
class PatchVisualizer : public Module {
public:
  PatchVisualizer(GuiContext* ctx);
  virtual ~PatchVisualizer();
  virtual void execute();
  void tcl_command(GuiArgs& args, void* userdata);

private:
  void addBoxGeometry(GeomLines* edges, const Box& box,
		      const Vector & change);
  bool getGrid();
  void setupColors();
  int getScheme(string scheme);
  MaterialHandle getColor(string color, float value);
  
  ArchiveIPort* in;
  GeometryOPort* ogeom;
  ColorMapIPort *inColorMap;
  MaterialHandle level_color[6];
  int level_color_scheme[6];
  
  GuiString level0_grid_color;
  GuiString level1_grid_color;
  GuiString level2_grid_color;
  GuiString level3_grid_color;
  GuiString level4_grid_color;
  GuiString level5_grid_color;
  GuiString level0_color_scheme;
  GuiString level1_color_scheme;
  GuiString level2_color_scheme;
  GuiString level3_color_scheme;
  GuiString level4_color_scheme;
  GuiString level5_color_scheme;
  GuiInt nl;
  GuiInt patch_seperate;
  
  vector< double > times;
  DataArchive* archive;
  int old_generation;
  int old_timestep;
  int numLevels;
  GridP grid;
  
};

static string widget_name("PatchVisualizer Widget");
 
DECLARE_MAKER(PatchVisualizer)

  PatchVisualizer::PatchVisualizer(GuiContext* ctx)
: Module("PatchVisualizer", ctx, Filter, "Visualization", "Uintah"),
  level0_grid_color(ctx->subVar("level0_grid_color")),
  level1_grid_color(ctx->subVar("level1_grid_color")),
  level2_grid_color(ctx->subVar("level2_grid_color")),
  level3_grid_color(ctx->subVar("level3_grid_color")),
  level4_grid_color(ctx->subVar("level4_grid_color")),
  level5_grid_color(ctx->subVar("level5_grid_color")),
  level0_color_scheme(ctx->subVar("level0_color_scheme")),
  level1_color_scheme(ctx->subVar("level1_color_scheme")),
  level2_color_scheme(ctx->subVar("level2_color_scheme")),
  level3_color_scheme(ctx->subVar("level3_color_scheme")),
  level4_color_scheme(ctx->subVar("level4_color_scheme")),
  level5_color_scheme(ctx->subVar("level5_color_scheme")),
  nl(ctx->subVar("nl")),
  patch_seperate(ctx->subVar("patch_seperate")),
  old_generation(-1), old_timestep(0), numLevels(0),
  grid(NULL)
{

  // seed the random number generator 
  time_t t;
  time(&t);
  srand(t);
  
}

PatchVisualizer::~PatchVisualizer()
{
}

// assigns a grid based on the archive and the timestep to grid
// return true if there was a new grid, false otherwise
bool PatchVisualizer::getGrid()
{
  ArchiveHandle handle;
  if(!in->get(handle)){
    std::cerr<<"PatchVisualizer::getGrid() Didn't get a handle\n";
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
  }
  if (timestep != old_timestep) {
    double time = times[timestep];
    grid = archive->queryGrid(time);
    old_timestep = timestep;
    return true;
  }
  return false;
}

// adds the lines to edges that make up the box defined by box 
void PatchVisualizer::addBoxGeometry(GeomLines* edges, const Box& box,
				     const Vector & change)
{
  Point min = box.lower() + change;
  Point max = box.upper() - change;
  
  edges->add(Point(min.x(), min.y(), min.z()),
	     Point(min.x(), min.y(), max.z()));
  edges->add(Point(min.x(), min.y(), min.z()),
	     Point(min.x(), max.y(), min.z()));
  edges->add(Point(min.x(), min.y(), min.z()),
	     Point(max.x(), min.y(), min.z()));
  edges->add(Point(max.x(), min.y(), min.z()),
	     Point(max.x(), max.y(), min.z()));
  edges->add(Point(max.x(), min.y(), min.z()),
	     Point(max.x(), min.y(), max.z()));
  edges->add(Point(min.x(), max.y(), min.z()),
	     Point(max.x(), max.y(), min.z()));
  edges->add(Point(min.x(), max.y(), min.z()),
	     Point(min.x(), max.y(), max.z()));
  edges->add(Point(min.x(), min.y(), min.z()),
	     Point(min.x(), min.y(), max.z()));
  edges->add(Point(min.x(), min.y(), max.z()),
	     Point(max.x(), min.y(), max.z()));
  edges->add(Point(min.x(), min.y(), max.z()),
	     Point(min.x(), max.y(), max.z()));
  edges->add(Point(max.x(), max.y(), min.z()),
	     Point(max.x(), max.y(), max.z()));
  edges->add(Point(max.x(), min.y(), max.z()),
	     Point(max.x(), max.y(), max.z()));
  edges->add(Point(min.x(), max.y(), max.z()),
	     Point(max.x(), max.y(), max.z()));
}

// grabs the colors form the UI and assigns them to the local colors
void PatchVisualizer::setupColors() {
  ////////////////////////////////
  // Set up the colors used

  // assign some colors to the different levels
  level_color[0] = getColor(level0_grid_color.get(),1);
  level_color[1] = getColor(level1_grid_color.get(),1);
  level_color[2] = getColor(level2_grid_color.get(),1);
  level_color[3] = getColor(level3_grid_color.get(),1);
  level_color[4] = getColor(level4_grid_color.get(),1);
  level_color[5] = getColor(level5_grid_color.get(),1);

  // extract the coloring schemes
  level_color_scheme[0] = getScheme(level0_color_scheme.get());
  level_color_scheme[1] = getScheme(level1_color_scheme.get());
  level_color_scheme[2] = getScheme(level2_color_scheme.get());
  level_color_scheme[3] = getScheme(level3_color_scheme.get());
  level_color_scheme[4] = getScheme(level4_color_scheme.get());
  level_color_scheme[5] = getScheme(level5_color_scheme.get());
}

// returns an int corresponding to the string scheme
int PatchVisualizer::getScheme(string scheme) {
  if (scheme == "solid")
    return SOLID;
  if (scheme == "x")
    return X_DIM;
  if (scheme == "y")
    return Y_DIM;
  if (scheme == "z")
    return Z_DIM;
  if (scheme == "random")
    return RANDOM;
  cerr << "PatchVisualizer: Warning: Unknown color scheme!\n";
  return SOLID;
}

// returns a MaterialHandle where the hue is based on the clString passed in
// and the value is based on value.
MaterialHandle PatchVisualizer::getColor(string color, float value) {
  if (color == "red")
    return scinew Material(Color(0,0,0), Color(value,0,0),
			   Color(.5,.5,.5), 20);
  else if (color == "green")
    return scinew Material(Color(0,0,0), Color(0,value,0),
			   Color(.5,.5,.5), 20);
  else if (color == "yellow")
    return scinew Material(Color(0,0,0), Color(value,value,0),
			   Color(.5,.5,.5), 20);
  else if (color == "magenta")
    return scinew Material(Color(0,0,0), Color(value,0,value),
			   Color(.5,.5,.5), 20);
  else if (color == "cyan")
    return scinew Material(Color(0,0,0), Color(0,value,value),
			   Color(.5,.5,.5), 20);
  else if (color == "blue")
    return scinew Material(Color(0,0,0), Color(0,0,value),
			   Color(.5,.5,.5), 20);
  else
    return scinew Material(Color(0,0,0), Color(value,value,value),
			   Color(.5,.5,.5), 20);
}

void PatchVisualizer::execute()
{

  // Create the input port
  in= (ArchiveIPort *) get_iport("Data Archive");
  // color map
  inColorMap =  (ColorMapIPort *) get_iport("ColorMap");
  // Create the output port
  ogeom= (GeometryOPort *) get_oport("Geometry");

  ogeom->delAll();

  // Get the handle on the grid and the number of levels
  bool new_grid = getGrid();
  if(!grid)
    return;
  int numLevels = grid->numLevels();

  
  ColorMapHandle cmap;
  int have_cmap=inColorMap->get( cmap );

  // setup the tickle stuff
  setupColors();
  if (new_grid) {
    nl.set(numLevels);
    string visible;
    gui->eval(id + " isVisible", visible);
    if ( visible == "1") {
      gui->execute(id + " Rebuild");
      
      gui->execute("update idletasks");
      reset_vars();
    }
  }

  //////////////////////////////////////////////////////////////////
  // Extract the geometry from the archive
  //////////////////////////////////////////////////////////////////
  
  // it's faster when we don't use push_back and know the exact size
  vector< vector<Box> > patches(numLevels);

  // initialize the min and max for the entire region.
  // Note: there are already function that compute this, but they currently
  //       iterate over the same data.  Rather than interate over all the data
  //       twice I will compute min/max here while I iterate over the data.
  //       This will make the code faster.
  Point min(DBL_MAX,DBL_MAX,DBL_MAX), max(DBL_MIN,DBL_MIN,DBL_MIN);

  for(int l = 0;l<numLevels;l++){
    LevelP level = grid->getLevel(l);

    vector<Box> patch_list(level->numPatches());
    Level::const_patchIterator iter;
    //---------------------------------------
    // for each patch in the level
    int i = 0;
    for(iter=level->patchesBegin();iter != level->patchesEnd(); iter++){
      const Patch* patch=*iter;
      Box box = patch->getBox();
      patch_list[i] = box;

      // determine boundaries
      if (box.upper().x() > max.x())
	max.x(box.upper().x());
      if (box.upper().y() > max.y())
	max.y(box.upper().y());
      if (box.upper().z() > max.z())
	max.z(box.upper().z());
      
      if (box.lower().x() < min.x())
	min.x(box.lower().x());
      if (box.lower().y() < min.y())
	min.y(box.lower().y());
      if (box.lower().z() < min.z())
	min.z(box.lower().z());
      i++;
    }
    patches[l] = patch_list;
  }
  // the change in size of the patches.
  // This is based on the actual size of the data, so it should be a pretty
  // good guess.
  Vector change_v; 
  if (patch_seperate.get() == 1) {
    double change_factor = 100;
    change_v = Vector((max.x() - min.x())/change_factor,
		      (max.y() - min.y())/change_factor,
		      (max.z() - min.z())/change_factor);
  } else {
    change_v = Vector(0,0,0);
  }

  //////////////////////////////////////////////////////////////////
  // Create the geometry for export
  //////////////////////////////////////////////////////////////////
  
  // loops over all the levels
  for(unsigned int l = 0;l<patches.size();l++){
    // there can be up to 6 levels only, after that the value of the last
    // level is used.
    unsigned int level_index = l;
    if (level_index >= 6)
      level_index = 5;

    // all the geometry for this level.  It will be added to the screen graph
    // seperately in order to be able to select and unselect them from
    // the renderer.
    GeomGroup *level_geom = scinew GeomGroup();

    int scheme;
    // if we don't have a colormap, then use solid color coloring
    if (have_cmap)
      scheme = level_color_scheme[level_index];
    else
      scheme = SOLID;
    
    // generate the geometry based the coloring scheme
    switch (scheme) {
    case SOLID: {
      
      // edges is all the edges made up all the patches in the level
      GeomLines* edges = scinew GeomLines();
      
      //---------------------------------------
      // for each patch in the level
      for(unsigned int i = 0; i < patches[l].size(); i++){
	addBoxGeometry(edges, patches[l][i], change_v);
      }
      
      level_geom->add(scinew GeomMaterial(edges, level_color[level_index]));
    }
    break;
    case X_DIM:
      cmap->Scale(min.x(),max.x());

      //---------------------------------------
      // for each patch in the level
      for(unsigned int i = 0; i < patches[l].size(); i++){
	GeomLines* edges = scinew GeomLines();
	addBoxGeometry(edges, patches[l][i], change_v);
	level_geom->add(scinew GeomMaterial(edges,
			       cmap->lookup(patches[l][i].lower().x())));
      }
      
      break;
    case Y_DIM:
      cmap->Scale(min.y(),max.y());

      //---------------------------------------
      // for each patch in the level
      for(unsigned int i = 0; i < patches[l].size(); i++){
	GeomLines* edges = scinew GeomLines();
	addBoxGeometry(edges, patches[l][i], change_v);
	level_geom->add(scinew GeomMaterial(edges,
			       cmap->lookup(patches[l][i].lower().y())));
      }
      
      break;
    case Z_DIM:
      cmap->Scale(min.z(),max.z());

      //---------------------------------------
      // for each patch in the level
      for(unsigned int i = 0; i < patches[l].size(); i++){
	GeomLines* edges = scinew GeomLines();
	addBoxGeometry(edges, patches[l][i], change_v);
	level_geom->add(scinew GeomMaterial(edges,
			       cmap->lookup(patches[l][i].lower().z())));
      }
      
      break;
    case RANDOM:
      // drand48 returns valued between 0 and 1
      cmap->Scale(0, 1);

      //---------------------------------------
      // for each patch in the level
      for(unsigned int i = 0; i < patches[l].size(); i++){
	GeomLines* edges = scinew GeomLines();
	addBoxGeometry(edges, patches[l][i], change_v);
	level_geom->add(scinew GeomMaterial(edges, cmap->lookup(drand48())));
      }
      
      break;
    } // end of switch
    
    // add all the edges for the level
    ostringstream name_edges;
    name_edges << "Patches - level " << l;
    ogeom->addObj(level_geom, name_edges.str().c_str());
  }
}

// This is called when the tcl code explicity calls a function besides
// needexecute.
void PatchVisualizer::tcl_command(GuiArgs& args, void* userdata)
{
  if(args.count() < 2) {
    args.error("Streamline needs a minor command");
    return;
  }
  else {
    Module::tcl_command(args, userdata);
  }
}


} // End namespace Uintah

