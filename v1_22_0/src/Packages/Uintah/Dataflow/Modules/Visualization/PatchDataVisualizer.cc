/*
 *  PatchVisualizer.cc:  Displays Patch boundaries
 *
 *  This module displays spheres on the center of the patches.  In the future
 *  it may be able to color the spheres based on data associated with the
 *  patch.
 *
 *  One can change values for up to 6 different levels.  After that, levels
 *  6 and above will use the settings for level 5.
 *  
 *
 *  Written by:
 *   James Bigler
 *   Department of Computer Science
 *   University of Utah
 *   June 2001
 *
 *  Copyright (C) 2001 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/ColorMapPort.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/GeomLine.h>
#include <Core/Geom/GeomPick.h>
#include <Core/Geom/GeomSphere.h>
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

struct PatchData {
  Vector loc;
  int level;
  int id;
  double val;
};
  
class PatchDataVisualizer : public Module {
public:
  PatchDataVisualizer(GuiContext* ctx);
  virtual ~PatchDataVisualizer();
  virtual void execute();
  void tcl_command(GuiArgs& args, void* userdata);

private:
  void addBoxGeometry(GeomLines* edges, const Box& box,
		      const Vector & change);
  bool getGrid();
  virtual void geom_pick(GeomPickHandle pick, void* userdata, GeomHandle picked);

  ArchiveIPort* in;
  GeometryOPort* ogeom;
  ColorMapIPort *inColorMap;
  MaterialHandle level_color[6];
  int level_color_scheme[6];
  
  void getnunv(int* nu, int* nv);
  GuiDouble radius;
  GuiInt polygons;
  
  DataArchive* archive;
  int old_generation;
  int old_timestep;
  int numLevels;
  GridP grid;
  
  // this is the collection of PatchData that we are collecting
  vector < PatchData > patch_centers;
};

static string widget_name("PatchDataVisualizer Widget");
 
DECLARE_MAKER(PatchDataVisualizer)

  PatchDataVisualizer::PatchDataVisualizer(GuiContext* ctx)
: Module("PatchDataVisualizer", ctx, Filter, "Visualization", "Uintah"),
  radius(ctx->subVar("radius")),
  polygons(ctx->subVar("polygons")),
  old_generation(-1), old_timestep(0),
  grid(NULL)
{
  radius.set(0.01);
}

PatchDataVisualizer::~PatchDataVisualizer()
{
}

void PatchDataVisualizer::getnunv(int* nu, int* nv) {
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

// assigns a grid based on the archive and the timestep to grid
// return true if there was a new grid, false otherwise
bool PatchDataVisualizer::getGrid()
{
  ArchiveHandle handle;
  if(!in->get(handle)){
    std::cerr<<"PatchDataVisualizer::getGrid() Didn't get a handle\n";
    grid = NULL;
    return false;
  }

  // access the grid through the handle and dataArchive
  archive = (*(handle.get_rep()))();
  int new_generation = (*(handle.get_rep())).generation;
  bool archive_dirty =  new_generation != old_generation;
  int timestep = (*(handle.get_rep())).timestep();
  vector< double > times; 
  if (archive_dirty) {
    old_generation = new_generation;
    vector< int > indices;
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

void PatchDataVisualizer::execute()
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
  if (new_grid) {
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
  
  // Note: there are already function that compute this, but they currently
  //       iterate over the same data.  Rather than interate over all the data
  //       twice I will compute min/max here while I iterate over the data.
  //       This will make the code faster.
  Point min(DBL_MAX,DBL_MAX,DBL_MAX), max(DBL_MIN,DBL_MIN,DBL_MIN);

  // this is the collection of PatchData that we are collecting
  patch_centers.clear();

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

      // extract the data we need
      PatchData data;
      data.loc = (box.upper() - box.lower()) / 2;
      data.level = l;
      data.id = patch->getID();
      // data.val can any scalar value castable to a double
      data.val = data.id;
      patch_centers.push_back(data);
      
      // extend grid boundaries to include patch
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
  }

  //////////////////////////////////////////////////////////////////
  // Create the geometry for export
  //////////////////////////////////////////////////////////////////
  
  // find the bounds on the data values
  double data_min, data_max;
  data_min = data_max = patch_centers[0].val;
  for (unsigned int i = 0; i < patch_centers.size(); i++) {
    if (patch_centers[0].val < data_min)
      data_min = patch_centers[0].val;
    if (patch_centers[0].val > data_max)
      data_max = patch_centers[0].val;
  }

  // Now add all the spheres
  GeomGroup* spheres = scinew GeomGroup;
  int nu,nv;
  double rad = radius.get();
  getnunv(&nu,&nv);

  // now add the geometry
  for (unsigned int i = 0; i < patch_centers.size(); i++) {
    GeomSphere * sphere =scinew GeomSphere((Point)patch_centers[i].loc,
					   rad,nu,nv);
    sphere->setId((int)i);
    sphere->setId(IntVector(patch_centers[i].id,
			    patch_centers[i].level,
			    0));
    double normal = (data_max - patch_centers[i].val)/(data_max - data_min);
    if (have_cmap) 
      spheres->add(scinew GeomMaterial(sphere, cmap->lookup(normal)));
    else
      spheres->add(scinew GeomMaterial((GeomObj*)sphere,
				       scinew Material(Color(0,0,0),
						       Color(0,1,0),
						       Color(.5,.5,.5), 20)));
  }
  
  GeomPick* pick = scinew GeomPick(spheres,this);
  ogeom->addObj(pick,"Patch Centers");
}

// This is called when the tcl code explicity calls a function besides
// needexecute.
void PatchDataVisualizer::tcl_command(GuiArgs& args, void* userdata)
{
  if(args.count() < 2) {
    args.error("Streamline needs a minor command");
    return;
  }
  else {
    Module::tcl_command(args, userdata);
  }
}

void PatchDataVisualizer::geom_pick(GeomPickHandle /*pick*/, void* /*userdata*/,
				    GeomHandle picked)
{
#if DEBUG
  cerr << "Caught pick event in PatchDataVisualizer!\n";
  cerr << "this = " << this << ", pick = " << pick << endl;
  cerr << "User data = " << userdata << endl;
#endif
  IntVector id;
  int index;
  if ( picked->getId( id ) && picked->getId(index)) {
    cerr<<"Index = " << index << "Id = "<< id.x() << " Level = " << id.y() << " Value = " << patch_centers[index].val << endl;
  }
  else
    cerr<<"Not getting the correct data\n";
}

} // End namespace Uintah

