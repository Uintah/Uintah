
/*
 *  NodeHedgehog.cc:  
 *
 *  Written by:
 *   James Bigler & Steven G. Parker 
 *   Department of Computer Science
 *   University of Utah
 *   June 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#include <Core/Containers/Array1.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/ColorMapPort.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Geom/GeomArrows.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/GeomLine.h>
#include <Core/Geom/Material.h>
#include <Core/Geometry/Point.h>
#include <Core/Math/MinMax.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Thread/CrowdMonitor.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Grid/GridP.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/NodeIterator.h>
#include <Packages/Uintah/Core/Grid/CellIterator.h>
#include <Packages/Uintah/Dataflow/Ports/ArchivePort.h>
#include <Packages/Uintah/Core/Datatypes/Archive.h>

#include <Core/Datatypes/LatticeVol.h>
#include <Core/Datatypes/LatVolMesh.h>
#include <Dataflow/Widgets/ScaledBoxWidget.h>
#include <Dataflow/Widgets/ScaledFrameWidget.h>
#include <Packages/Uintah/Core/Datatypes/LevelMesh.h>
#include <Packages/Uintah/Core/Datatypes/LevelField.h>

#include <iostream>
#include <vector>

using std::cerr;
using std::vector;

#define CP_PLANE 0
#define CP_SURFACE 1
#define CP_CONTOUR 2

namespace Uintah {

using namespace SCIRun;

/**************************************
CLASS
   NodeHedgehog
      NodeHedgehog produces arrows at node locations in a vector field.


GENERAL INFORMATION

   NodeHedgehog
  
   Author:  Steven G. Parker (sparker@cs.utah.edu)
            James Bigler (bigler@cs.utah.edu)
            
            Department of Computer Science
            
            University of Utah
   
   Date:    June 2000
   
   C-SAFE
   
   Copyright <C> 2000 SCI Group

KEYWORDS
   Visualization, vector_field, GenColormap

DESCRIPTION
   NodeHedgehog produces arrows at node locations in a vector
   field.  The length of the arrow indicates the magnitude of the
   field at that point, and the orientation indicates the
   direction of the field.  In addition, the shaft of the arrow
   is mapped to a scalar field value using a colormap produced by
   GenColormap.

WARNING
   None



****************************************/

class NodeHedgehog : public Module {
  ArchiveIPort* ingrid;
  FieldIPort *invectorfield;
  FieldIPort* inscalarfield;
  ColorMapIPort *inColorMap;
  GeometryOPort* ogeom;
  CrowdMonitor widget_lock;
  int init;
  int widget_id;
  ScaledBoxWidget* widget3d;
  ScaledFrameWidget *widget2d;
  
  // GROUP:  Widgets:
  //////////////////////
  // widget_moved -  
  virtual void widget_moved(int last);

  // GROUP: Private Function:
  //////////////////////
  // getGrid -
  GridP getGrid();

  // GROUP: Private Function:
  //////////////////////
  // add_arrow -
  void add_arrow(FieldHandle vfield, FieldHandle ssfield,
		 int have_sfield, ColorMapHandle cmap,
		 double lenscale, GeomArrows* arrows, Point p);

  GuiDouble length_scale;
  GuiDouble width_scale;
  GuiDouble head_length;
  GuiString type;
  GuiInt drawcylinders;
  GuiInt skip_node;
  GuiDouble shaft_rad;
  GuiInt var_orientation; // whether node center or cell centered
  MaterialHandle outcolor;
  int grid_id;
  int need_find2d;
  int need_find3d;

  // max Vector stuff
  Vector max_vector;
  double max_length;
  GuiDouble max_vector_x;
  GuiDouble max_vector_y;
  GuiDouble max_vector_z;
  GuiDouble max_vector_length;
public:
 
  // GROUP:  Constructors:
  ///////////////////////////
  // Constructs an instance of class NodeHedgehog
  // Constructor taking
  //    [in] id as an identifier
  NodeHedgehog(const string& id);
  
  // GROUP:  Destructor:
  ///////////////////////////
  // Destructor
  virtual ~NodeHedgehog();
  
  // GROUP:  Access functions:
  ///////////////////////////
  // execute() - execution scheduled by scheduler
  virtual void execute();
  
  //////////////////////////
  // tcl_commands - overides tcl_command in base class Module, takes:
  //                                  findxy,
  //                                  findyz,
  //                                  findxz
  virtual void tcl_command(TCLArgs&, void*);

  // used to get the values.
  bool interpolate(FieldHandle f, const Point& p, Vector& val);
  bool interpolate(FieldHandle f, const Point& p, double& val);
};
  
static string module_name("NodeHedgehog");
static string widget_name("NodeHedgehog Widget");

extern "C" Module* make_NodeHedgehog(const string& id) {
  return scinew NodeHedgehog(id);
}       

// obtains a pointer to the grid via the Archive in Port
GridP NodeHedgehog::getGrid()
{
  ArchiveHandle handle;
  if(!ingrid->get(handle)){
    std::cerr<<"Didn't get a handle on archive\n";
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

NodeHedgehog::NodeHedgehog(const string& id)
: Module("NodeHedgehog", id, Filter, "Visualization", "Uintah"),
  widget_lock("NodeHedgehog widget lock"),
  length_scale("length_scale", id, this),
  width_scale("width_scale", id, this),
  head_length("head_length", id, this),
  type("type", id, this),
  drawcylinders("drawcylinders", id, this),
  skip_node("skip_node", id, this),
  shaft_rad("shaft_rad", id, this),
  var_orientation("var_orientation",id,this),
  max_vector(0,0,0), max_length(0),
  max_vector_x("max_vector_x", id, this),
  max_vector_y("max_vector_y", id, this),
  max_vector_z("max_vector_z", id, this),
  max_vector_length("max_vector_length", id, this)
{
  init = 1;
  float INIT(.1);
  
  widget2d = scinew ScaledFrameWidget(this, &widget_lock, INIT);
  widget3d = scinew ScaledBoxWidget(this, &widget_lock, INIT);
  grid_id=0;
  
  need_find2d=1;
  need_find3d=1;
  
  drawcylinders.set(0);
  outcolor=scinew Material(Color(0,0,0), Color(0,0,0), Color(0,0,0), 0);
}

NodeHedgehog::~NodeHedgehog()
{
}

void
NodeHedgehog::add_arrow(FieldHandle vfield, FieldHandle ssfield,
			int have_sfield, ColorMapHandle cmap,
			double lenscale, GeomArrows* arrows, Point p) {
  Vector vv;
  if (interpolate( vfield, p, vv)){
    double length = vv.length();
    if (length > max_length) {
      max_length = length;
      max_vector = vv;
    }
    if(have_sfield) {
      // get the color from cmap for p 	    
      MaterialHandle matl;
      double sval;
      if (interpolate( ssfield, p, sval))
	matl = cmap->lookup( sval);
      else {
	matl = outcolor;
      }

      if(length*lenscale > 1.e-3) {
	arrows->add(p, vv*lenscale, matl, matl, matl);
      }
    }
    else {
      if(length*lenscale > 1.e-3) {
	arrows->add(p, vv*lenscale);
      }
    }
  }
}

void NodeHedgehog::execute()
{
  int old_grid_id = grid_id;

    // Create the input port
  ingrid = (ArchiveIPort *) get_iport("Data Archive");
  inscalarfield = (FieldIPort *) get_iport("Scalar Field");
  invectorfield = (FieldIPort *) get_iport("Vector Field");
  inColorMap = (ColorMapIPort *) get_iport("ColorMap");
					
  // Create the output port
  ogeom = (GeometryOPort *) get_oport("Geometry"); 
  // Must have a vector field and a grid, otherwise exit
  FieldHandle vfield;
  if (!invectorfield->get( vfield ))
    return;

  if( vfield->get_type_name(1) != "Vector" ){
    cerr<<"First field must be a Vector field.\n";
    return;
  }
  
  if(vfield->get_type_name(0) != "LevelField" &&
     vfield->get_type_name(0) != "LatticeVol" ){
    cerr<<"Not a LatticeVol or LevelField\n";
  }

  GridP grid = getGrid();
  if (!grid)
    return;
  
  // Get the scalar field and ColorMap...if you can
  FieldHandle ssfield;
  int have_sfield=inscalarfield->get( ssfield );
  if( have_sfield ){
    if( !ssfield->is_scalar() ){
      cerr<<"Second field is not a scalar field.  No Colormapping.\n";
      have_sfield = 0;
    }
  }
  ColorMapHandle cmap;
  int have_cmap=inColorMap->get( cmap );
  if(!have_cmap)
    have_sfield=0;
  
  if (init == 1) {
    init = 0;
    GeomObj *w2d = widget2d->GetWidget() ;
    GeomObj *w3d = widget3d->GetWidget() ;
    GeomGroup* w = scinew GeomGroup;
    w->add(w2d);
    w->add(w3d);
    widget_id = ogeom->addObj( w, widget_name, &widget_lock );
    
    widget2d->Connect( ogeom );
    widget2d->SetRatioR( 1/20.0 );
    widget2d->SetRatioD( 1/20.0 );
    
    widget3d->Connect( ogeom );
    widget3d->SetRatioR( 1/20.0 );
    widget3d->SetRatioD( 1/20.0 );
    widget3d->SetRatioI( 1/20.0 );
  }
  int do_3d=1;
  if(type.get() == "2D")
    do_3d=0;

  // turn on/off widgets
  widget2d->SetState(!do_3d);
  widget3d->SetState(do_3d);
  // set thier mode to resize/translate only
  widget2d->SetCurrentMode(6);
  widget3d->SetCurrentMode(6);

  // draws the widget based on the vfield's size and position
  if (do_3d){
    if(need_find3d != 0){
      BBox box;
      Point min, max;
      box = vfield->mesh()->get_bounding_box();
      min = box.min(); max = box.max();
      Point center = min + (max-min)/2.0;
      Point right( max.x(), center.y(), center.z());
      Point down( center.x(), min.y(), center.z());
      Point in( center.x(), center.y(), min.z());
      widget3d->SetPosition( center, right, down, in);
      double ld = Max (Max( (max.x() - min.x()), (max.y() - min.y()) ),
		       max.z() - min.z());
      widget3d->SetScale( ld/20. );
    }
    need_find3d = 0;
  } else {
    if (need_find2d != 0){
      BBox box;
      Point min, max;
      box = vfield->mesh()->get_bounding_box();
      min = box.min(); max = box.max();
      
      Point center = min + (max-min)/2.0;
      double max_scale;
      if (need_find2d == 1) {
	// Find the field and put in optimal place
	// in xy plane with reasonable frame thickness
	Point right( max.x(), center.y(), center.z());
	Point down( center.x(), min.y(), center.z());
	widget2d->SetPosition( center, right, down);
	max_scale = Max( (max.x() - min.x()), (max.y() - min.y()) );
      } else if (need_find2d == 2) {
	// Find the field and put in optimal place
	// in yz plane with reasonable frame thickness
	Point right( center.x(), center.y(), max.z());
	Point down( center.x(), min.y(), center.z());	    
	widget2d->SetPosition( center, right, down);
	max_scale = Max( (max.z() - min.z()), (max.y() - min.y()) );
      } else {
	// Find the field and put in optimal place
	// in xz plane with reasonable frame thickness
	Point right( max.x(), center.y(), center.z());
	Point down( center.x(), center.y(), min.z());	    
	widget2d->SetPosition( center, right, down);
	max_scale = Max( (max.x() - min.x()), (max.z() - min.z()) );
      }
      widget2d->SetScale( max_scale/20. );
      need_find2d = 0;
    }
  }
  
  // because skip is used for the interator increment
  // it must be 1 or more otherwise you enter an
  // infinite loop (and that really sucks for performance)
  int skip = skip_node.get();
  if (skip < 1)
    skip = 1;
  // get the position of the frame widget and determine
  // the boundaries
  Point center, R, D, I;
  if(do_3d){
    widget3d->GetPosition( center, R, D, I);
  } else {
    widget2d->GetPosition( center, R, D);
    I = center;
  }
  Vector v1 = R - center;
  Vector v2 = D - center;
  Vector v3 = I - center;

  // calculate the edge points
  Point upper = center + v1 + v2 + v3;
  Point lower = center - v1 - v2 - v3;

  // need to determine extents of lower/upper
  Point temp1 = upper;
  upper = Max(temp1,lower);
  lower = Min(temp1,lower);
  Box boundaryRegion(lower,upper);
  
  // create the grid for the cutting plane
  double lenscale = length_scale.get();
  double widscale = width_scale.get();
  double headlen = head_length.get();
  GeomArrows* arrows = scinew GeomArrows(widscale, 1.0-headlen, drawcylinders.get(), shaft_rad.get() );

  // loop over all the nodes in the graph taking the position
  // of the node and determining the vector value.
  int numLevels = grid->numLevels();

  // reset max_length and max_vector
  max_length = 0;
  max_vector = Vector(0,0,0);
  
  //-----------------------------------------
  // for each level in the grid
  for(int l = 0;l<numLevels;l++){
    LevelP level = grid->getLevel(l);

    Level::const_patchIterator iter;
    //---------------------------------------
    // for each patch in the level
    for(iter=level->patchesBegin(); iter != level->patchesEnd(); iter++){
      Patch* patch = *iter;


      switch (var_orientation.get()) {
      case 0: //NC_VARIABLE
	//------------------------------------
	// for each node in the patch
	for(NodeIterator iter = patch->getNodeIterator(boundaryRegion); !iter.done(); iter+=skip){
	  Point p = patch->nodePosition(*iter);
	  add_arrow(vfield, ssfield, have_sfield, cmap,
		    lenscale, arrows, p);
	} // end for loop
	break;
      case 1: // CC_VARIABLE
	for(CellIterator iter = patch->getCellIterator(boundaryRegion); !iter.done(); iter+=skip){
	  Point p = patch->cellPosition(*iter);
	  add_arrow(vfield, ssfield, have_sfield, cmap,
		    lenscale, arrows, p);
	}
	break;
      } // end switch
    }
  }

  // delete the old grid/cutting plane
  if (old_grid_id != 0)
    ogeom->delObj( old_grid_id );
  
  grid_id = ogeom->addObj(arrows, module_name);

  // set the max vector stuff in the tcl code
  max_vector_length.set(max_length);
  max_vector_x.set(max_vector.x());
  max_vector_y.set(max_vector.y());
  max_vector_z.set(max_vector.z());
}

void NodeHedgehog::widget_moved(int last)
{
  if(last && !abort_flag) {
    abort_flag=1;
    want_to_execute();
  }
}


void NodeHedgehog::tcl_command(TCLArgs& args, void* userdata)
{
  if(args.count() < 2) {
    args.error("Streamline needs a minor command");
    return;
  }
  if(args[1] == "findxy") {
    if(type.get() == "2D")
      need_find2d=1;
    else
      need_find3d=1;
    want_to_execute();
  }
  else if(args[1] == "findyz") {
    if(type.get() == "2D")
      need_find2d=2;
    else
      need_find3d=1;
    want_to_execute();
  }
  else if(args[1] == "findxz") {
    if(type.get() == "2D")
      need_find2d=3;
    else
      need_find3d=1;
    want_to_execute();
  }
  else {
    Module::tcl_command(args, userdata);
  }
}

bool  
NodeHedgehog::interpolate(FieldHandle vfld, const Point& p, Vector& val)
{
  const string field_type = vfld->get_type_name(0);
  const string type = vfld->get_type_name(1);
  if( field_type == "LevelField"){
    if (type == "Vector") {
      if( LevelField<Vector> *fld =
	dynamic_cast<LevelField<Vector>*>(vfld.get_rep())){
	return fld->interpolate(val ,p);
      } else {
	return false;
      }
    } else {
      cerr << "Uintah::NodeHedgehog::interpolate:: error - unknown Field type: " << type << endl;
      return false;
    }
  } else {
    if( field_type == "LatticeVol"){
   // use virtual field interpolation
      VectorFieldInterface *vfi;
      if( vfi = vfld->query_vector_interface()){
	return vfi->interpolate( val, p);
      } 
    }    
    return false;
  }
  return false;
}


bool  
NodeHedgehog::interpolate(FieldHandle sfld, const Point& p, double& val)
{
  const string field_type = sfld->get_type_name(0);
  const string type = sfld->get_type_name(1);
  if( sfld->get_type_name(0) == "LevelField" ){
    if (type == "double") {
      if(LevelField<double> *fld =
	dynamic_cast<LevelField<double>*>(sfld.get_rep())){;
	return fld->interpolate(val ,p);
      } else {
	return false;
      }
    } else if (type == "float") {
      float result;
      bool success;
      LevelField<float> *fld =
	dynamic_cast<LevelField<float>*>(sfld.get_rep());
      success = fld->interpolate(result ,p);   
      val = (double)result;
      return success;
    } else if (type == "long") {
      long result;
      bool success;
      LevelField<long> *fld =
	dynamic_cast<LevelField<long>*>(sfld.get_rep());
      success =  fld->interpolate(result,p);
      val = (double)result;
      return success;
    } else {
      cerr << "Uintah::NodeHedgehog::interpolate:: error - unimplemented Field type: " << type << endl;
      return false;
    }
  } else if( sfld->get_type_name(0) == "LatticeVol" ){
    // use virtual field interpolation
    ScalarFieldInterface *sfi;
    if( sfi = sfld->query_scalar_interface()){
      return sfi->interpolate( val, p);
    }
  } else {
    return false;
  }
}

} // End namespace Uintah


