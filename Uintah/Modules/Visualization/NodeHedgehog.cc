//static char *id="@(#) $Id$";

/*
 *  NodeHedgehog.cc:  
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   June 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#include <SCICore/Containers/Array1.h>
#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/ColorMapPort.h>
#include <PSECore/Datatypes/GeometryPort.h>
#include <PSECore/Datatypes/ScalarFieldPort.h>
#include <PSECore/Datatypes/VectorFieldPort.h>
#include <SCICore/Geom/GeomArrows.h>
#include <SCICore/Geom/GeomGroup.h>
#include <SCICore/Geom/GeomLine.h>
#include <SCICore/Geom/Material.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Math/MinMax.h>
#include <SCICore/Malloc/Allocator.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <SCICore/Thread/CrowdMonitor.h>
#include <Uintah/Grid/GridP.h>
#include <Uintah/Grid/Grid.h>
#include <Uintah/Grid/Level.h>
#include <Uintah/Grid/Patch.h>
#include <Uintah/Grid/NodeIterator.h>
#include <Uintah/Datatypes/ArchivePort.h>
#include <Uintah/Datatypes/Archive.h>

#include <PSECore/Widgets/ScaledBoxWidget.h>
#include <PSECore/Widgets/ScaledFrameWidget.h>
#include <iostream>
#include <vector>

using std::cerr;
using std::vector;

#define CP_PLANE 0
#define CP_SURFACE 1
#define CP_CONTOUR 2

namespace PSECommon {
namespace Modules {

using namespace PSECore::Dataflow;
using namespace PSECore::Datatypes;
using namespace PSECore::Widgets;
using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;
using namespace SCICore::Math;
using namespace SCICore::Containers;
using namespace Uintah::Datatypes;
using namespace SCICore::Geometry;

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
  VectorFieldIPort *invectorfield;
  ScalarFieldIPort* inscalarfield;
  ColorMapIPort *inColorMap;
  GeometryOPort* ogeom;
  CrowdMonitor widget_lock;
  int init;
  int widget_id;
  ScaledBoxWidget* widget3d;
  ScaledFrameWidget *widget2d;
  
  // GROUP:  Widgets:
  //////////////////////
  //
  // widget_moved -  
  virtual void widget_moved(int last);

  // GROUP: Private Function:
  //////////////////////
  //
  // getGrid -
  GridP getGrid();
  TCLdouble length_scale;
  TCLdouble width_scale;
  TCLdouble head_length;
  TCLstring type;
  TCLint exhaustive_flag;
  TCLint drawcylinders;
  TCLint skip_node;
  TCLdouble shaft_rad;
  MaterialHandle outcolor;
  int grid_id;
  int need_find2d;
  int need_find3d;
  
public:
 
        // GROUP:  Constructors:
        ///////////////////////////
        //
        // Constructs an instance of class NodeHedgehog
        //
        // Constructor taking
        //    [in] id as an identifier
        //
   NodeHedgehog(const clString& id);
       
        // GROUP:  Destructor:
        ///////////////////////////
        // Destructor
   virtual ~NodeHedgehog();
  
        // GROUP:  Access functions:
        ///////////////////////////
        //
        // execute() - execution scheduled by scheduler
   virtual void execute();

        //////////////////////////
        //
        // tcl_commands - overides tcl_command in base class Module, takes:
        //                                  findxy,
        //                                  findyz,
        //                                  findxz
   virtual void tcl_command(TCLArgs&, void*);
};

static clString module_name("NodeHedgehog");
static clString widget_name("NodeHedgehog Widget");

extern "C" Module* make_NodeHedgehog(const clString& id) {
  return new NodeHedgehog(id);
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

NodeHedgehog::NodeHedgehog(const clString& id)
: Module("NodeHedgehog", id, Filter),
  widget_lock("NodeHedgehog widget lock"),
  length_scale("length_scale", id, this),
  width_scale("width_scale", id, this),
  head_length("head_length", id, this),
  type("type", id, this),
  drawcylinders("drawcylinders", id, this),
  skip_node("skip_node", id, this),
  shaft_rad("shaft_rad", id, this),
  exhaustive_flag("exhaustive_flag", id, this)
{
  // Create the input ports
  // grid
  ingrid=new ArchiveIPort(this, "Data Archive",
		      ArchiveIPort::Atomic);
  add_iport(ingrid);

  // scalar field
  inscalarfield = scinew ScalarFieldIPort( this, "Scalar Field",
					   ScalarFieldIPort::Atomic);
  add_iport( inscalarfield);

  // vector field
  invectorfield = scinew VectorFieldIPort( this, "Vector Field",
					   VectorFieldIPort::Atomic);
  add_iport( invectorfield);

  // color map
  inColorMap = scinew ColorMapIPort( this, "ColorMap",
				     ColorMapIPort::Atomic);
  add_iport( inColorMap);
  
// Create the output port
  ogeom = scinew GeometryOPort(this, "Geometry", 
			       GeometryIPort::Atomic);
  add_oport(ogeom);
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

void NodeHedgehog::execute()
{
  int old_grid_id = grid_id;

  // Must have a vector field and a grid, otherwise exit
  VectorFieldHandle vfield;
  if (!invectorfield->get( vfield ))
    return;

  GridP grid = getGrid();
  if (!grid)
    return;
  
  // Get the scalar field and ColorMap...if you can
  ScalarFieldHandle ssfield;
  int have_sfield=inscalarfield->get( ssfield );

  ColorMapHandle cmap;
  int have_cmap=inColorMap->get( cmap );
  if(!have_cmap)
    have_sfield=0;
  
  if (init == 1) {
    init = 0;
    GeomObj *w2d = widget2d->GetWidget() ;
    GeomObj *w3d = widget3d->GetWidget() ;
    GeomGroup* w = new GeomGroup;
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
  double ld=vfield->longest_dimension();
  if (do_3d){
    if(need_find3d != 0){
      Point min, max;
      vfield->get_bounds( min, max );
      Point center = min + (max-min)/2.0;
      Point right( max.x(), center.y(), center.z());
      Point down( center.x(), min.y(), center.z());
      Point in( center.x(), center.y(), min.z());
      widget3d->SetPosition( center, right, down, in);
      widget3d->SetScale( ld/20. );
    }
    need_find3d = 0;
  } else {
    if (need_find2d != 0){
      Point min, max;
      vfield->get_bounds( min, max );
      
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
	max_scale = Max( (max.x() - min.x()), (max.z() - min.z()) );
      } else {
	// Find the field and put in optimal place
	// in xz plane with reasonable frame thickness
	Point right( max.x(), center.y(), center.z());
	Point down( center.x(), center.y(), min.z());	    
	widget2d->SetPosition( center, right, down);
      }
      widget2d->SetScale( max_scale/20. );
      need_find2d = 0;
    }
  }
  
  // get the position of the frame widget and determine
  // the scale of the sliders.  Only the value on one of
  // the sliders is used to determine the density of the
  // vectors.
  Point center, R, D, I;
  // because skip is used for the interator increment
  // it must be 1 or more otherwise you enter an
  // infinite loop (and that really sucks for performance)
  int skip = skip_node.get();
  if (skip < 1)
    skip = 1;
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
  int exhaustive = exhaustive_flag.get();
  GeomArrows* arrows = new GeomArrows(widscale, 1.0-headlen, drawcylinders.get(), shaft_rad.get() );

  // loop over all the nodes in the graph taking the position
  // of the node and determining the vector value.
  int numLevels = grid->numLevels();

  //-----------------------------------------
  // for each level in the grid
  for(int l = 0;l<numLevels;l++){
    LevelP level = grid->getLevel(l);

    Level::const_patchIterator iter;
    //---------------------------------------
    // for each patch in the level
    for(iter=level->patchesBegin(); iter != level->patchesEnd(); iter++){
      Patch* patch = *iter;
      //------------------------------------
      // for each node in the patch
      for(NodeIterator iter = patch->getNodeIterator(boundaryRegion); !iter.done(); iter+=skip){
	Point p = patch->nodePosition(*iter);
	Vector vv;
	int ii = 0;
	if (vfield->interpolate( p, vv, ii, exhaustive)){
	  if(have_sfield) {
	    // get the color from cmap for p 	    
	    MaterialHandle matl;
	    double sval;
	    if (ssfield->interpolate( p, sval, ii, exhaustive))
	      matl = cmap->lookup( sval);
	    else {
	      matl = outcolor;
	    }
	    
	    if(vv.length2()*lenscale > 1.e-3) {
	      arrows->add(p, vv*lenscale, matl, matl, matl);
	    }
	  }
	  else {
	    if(vv.length2()*lenscale > 1.e-3) {
	      arrows->add(p, vv*lenscale);
	    }
	  }
	}
      }
    }
  }

  // delete the old grid/cutting plane
  if (old_grid_id != 0)
    ogeom->delObj( old_grid_id );
  
  grid_id = ogeom->addObj(arrows, module_name);
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

} // End namespace Modules
} // End namespace PSECommon

//
// $Log$
// Revision 1.2  2000/06/27 17:13:19  bigler
// Took out some old code.
//
// Revision 1.1  2000/06/27 15:23:56  bigler
// Inital versions of a module that works like Hedgehog
// except the vectors are originated from the node positions.
//
// Revision 1.8  2000/03/17 09:27:32  sparker
// New makefile scheme: sub.mk instead of Makefile.in
// Use XML-based files for module repository
// Plus many other changes to make these two things work
//
// Revision 1.7  1999/10/07 02:07:06  sparker
// use standard iostreams and complex type
//
// Revision 1.6  1999/08/29 00:46:46  sparker
// Integrated new thread library
// using statement tweaks to compile with both MipsPRO and g++
// Thread library bug fixes
//
// Revision 1.5  1999/08/25 03:48:07  sparker
// Changed SCICore/CoreDatatypes to SCICore/Datatypes
// Changed PSECore/CommonDatatypes to PSECore/Datatypes
// Other Misc. directory tree updates
//
// Revision 1.4  1999/08/19 23:17:57  sparker
// Removed a bunch of #include <SCICore/Util/NotFinished.h> statements
// from files that did not need them.
//
// Revision 1.3  1999/08/18 20:20:07  sparker
// Eliminated copy constructor and clone in all modules
// Added a private copy ctor and a private clone method to Module so
//  that future modules will not compile until they remvoe the copy ctor
//  and clone method
// Added an ASSERTFAIL macro to eliminate the "controlling expression is
//  constant" warnings.
// Eliminated other miscellaneous warnings
//
// Revision 1.2  1999/08/17 06:37:49  sparker
// Merged in modifications from PSECore to make this the new "blessed"
// version of SCIRun/Uintah.
//
// Revision 1.1  1999/07/27 16:58:13  mcq
// Initial commit
//
// Revision 1.3  1999/06/21 23:52:52  dav
// updated makefiles.main
//
// Revision 1.2  1999/05/11 19:48:03  dav
// updated Hedgehog
//
// Revision 1.1.1.1  1999/04/24 23:12:33  dav
// Import sources
//
//
