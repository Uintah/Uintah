/*
 *  ShowGeometry.cc
 *
 *  Written by:
 *   Martin Cole
 *   School of Computing
 *   University of Utah
 *   Aug 31, 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#include <PSECore/Dataflow/Module.h>
#include <PSECore/Datatypes/ColorMapPort.h>
#include <PSECore/Datatypes/GeometryPort.h>
#include <SCICore/Datatypes/Mesh.h>
#include <SCICore/Datatypes/Lattice3Geom.h>
#include <SCICore/Datatypes/SField.h>
#include <PSECore/Datatypes/SFieldPort.h>
#include <SCICore/Geom/GeomGroup.h>
#include <SCICore/Geom/Material.h>
#include <SCICore/Geom/Switch.h>
#include <SCICore/Geom/GeomSphere.h>
#include <SCICore/Geom/GeomLine.h>
#include <SCICore/TclInterface/TCLvar.h>
#include <SCICore/Util/DebugStream.h>

#include <map.h>
#include <iostream>
using std::cerr;
using std::cin;
using std::endl;
#include <sstream>
using std::ostringstream;

namespace PSECommon {
namespace Modules {

using namespace PSECore::Datatypes;
using namespace SCICore::Geometry;
using namespace SCICore::TclInterface;
using namespace SCICore::GeomSpace;

class ShowGeometry : public Module 
{
  
  // GROUP: Private Data
  ///////////////////////////
  // 
  // all private data
  

  DebugStream              d_dbg;  
  SFieldIPort*             d_infield;
  SFieldHandle             d_sfield;
  GeometryOPort           *d_ogeom;  

  // scene graph ID's
  int                      d_nodeId;
  int                      d_conId;

  // Materials
  MaterialHandle           d_nodeMat;
  MaterialHandle           d_conMat;
  
  // top level nodes for
  GeomSwitch*              d_conSwitch;
  GeomSwitch*              d_nodeSwitch;

  TCLstring                d_nodeDisplayType;
  TCLint                   d_showConP;
  TCLint                   d_showProgress;

  // Display color for nodes.
  TCLdouble                d_nodeChanR;
  TCLdouble                d_nodeChanG;
  TCLdouble                d_nodeChanB;
  
  // Display color for connections.
  TCLdouble                d_conChanR;
  TCLdouble                d_conChanG;
  TCLdouble                d_conChanB;

  
  // GROUP: Private Methods
  ///////////////////////////
  // 
  // all private methods

  //////////
  // displayNode
  // 
  inline void displayNode(bool val) { 
    if (d_nodeSwitch) {
      d_nodeSwitch->set_state(val);
    }
  }

  //////////
  // displayConnections
  // 
  inline void displayConnections(bool val) { 
    if (d_conSwitch) {
      d_conSwitch->set_state(val);
    }
  }

  //////////
  // reloadNodeColor
  // 
  inline void reloadConColor() {
    d_conMat->diffuse = Color(d_conChanR.get(), d_conChanG.get(), 
			      d_conChanB.get());
  }
  
  inline void reloadNodeColor() {
    d_nodeMat->diffuse = Color(d_nodeChanR.get(), d_nodeChanG.get(), 
			       d_nodeChanB.get());
  }
  
public:
  // GROUP:  Constructors:
  ///////////////////////////
  // Constructs an instance of class ShowGeometry
  //


  //////////
  // ShowGeometry
  // 
  ShowGeometry(const clString& id) : 
    Module("ShowGeometry", id, Filter), 
    d_showProgress("show_progress", id, this), 
    d_nodeDisplayType("nodeDisplayType", id, this),
    d_showConP("showConP", id, this),
    d_nodeChanR("nodeChan-r", id, this), 
    d_nodeChanG("nodeChan-g", id, this),
    d_nodeChanB("nodeChan-b", id, this),
    d_conChanR("conChan-r", id, this), 
    d_conChanG("conChan-g", id, this),
    d_conChanB("conChan-b", id, this),
    d_dbg("ShowGeometry", true),
    d_nodeId(0),
    d_conId(0),
    d_conSwitch(NULL),
    d_nodeSwitch(NULL)
  {
    // Create the input ports
    d_infield = scinew SFieldIPort(this, "Field", SFieldIPort::Atomic);
    add_iport(d_infield);
    
    // Create the output port
    d_ogeom = scinew GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
    add_oport(d_ogeom);

    d_nodeMat = scinew Material(Color(0,.3,0), 
				Color(d_nodeChanR.get(),
				      d_nodeChanG.get(),
				      d_nodeChanB.get()),
				Color(.7,.7,.7), 50);
    
    d_conMat = scinew Material(Color(0,.3,0), 
			       Color(d_conChanR.get(),
				     d_conChanG.get(),
				     d_conChanB.get()),
			       Color(.7,.7,.7), 50);

  }

  // GROUP:  Destructor:
  ///////////////////////////
  // Destructor
  virtual ~ShowGeometry() {}

  //////////
  // execute
  // 
  virtual void execute()
  {
    // Check for generation number. FIX_ME

    // tell module downstream to delete everything we have sent it before.
    // This is typically salmon, it owns the scene graph memory we create here.
    d_ogeom->delAll(); 
    d_infield->get(d_sfield);
    if(!d_sfield.get_rep()){
      return;
    }

    d_dbg << d_sfield->get_attrib()->get_info();
  
    Gradient *gradient = d_sfield->query_interface((Gradient *)0);
    if(!gradient){
      error("Gradient not supported by input field");
    }

    SLInterpolate *slinterpolate = 
      d_sfield->query_interface((SLInterpolate *)0);
    if(!slinterpolate){
      error("SLInterpolate not supported by input field");
    }

    BBox bbox;
    d_sfield->get_geom()->get_bbox(bbox);

    d_dbg << bbox.min().x() << " " << bbox.min().y() << " " 
	  << bbox.min().z() << " " << bbox.max().x() << " " << bbox.max().y() 
	  << " " << bbox.max().z() << endl;

    GeomGroup *bb = scinew GeomGroup;
    d_conSwitch = scinew GeomSwitch(bb);

    GeomGroup *verts = scinew GeomGroup;
    Geom *geom = d_sfield->get_geom();
    Lattice3Geom* grid = dynamic_cast<Lattice3Geom*>(geom);
    //LatticeGeom *grid = geom->get_latticegeom();

    if (grid) {

      int nx = grid->get_nx();
      int ny = grid->get_ny();
      int nz = grid->get_nz();

      d_dbg << "grid->nx" << nx << endl;
      d_dbg << "grid->ny" << ny << endl;
      d_dbg << "grid->nz" << nz << endl;

      Vector xDir, yDir, zDir;
      double sx, sy, sz, aveScale;

      setUpDirs(xDir, yDir, zDir, sx, sy, sz, grid, bbox);
      aveScale = (sx + sy + sz) * 0.33L;
      bb->add(scinew GeomSphere(bbox.min(), aveScale*2.0, 8, 4));
      bb->add(scinew GeomSphere(bbox.max(), aveScale*2.0, 8, 4));
      

      for (int i = 0; i < nx; i++) {
	for (int j = 0; j < ny; j++) {
	  for (int k = 0; k < nz; k++) {	
	    if (d_nodeDisplayType.get() == "Spheres") {
	      addSphere(i, j, k, grid, verts, aveScale);
	    } else {
	      addAxis(i, j, k, xDir, yDir, zDir, grid, verts);
	    }
	    addConnections(i, j, k, 
			   i >= nx - 1, j >= ny - 1, k >= nz - 1,
			   grid, bb);
	  }
	}
      }
    } else {
      d_dbg << "Not a LatticGeom!!" << endl;
    }

    reloadNodeColor();
    GeomMaterial *nodeGM = scinew GeomMaterial(verts, d_nodeMat);
    d_nodeSwitch = scinew GeomSwitch(nodeGM);

    reloadConColor();
    GeomMaterial *conGM = scinew GeomMaterial(bb, d_conMat);
    d_conSwitch = scinew GeomSwitch(conGM);
  
    d_nodeId = d_ogeom->addObj(d_nodeSwitch, "Nodes");
    d_conId = d_ogeom->addObj(d_conSwitch, "Connections");
    d_ogeom->flushViews();

  }

  //////////
  // addConnections
  // 
  inline void addConnections(int i, int j, int k, 
			     bool lastI, bool lastJ, bool lastK,
			     Lattice3Geom *grid, GeomGroup *g) {
    Point p0 = grid->get_point(i, j, k);
    Point p1;
    if (! lastI) {
      p1 = grid->get_point(i + 1, j, k);
      g->add(new GeomLine(p0, p1));
    }
    if (! lastJ) {
      p1 = grid->get_point(i, j + 1, k);
      g->add(new GeomLine(p0, p1));
    }
    if (! lastK) {
      p1 = grid->get_point(i, j, k + 1);
      g->add(new GeomLine(p0, p1));
    }

  }

  //////////
  // addSphere
  // 
  inline void addSphere(int i, int j, int k, Lattice3Geom *grid, 
			GeomGroup *g, double size) {

    Point p0 = grid->get_point(i, j, k);
    g->add(scinew GeomSphere(p0, size, 8, 4));
  }

  //////////
  // addAxis
  // 
  inline void addAxis(int i, int j, int k,
		      Vector &x, Vector &y, Vector &z, 
		      Lattice3Geom *grid, GeomGroup *g) {

    Point p0 = grid->get_point(i, j, k);
    Point p1 = p0 + x;
    Point p2 = p0 - x;
    g->add(new GeomLine(p1, p2));
    p1 = p0 + y;
    p2 = p0 - y;
    g->add(new GeomLine(p1, p2));
    p1 = p0 + z;
    p2 = p0 - z;
    g->add(new GeomLine(p1, p2));
  }

  //////////
  // tcl_command
  // 
  virtual void tcl_command(TCLArgs& args, void* userdata) {
    if(args.count() < 2){
      args.error("ShowGeometry needs a minor command");
      return;
    }
    d_dbg << "tcl_command: " << args[1] << endl;

    if (args[1] == "nodeSphereP") {
      // toggle spheresP
      // Call reset so that we really get the value, not the cached one.
      d_nodeDisplayType.reset();
      if (d_nodeDisplayType.get() == "Spheres") {
	d_dbg << "Render Spheres." << endl;
      } else {
	d_dbg << "Render Axes." << endl;
      }
      // Tell salmon to redraw itself.
      d_ogeom->flushViews();

    } else if (args[1] == "connectionDisplayChange"){

      // Toggle the GeomSwitch.
      bool toggle = ! d_conSwitch->get_state();
      d_conSwitch->set_state(toggle);
      // Tell salmon to redraw itself.
      d_ogeom->flushViews();

    } else if (args[1] == "conColorChange"){

      // Fetch correct values from TCL
      d_conChanR.reset();
      d_conChanG.reset();
      d_conChanB.reset();
      // Set new color in material.
      reloadConColor();
      // Tell salmon to redraw itself.
      d_ogeom->flushViews();

    } else if (args[1] == "nodeColorChange"){

      // Fetch correct values from TCL.
      d_nodeChanR.reset();
      d_nodeChanG.reset();
      d_nodeChanB.reset();
      // Set new color in material.
      reloadNodeColor();
      // Tell salmon to redraw itself.
      d_ogeom->flushViews();

    } else {
      Module::tcl_command(args, userdata);
    }
  }

  //////////
  // setUpDirs
  // 
  void setUpDirs(Vector &x, Vector &y, Vector &z, 
		 double &sx, double &sy,  double &sz, 
		 Lattice3Geom *grid, BBox &bbox) {

    sx = (bbox.max().x() - bbox.min().x()) * 0.2L;
    sy = (bbox.max().y() - bbox.min().y()) * 0.2L;
    sz = (bbox.max().z() - bbox.min().z()) * 0.2L;
    d_dbg << "sx: " << sx << endl;
    d_dbg << "sy: " << sy << endl;
    d_dbg << "sz: " << sz << endl;

    int nx = grid->get_nx();
    int ny = grid->get_ny();
    int nz = grid->get_nz();

    sx /= nx;
    sy /= ny;
    sz /= nz;

    if (nx > 0) {
      Point p0 = grid->get_point(0, 0, 0);
      Point p1 = grid->get_point(1, 0, 0);
      x = p1 - p0;
    } else {
      x.x(1.0L);
      x.y(0.0L);
      x.z(0.0L);
    }

    if (ny > 0) {
      Point p0 = grid->get_point(0, 0, 0);
      Point p1 = grid->get_point(0, 1, 0);
      y = p1 - p0;
    } else {
      y.x(0.0L);
      y.y(1.0L);
      y.z(0.0L);
    }
      
    if (nz > 0) {
      Point p0 = grid->get_point(0, 0, 0);
      Point p1 = grid->get_point(0, 0, 1);
      z = p1 - p0;
    } else {
      z.x(0.0L);
      z.y(0.0L);
      z.z(1.0L);
    }

    //Scale the dirs...
    x.normalize();
    x *= sx;
    y.normalize();
    y *= sy;
    z.normalize();
    z *= sz;
  }
};

extern "C" Module* make_ShowGeometry(const clString& id) {
  return new ShowGeometry(id);
}

} // End namespace Modules
} // End namespace PSECommon

