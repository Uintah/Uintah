/*
 *  ShowField.cc
 *
 *  Written by:
 *   Martin Cole
 *   School of Computing
 *   University of Utah
 *   Aug 31, 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/ColorMapPort.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Core/Datatypes/Mesh.h>
#include <Core/Datatypes/LatticeGeom.h>
#include <Core/Datatypes/TriSurfGeom.h>
#include <Core/Datatypes/Field.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/Material.h>
#include <Core/Geom/Switch.h>
#include <Core/Geom/GeomSphere.h>
#include <Core/Geom/GeomLine.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Util/DebugStream.h>

#include <map.h>
#include <iostream>
using std::cerr;
using std::cin;
using std::endl;
#include <sstream>
using std::ostringstream;

namespace SCIRun {


class ShowField : public Module 
{
  
  // GROUP: Private Data
  ///////////////////////////
  // all private data
  

  DebugStream              dbg_;  
  FieldIPort*              infield_;
  FieldHandle              sfield_;
  GeometryOPort           *ogeom_;  

  // scene graph ID's
  int                      nodeId_;
  int                      conId_;

  // Materials
  MaterialHandle           nodeMat_;
  MaterialHandle           conMat_;
  
  // top level nodes for
  GeomSwitch*              conSwitch_;
  GeomSwitch*              nodeSwitch_;

  GuiString                nodeDisplayType_;
  GuiInt                   showConP_;
  GuiInt                   showProgress_;

  // Display color for nodes.
  GuiDouble                nodeChanR_;
  GuiDouble                nodeChanG_;
  GuiDouble                nodeChanB_;
  
  // Display color for connections.
  GuiDouble                conChanR_;
  GuiDouble                conChanG_;
  GuiDouble                conChanB_;

  
  // GROUP: Private Methods
  ///////////////////////////
  // all private methods

  //////////
  // displayNode
  inline void displayNode(bool val) { 
    if (nodeSwitch_) {
      nodeSwitch_->set_state(val);
    }
  }

  //////////
  // displayConnections
  inline void displayConnections(bool val) { 
    if (conSwitch_) {
      conSwitch_->set_state(val);
    }
  }

  //////////
  // reloadNodeColor
  inline void reloadConColor() {
    conMat_->diffuse = Color(conChanR_.get(), conChanG_.get(), 
			      conChanB_.get());
  }
  
  inline void reloadNodeColor() {
    nodeMat_->diffuse = Color(nodeChanR_.get(), nodeChanG_.get(), 
			       nodeChanB_.get());
  }
  
public:
  // GROUP:  Constructors:
  ///////////////////////////
  // Constructs an instance of class ShowField


  //////////
  // ShowField
  ShowField(const clString& id) : 
    Module("ShowField", id, Filter), 
    showProgress_("show_progress", id, this), 
    nodeDisplayType_("nodeDisplayType", id, this),
    showConP_("showConP", id, this),
    nodeChanR_("nodeChan-r", id, this), 
    nodeChanG_("nodeChan-g", id, this),
    nodeChanB_("nodeChan-b", id, this),
    conChanR_("conChan-r", id, this), 
    conChanG_("conChan-g", id, this),
    conChanB_("conChan-b", id, this),
    dbg_("ShowField", true),
    nodeId_(0),
    conId_(0),
    conSwitch_(NULL),
    nodeSwitch_(NULL)
  {
    // Create the input ports
    infield_ = scinew FieldIPort(this, "Field", FieldIPort::Atomic);
    add_iport(infield_);
    
    // Create the output port
    ogeom_ = scinew GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
    add_oport(ogeom_);

    nodeMat_ = scinew Material(Color(0,.3,0), 
				Color(nodeChanR_.get(),
				      nodeChanG_.get(),
				      nodeChanB_.get()),
				Color(.7,.7,.7), 50);
    
    conMat_ = scinew Material(Color(0,.3,0), 
			       Color(conChanR_.get(),
				     conChanG_.get(),
				     conChanB_.get()),
			       Color(.7,.7,.7), 50);

  }

  // GROUP:  Destructor:
  ///////////////////////////
  // Destructor
  virtual ~ShowField() {}

  //////////
  // execute
  virtual void execute()
  {
    // Check for generation number. FIX_ME
    cerr << "Starting Execution..." << endl;
    dbg_ << "SHOWGEOMETRY EXECUTING" << endl;

    // tell module downstream to delete everything we have sent it before.
    // This is typically salmon, it owns the scene graph memory we create here.
    
    ogeom_->delAll(); 
    infield_->get(sfield_);
    if(!sfield_.get_rep()){
      return;
    }

    if (!sfield_->getAttrib().get_rep())
      cerr << "NO attribute!!1" << endl;
    dbg_ << sfield_->getAttrib()->getInfo();

    Gradient *gradient = sfield_->query_interface((Gradient *)0);
    if(!gradient){
      error("Gradient not supported by input field");
    }

    SLInterpolate *slinterpolate = 
      sfield_->query_interface((SLInterpolate *)0);
    if(!slinterpolate){
      error("SLInterpolate not supported by input field");
    }

    BBox bbox;
    sfield_->get_geom()->getBoundingBox(bbox);

    dbg_ << bbox.min().x() << " " << bbox.min().y() << " " 
	  << bbox.min().z() << " " << bbox.max().x() << " " << bbox.max().y() 
	  << " " << bbox.max().z() << endl;

    GeomGroup *bb = scinew GeomGroup;
    conSwitch_ = scinew GeomSwitch(bb);

    GeomGroup *verts = scinew GeomGroup;
    GeomHandle geom = sfield_->get_geom();

    dbg_ << geom->getInfo();

    LatticeGeomHandle grid = geom->downcast((LatticeGeom*)0);
    //LatticeGeom *grid = geom->get_latticegeom();
    TriSurfGeomHandle tsurf = geom->downcast((TriSurfGeom*)0);

    dbg_ << "Cast to Lattice and TriSurf is done \n" << endl;
    if (grid.get_rep()) {

      int nx = grid->getSizeX();
      int ny = grid->getSizeY();
      int nz = grid->getSizeZ();

      Vector xDir, yDir, zDir;
      double sx, sy, sz, aveScale;

      setUpDirs(xDir, yDir, zDir, sx, sy, sz, grid, bbox);
      aveScale = (sx + sy + sz) * 0.33L;
      bb->add(scinew GeomSphere(bbox.min(), aveScale*2.0, 8, 4));
      bb->add(scinew GeomSphere(bbox.max(), aveScale*2.0, 8, 4));
      

      for (int i = 0; i < nx; i++) {
	for (int j = 0; j < ny; j++) {
	  for (int k = 0; k < nz; k++) {	
	    if (nodeDisplayType_.get() == "Spheres") {
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
    } else if (tsurf.get_rep()) {
      dbg_ << "Writing out surface" << endl;
      displaySurface(tsurf, verts);
    } else {
      dbg_ << "Not a LatticGeom!!" << endl;
    }

    reloadNodeColor();
    GeomMaterial *nodeGM = scinew GeomMaterial(verts, nodeMat_);
    nodeSwitch_ = scinew GeomSwitch(nodeGM);

    reloadConColor();
    GeomMaterial *conGM = scinew GeomMaterial(bb, conMat_);
    conSwitch_ = scinew GeomSwitch(conGM);
  
    nodeId_ = ogeom_->addObj(nodeSwitch_, "Nodes");
    conId_ = ogeom_->addObj(conSwitch_, "Connections");
    ogeom_->flushViews();

  }


  void displaySurface(TriSurfGeomHandle surf, GeomGroup *g)
  {
    BBox bbox;
    surf->getBoundingBox(bbox);
    
    const double pointsize =
      bbox.diagonal().length() / pow(surf->pointSize(), 0.33L) * 0.1;
    
    int i;
    for (i = 0; i < surf->pointSize(); i++) {
      const Point &p = surf->point(i);
      g->add(scinew GeomSphere(p, pointsize, 8, 4));
    }

    for (i = 0; i < surf->triangleSize(); i++) {
      const int p0i = surf->edge(i*3+0).pointIndex();
      const int p1i = surf->edge(i*3+1).pointIndex();
      const int p2i = surf->edge(i*3+2).pointIndex();

      const Point &p0 = surf->point(p0i);
      const Point &p1 = surf->point(p1i);
      const Point &p2 = surf->point(p2i);

      // draw three half edges
      g->add(new GeomLine(p0, p1));
      g->add(new GeomLine(p1, p2));
      g->add(new GeomLine(p2, p0));
      
      // draw three connections.
    }
  }


  //////////
  // addConnections
  inline void addConnections(int i, int j, int k, 
			     bool lastI, bool lastJ, bool lastK,
			     LatticeGeomHandle grid, GeomGroup *g) {
    Point p0 = grid->getPoint(i, j, k);
    Point p1;
    if (! lastI) {
      p1 = grid->getPoint(i + 1, j, k);
      g->add(new GeomLine(p0, p1));
    }
    if (! lastJ) {
      p1 = grid->getPoint(i, j + 1, k);
      g->add(new GeomLine(p0, p1));
    }
    if (! lastK) {
      p1 = grid->getPoint(i, j, k + 1);
      g->add(new GeomLine(p0, p1));
    }

  }

  //////////
  // addSphere
  inline void addSphere(int i, int j, int k, LatticeGeomHandle grid, 
			GeomGroup *g, double size) {

    Point p0 = grid->getPoint(i, j, k);
    g->add(scinew GeomSphere(p0, size, 8, 4));
  }

  //////////
  // addAxis
  inline void addAxis(int i, int j, int k,
		      Vector &x, Vector &y, Vector &z, 
		      LatticeGeomHandle grid, GeomGroup *g) {

    Point p0 = grid->getPoint(i, j, k);
    Point p1 = p0 + x;
    Point p2 = p0 - x;
    GeomLine *l = new GeomLine(p1, p2);
    l->setLineWidth(3.0);
    g->add(l);
    p1 = p0 + y;
    p2 = p0 - y;
    l = new GeomLine(p1, p2);
    l->setLineWidth(3.0);
    g->add(l);
    p1 = p0 + z;
    p2 = p0 - z;
    l = new GeomLine(p1, p2);
    l->setLineWidth(3.0);
    g->add(l);
  }

  //////////
  // tcl_command
  virtual void tcl_command(TCLArgs& args, void* userdata) {
    if(args.count() < 2){
      args.error("ShowField needs a minor command");
      return;
    }
    dbg_ << "tcl_command: " << args[1] << endl;

    if (args[1] == "nodeSphereP") {
      // toggle spheresP
      // Call reset so that we really get the value, not the cached one.
      nodeDisplayType_.reset();
      if (nodeDisplayType_.get() == "Spheres") {
	dbg_ << "Render Spheres." << endl;
      } else {
	dbg_ << "Render Axes." << endl;
      }
      // Tell salmon to redraw itself.
      ogeom_->flushViews();

    } else if (args[1] == "connectionDisplayChange"){

      // Toggle the GeomSwitch.
      bool toggle = ! conSwitch_->get_state();
      conSwitch_->set_state(toggle);
      // Tell salmon to redraw itself.
      ogeom_->flushViews();

    } else if (args[1] == "conColorChange"){

      // Fetch correct values from TCL
      conChanR_.reset();
      conChanG_.reset();
      conChanB_.reset();
      // Set new color in material.
      reloadConColor();
      // Tell salmon to redraw itself.
      ogeom_->flushViews();

    } else if (args[1] == "nodeColorChange"){

      // Fetch correct values from TCL.
      nodeChanR_.reset();
      nodeChanG_.reset();
      nodeChanB_.reset();
      // Set new color in material.
      reloadNodeColor();
      // Tell salmon to redraw itself.
      ogeom_->flushViews();

    } else {
      Module::tcl_command(args, userdata);
    }
  }

  //////////
  // setUpDirs
  void setUpDirs(Vector &x, Vector &y, Vector &z, 
		 double &sx, double &sy,  double &sz, 
		 LatticeGeomHandle grid, BBox &bbox) {

    sx = (bbox.max().x() - bbox.min().x()) * 0.2L;
    sy = (bbox.max().y() - bbox.min().y()) * 0.2L;
    sz = (bbox.max().z() - bbox.min().z()) * 0.2L;
    dbg_ << "sx: " << sx << endl;
    dbg_ << "sy: " << sy << endl;
    dbg_ << "sz: " << sz << endl;

    int nx = grid->getSizeX();
    int ny = grid->getSizeY();
    int nz = grid->getSizeZ();

    sx /= nx;
    sy /= ny;
    sz /= nz;

    if (nx > 0) {
      Point p0 = grid->getPoint(0, 0, 0);
      Point p1 = grid->getPoint(1, 0, 0);
      x = p1 - p0;
    } else {
      x.x(1.0L);
      x.y(0.0L);
      x.z(0.0L);
    }

    if (ny > 0) {
      Point p0 = grid->getPoint(0, 0, 0);
      Point p1 = grid->getPoint(0, 1, 0);
      y = p1 - p0;
    } else {
      y.x(0.0L);
      y.y(1.0L);
      y.z(0.0L);
    }
      
    if (nz > 0) {
      Point p0 = grid->getPoint(0, 0, 0);
      Point p1 = grid->getPoint(0, 0, 1);
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

extern "C" Module* make_ShowField(const clString& id) {
  return new ShowField(id);
}

} // End namespace SCIRun

