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
#include <Core/Malloc/Allocator.h>
#include <Dataflow/Network/Module.h>
#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/TetVol.h>
#include <Core/Datatypes/LatticeVol.h>
#include <Core/Datatypes/FieldAlgo.h>

#include <Dataflow/Ports/ColorMapPort.h>
#include <Dataflow/Ports/GeometryPort.h>
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
  
  //! Private Data
  DebugStream              dbg_;  

  //! input ports
  FieldIPort*              geom_;
  FieldIPort*              data_;
  ColorMapIPort*           color_;
  //! output port
  GeometryOPort           *ogeom_;  

  //! Scene graph ID's
  int                      nodeId_;
  int                      conId_;

  //! Materials
  MaterialHandle           nodeMat_;
  MaterialHandle           conMat_;
  
  //! top level nodes for switching on and off..
  //! connections.
  GeomSwitch*              conSwitch_;
  //! and nodes.
  GeomSwitch*              nodeSwitch_;

  GuiString                nodeDisplayType_;
  GuiInt                   showConP_;
  GuiInt                   showProgress_;

  //! Display color for nodes.
  GuiDouble                nodeChanR_;
  GuiDouble                nodeChanG_;
  GuiDouble                nodeChanB_;
  
  //! Display color for connections.
  GuiDouble                conChanR_;
  GuiDouble                conChanG_;
  GuiDouble                conChanB_;

  
  //! Private Methods
  inline void displayNode(bool val);
  inline void displayConnections(bool val);
  inline void reloadConColor();
  inline void reloadNodeColor();

  template <class Field>
  void render(Field *f);

public:
  ShowField(const clString& id);
  virtual ~ShowField();
  virtual void execute();
//    void displaySurface(TriSurfGeomHandle surf, GeomGroup *g);
//    inline void addConnections(int i, int j, int k, 
//  			     bool lastI, bool lastJ, bool lastK,
//  			     LatticeGeomHandle grid, GeomGroup *g);
//    inline void addSphere(int i, int j, int k, LatticeGeomHandle grid, 
//  			GeomGroup *g, double size);

  inline void addAxis(Point p0, double scale, GeomGroup *g);
  virtual void tcl_command(TCLArgs& args, void* userdata);

//    void setUpDirs(Vector &x, Vector &y, Vector &z, 
//  		 double &sx, double &sy,  double &sz, 
//  		 LatticeGeomHandle grid, BBox &bbox);
};

void 
ShowField::displayNode(bool val) { 
  if (nodeSwitch_) {
    nodeSwitch_->set_state(val);
  }
}

void 
ShowField::displayConnections(bool val) { 
  if (conSwitch_) {
    conSwitch_->set_state(val);
  }
}

void 
ShowField::reloadConColor() {
  conMat_->diffuse = Color(conChanR_.get(), conChanG_.get(), 
			   conChanB_.get());
}

void 
ShowField::reloadNodeColor() {
  nodeMat_->diffuse = Color(nodeChanR_.get(), nodeChanG_.get(), 
			    nodeChanB_.get());
}
  

ShowField::ShowField(const clString& id) : 
  Module("ShowField", id, Filter), 
  dbg_("ShowField", true),
  nodeId_(0),
  conId_(0),
  conSwitch_(NULL),
  nodeSwitch_(NULL),
  nodeDisplayType_("nodeDisplayType", id, this),
  showConP_("showConP", id, this),
  showProgress_("show_progress", id, this), 
  nodeChanR_("nodeChan-r", id, this), 
  nodeChanG_("nodeChan-g", id, this),
  nodeChanB_("nodeChan-b", id, this),
  conChanR_("conChan-r", id, this), 
  conChanG_("conChan-g", id, this),
  conChanB_("conChan-b", id, this)
{
  // Create the input ports
  geom_ = scinew FieldIPort(this, "Field-Geometry", FieldIPort::Atomic);
  add_iport(geom_);
  data_ = scinew FieldIPort(this, "Field-ColorIndex", FieldIPort::Atomic);
  add_iport(data_);
  color_ = scinew ColorMapIPort(this, "ColorMap", FieldIPort::Atomic);
  add_iport(color_);
    
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

ShowField::~ShowField() {}

void 
ShowField::execute()
{

  // Check for generation number. FIX_ME
  cerr << "Starting Execution..." << endl;
  dbg_ << "SHOWGEOMETRY EXECUTING" << endl;

  // tell module downstream to delete everything we have sent it before.
  // This is typically salmon, it owns the scene graph memory we create here.
    
  ogeom_->delAll(); 
  FieldHandle geom_handle;
  geom_->get(geom_handle);
  if(!geom_handle.get_rep()){
    cerr << "No Geometry in port 1 field" << endl;
    return;
  }
  FieldHandle data_handle;
  data_->get(data_handle);
  if(!data_handle.get_rep()){
    cerr << "No Data in port 2 field" << endl;
    return;
  }
  ColorMapHandle color_handle;
  color_->get(color_handle);
  if(!color_handle.get_rep()){
    cerr << "No ColorMap in port 3 ColorMap" << endl;
    return;
  }

  bool error = false;
  string msg;
  string name = geom_handle->get_type_name(0);
  if (name == "TetVol") {
    if (geom_handle->get_type_name(1) == "double") {
      TetVol<double> *tv = 0;
      tv = dynamic_cast<TetVol<double>*>(geom_handle.get_rep());
      if (tv) { render(tv); }
      else { error = true; msg = "Not a valid TetVol."; }
    } else {
      error = true; msg ="TetVol of unknown type.";
    }
  } /*else if (name == "LatticeVol") {
    if (geom_handle->get_type_name(1) == "double") {
      LatticeVol<double> *lv = 0;
      lv = dynamic_cast<LatticeVol<double>*>(geom_handle.get_rep());
      if (lv) { render(lv); }
      else { error = true; msg = "Not a valid LatticeVol."; }
    } else {
      error = true; msg ="LatticeVol of unknown type.";
    }
    }*/ else if (error) {
    cerr << "ShowField Error: " << msg << endl;
    return;
  }

  nodeId_ = ogeom_->addObj(nodeSwitch_, "Nodes");
  conId_ = ogeom_->addObj(conSwitch_, "Connections");
  ogeom_->flushViews();


}

template <class Field>
void 
ShowField::render(Field *f) {

  GeomGroup *bb = scinew GeomGroup;
  conSwitch_ = scinew GeomSwitch(bb);

  GeomGroup *verts = scinew GeomGroup;
  typename Field::mesh_handle_type mesh = f->get_typed_mesh();

  // First pass over the nodes
  typename Field::mesh_type::node_iterator niter = mesh->node_begin();  
  while (niter != mesh->node_end()) {
    
    Point p;
    mesh->get_point(p, *niter);
    if (nodeDisplayType_.get() == "Spheres") {
      verts->add(scinew GeomSphere(p, 0.03, 8, 4));
    } else {
      addAxis(p, 0.03, verts);
    }
    ++niter;

  }
  
  // Second pass over the edges
  mesh->compute_edges();
  typename Field::mesh_type::edge_iterator eiter = mesh->edge_begin();  
  while (eiter != mesh->edge_end()) {        
    typename Field::mesh_type::node_array nodes;
    mesh->get_nodes(nodes, *eiter); ++eiter;
    
    Point p1, p2;
    mesh->get_point(p1, nodes[0]);
    mesh->get_point(p2, nodes[1]);
    verts->add(new GeomLine(p1, p2));
  }

  reloadNodeColor();
  GeomMaterial *nodeGM = scinew GeomMaterial(verts, nodeMat_);
  nodeSwitch_ = scinew GeomSwitch(nodeGM);

  reloadConColor();
  GeomMaterial *conGM = scinew GeomMaterial(bb, conMat_);
  conSwitch_ = scinew GeomSwitch(conGM);
  

#if 0 // old code for reference.
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
#endif
}

//  void 
//  ShowField::displaySurface(TriSurfGeomHandle surf, GeomGroup *g)
//  {
//    BBox bbox;
//    surf->getBoundingBox(bbox);
    
//    const double pointsize =
//      bbox.diagonal().length() / pow(surf->pointSize(), 0.33L) * 0.1;
    
//    int i;
//    for (i = 0; i < surf->pointSize(); i++) {
//      const Point &p = surf->point(i);
//      g->add(scinew GeomSphere(p, pointsize, 8, 4));
//    }

//    for (i = 0; i < surf->triangleSize(); i++) {
//      const int p0i = surf->edge(i*3+0).pointIndex();
//      const int p1i = surf->edge(i*3+1).pointIndex();
//      const int p2i = surf->edge(i*3+2).pointIndex();

//      const Point &p0 = surf->point(p0i);
//      const Point &p1 = surf->point(p1i);
//      const Point &p2 = surf->point(p2i);

//      // draw three half edges
//      g->add(new GeomLine(p0, p1));
//      g->add(new GeomLine(p1, p2));
//      g->add(new GeomLine(p2, p0));
      
//      // draw three connections.
//    }
//  }


//  void 
//  ShowField::addConnections(int i, int j, int k, 
//  			  bool lastI, bool lastJ, bool lastK,
//  			  LatticeGeomHandle grid, GeomGroup *g) {
//    Point p0 = grid->getPoint(i, j, k);
//    Point p1;
//    if (! lastI) {
//      p1 = grid->getPoint(i + 1, j, k);
//      g->add(new GeomLine(p0, p1));
//    }
//    if (! lastJ) {
//      p1 = grid->getPoint(i, j + 1, k);
//      g->add(new GeomLine(p0, p1));
//    }
//    if (! lastK) {
//      p1 = grid->getPoint(i, j, k + 1);
//      g->add(new GeomLine(p0, p1));
//    }

//  }


//  void 
//  ShowField::addSphere(int i, int j, int k, LatticeGeomHandle grid, 
//  		     GeomGroup *g, double size) {
  
//    Point p0 = grid->getPoint(i, j, k);
//    g->add(scinew GeomSphere(p0, size, 8, 4));
//  }

void 
ShowField::addAxis(Point p0, double scale, GeomGroup *g) 
{
  const Vector x(1., 0., 0.);
  const Vector y(0., 1., 0.);
  const Vector z(0., 0., 1.);

  Point p1 = p0 + x * scale;
  Point p2 = p0 - x * scale;
  GeomLine *l = new GeomLine(p1, p2);
  l->setLineWidth(3.0);
  g->add(l);
  p1 = p0 + y * scale;
  p2 = p0 - y * scale;
  l = new GeomLine(p1, p2);
  l->setLineWidth(3.0);
  g->add(l);
  p1 = p0 + z * scale;
  p2 = p0 - z * scale;
  l = new GeomLine(p1, p2);
  l->setLineWidth(3.0);
  g->add(l);
}

void 
ShowField::tcl_command(TCLArgs& args, void* userdata) {
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

//  void 
//  ShowField::setUpDirs(Vector &x, Vector &y, Vector &z, 
//  		     double &sx, double &sy,  double &sz, 
//  		     LatticeGeomHandle grid, BBox &bbox) {

//    sx = (bbox.max().x() - bbox.min().x()) * 0.2L;
//    sy = (bbox.max().y() - bbox.min().y()) * 0.2L;
//    sz = (bbox.max().z() - bbox.min().z()) * 0.2L;
//    dbg_ << "sx: " << sx << endl;
//    dbg_ << "sy: " << sy << endl;
//    dbg_ << "sz: " << sz << endl;

//    int nx = grid->getSizeX();
//    int ny = grid->getSizeY();
//    int nz = grid->getSizeZ();

//    sx /= nx;
//    sy /= ny;
//    sz /= nz;

//    if (nx > 0) {
//      Point p0 = grid->getPoint(0, 0, 0);
//      Point p1 = grid->getPoint(1, 0, 0);
//      x = p1 - p0;
//    } else {
//      x.x(1.0L);
//      x.y(0.0L);
//      x.z(0.0L);
//    }

//    if (ny > 0) {
//      Point p0 = grid->getPoint(0, 0, 0);
//      Point p1 = grid->getPoint(0, 1, 0);
//      y = p1 - p0;
//    } else {
//      y.x(0.0L);
//      y.y(1.0L);
//      y.z(0.0L);
//    }
      
//    if (nz > 0) {
//      Point p0 = grid->getPoint(0, 0, 0);
//      Point p1 = grid->getPoint(0, 0, 1);
//      z = p1 - p0;
//    } else {
//      z.x(0.0L);
//      z.y(0.0L);
//      z.z(1.0L);
//    }

//    //Scale the dirs...
//    x.normalize();
//    x *= sx;
//    y.normalize();
//    y *= sy;
//    z.normalize();
//    z *= sz;
//  }

extern "C" Module* make_ShowField(const clString& id) {
  return new ShowField(id);
}

} // End namespace SCIRun


