//static char *id="@(#) $Id$";

/*
 *  FaceCuttingPlane.cc:  
 *
 *  Written by:
 *   Kurt Zimmerman
 *   Department of Computer Science
 *   University of Utah
 *   August 2002
 *
 *  Copyright (C) 2002 SCI Group
 */

/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/


/*
 * FaceCuttingPlane.cc
 *
 * 
 */


#include <Core/Datatypes/Field.h>
#include <Core/Datatypes/FieldInterface.h>
#include <Core/Datatypes/LatVolField.h>
#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Geometry/Point.h>
#include <Core/Geom/ColorMap.h>
#include <Core/Geom/GeomTriangles.h>
#include <Core/Geom/View.h>
#include <Core/Geom/GeomGrid.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/GeomLine.h>
#include <Core/Geom/Material.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MinMax.h>
#include <Core/Thread/CrowdMonitor.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/ColorMapPort.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Dataflow/Widgets/PointWidget.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include <iostream>
#include <algorithm>


using std::cerr;
using std::cin;
using std::endl;


using namespace::SCIRun;


namespace Uintah {

const double EPS = 1e-6;

class FaceCuttingPlane : public Module {

public:
  FaceCuttingPlane( GuiContext* ctx);

  virtual ~FaceCuttingPlane();
  virtual void widget_moved(bool last);    
  virtual void execute();
  template <class MyField> void real_execute(MyField *lvf, ColorMapHandle cmh);
  void tcl_command( GuiArgs&, void* );
  void get_minmax(FieldHandle f);
//   bool get_dimensions(FieldHandle f, int& nx, int& ny, int& nz);
  template <class Mesh>  bool get_dimensions(Mesh*, int&, int&, int&);

private:
  FieldIPort             *infield_;
  ColorMapIPort          *icmap_;
  GeometryOPort          *ogeom_;
  CrowdMonitor            control_lock_; 
  PointWidget            *control_widget_;
  GeomID                  control_id_;
  GuiInt                  drawX_;
  GuiInt                  drawY_;
  GuiInt                  drawZ_;
  GuiInt                  need_find;
  GuiDouble               where;
  GuiString               face_name;
  GuiInt                  line_size;
  Point                   dmin_;
  Vector                  ddx_;
  Vector                  ddy_;
  Vector                  ddz_;
  MaterialHandle outcolor;
  double                  ddview_;
  pair<double, double> minmax_;
  int grid_id;
  string msg;
  Point control_;
  LatVolMesh *mesh_;
};


DECLARE_MAKER(FaceCuttingPlane)
static string control_name("Control Widget");
  
FaceCuttingPlane::FaceCuttingPlane(GuiContext* ctx) : 
  Module("FaceCuttingPlane", ctx, Filter, "Visualization", "Uintah"), 
  control_lock_("FaceCuttingPlane lock"),
  control_widget_(0),
  control_id_(-1),
  need_find(ctx->subVar("need_find")), 
  where(ctx->subVar("where")),
  face_name(ctx->subVar("face_name")), 
  line_size(ctx->subVar("line_size")),
  drawX_(ctx->subVar("drawX")),
  drawY_(ctx->subVar("drawY")),
  drawZ_(ctx->subVar("drawZ")),
  mesh_(0)
{
    need_find.set(1);
    
    outcolor=scinew Material(Color(0,0,0), Color(0,0,0), Color(0,0,0), 0);
    msg="";
}

FaceCuttingPlane::~FaceCuttingPlane()
{
}

void FaceCuttingPlane::get_minmax(FieldHandle f)
{
  double mn, mx;
  ScalarFieldInterfaceHandle sfi(f->query_scalar_interface());
  sfi->compute_min_max(mn, mx);
  minmax_ = pair<double, double>(mn, mx);
}

void FaceCuttingPlane::tcl_command(GuiArgs& args, void* userdata)
{
  if(args.count() < 2)
  {
    args.error("FaceCuttingPlane needs a minor command");
    return;
  }

  if(args[1] == "findxy")
  {
    need_find.set(1);
    want_to_execute();
  }
  else if(args[1] == "findyz")
  {
    need_find.set(2);
    want_to_execute();
  }
  else if(args[1] == "findxz")
  {
    need_find.set(3);
    want_to_execute();
  }
  else if (args[1] == "MoveWidget") {
    if (!control_widget_) return;
    Point w(control_widget_->ReferencePoint());  
    bool fire = false;
    if(args[2] == "plusx"){
      msg="plusx"; fire = true;
      w+=ddx_;
    } else if(args[2] == "minusx") {
      msg="minusx"; fire = true;
      w-=ddx_;
    } else if(args[2] == "plusy") {
      msg="plusy"; fire = true;
      w+=ddy_;
    } else if(args[2] == "minusy") {
      msg="minusy"; fire = true;
      w-=ddy_;
    } else if(args[2] == "plusz") {
      msg="plusz"; fire = true;
      w+=ddz_;
    } else if(args[2] == "minusz") {
      msg="minusz"; fire = true;
      w-=ddz_;
    }
    if( fire ) { 
      want_to_execute();
      control_ = w;
      control_widget_->SetPosition(w);
      widget_moved(true);
      ogeom_->flushViews();				  
    }
  } else {
    Module::tcl_command(args, userdata);
  }
}

void FaceCuttingPlane::widget_moved(bool last)
{
  if( control_widget_ && mesh_ && last && !abort_flag)
  {
    Transform t;
    mesh_->transform(t);
    control_ = t.unproject(control_widget_->ReferencePoint());
    abort_flag=1;
    want_to_execute();
  }
}


void FaceCuttingPlane::execute(void)
{

  infield_ = (FieldIPort *) get_iport("Scalar Field");
  icmap_ = (ColorMapIPort *)get_iport("ColorMap");
  ogeom_ = (GeometryOPort *)get_oport("Geometry");

  FieldHandle field;
  if (!infield_->get( field ))
    return;

  if(field->get_type_name(0) != "LatVolField"){
    error("This module only works with a LatVolField as input. No action!");
    return;
  }

  if(!field->is_scalar()){
    error("Not a scalar field:  No action!");
    return;
  }

  if (!icmap_) {
    error("Unable to initialize iport 'Color Map'. No action!");
    return;
  }
  if (!ogeom_) {
    error("Unable to initialize oport 'Geometry'. No action!");
    return;
  }
  
  ColorMapHandle cmap;
  if( !icmap_->get(cmap)){
    return;
  }

  
  if(!(mesh_ = dynamic_cast<LatVolMesh *>(field->mesh().get_rep()))){
    error("Mesh must be a LatVolMesh. No action!");
    return;
  }


  if(LatVolField<double> *lvf = 
     dynamic_cast<LatVolField<double> *>(field.get_rep())){
    real_execute( lvf, cmap );
  } else if( LatVolField<int> *lvf =  
	     dynamic_cast<LatVolField<int> *>(field.get_rep())){
    real_execute( lvf, cmap );
  } else if( LatVolField<float> *lvf =  
	     dynamic_cast<LatVolField<float> *>(field.get_rep())){
    real_execute( lvf, cmap );
  } else if( LatVolField<long> *lvf =  
	     dynamic_cast<LatVolField<long> *>(field.get_rep())){
    real_execute( lvf, cmap );
  } else {
    error("Unknown field type line 239. No action!");
  }
}  
template<class MyField> 
void 
FaceCuttingPlane::real_execute(MyField *lvf, ColorMapHandle cmap)
{

  int td;
  lvf->get_property("vartype", td);
  if( td != TypeDescription::SFCXVariable &&
      td != TypeDescription::SFCYVariable &&
      td != TypeDescription::SFCZVariable ){
    warning("Did not receive a Uintah FaceVariable. No action!");
    return;
  }


  int old_grid_id = grid_id;
  static int find = -1;
  int cmapmin, cmapmax;

  BBox b = mesh_->get_bounding_box();
  Vector diagv(b.diagonal());
  int nx, ny, nz, nf;
  get_dimensions(mesh_, nx, ny, nz);
//   cerr<<"nx, ny, nz = "<<nx<<", "<<ny<<", "<<nz<<endl;
  if(!control_widget_){
    control_widget_=scinew PointWidget(this, &control_lock_, 0.2);
    ddx_ = Vector(diagv.x()/(nx-1),0,0);
    ddy_ = Vector(0,diagv.y()/(ny-1),0);
    ddz_ = Vector(0,0,diagv.z()/(nz-1));
    ddview_ = (diagv.length()/(std::max(nx, std::max(ny,nz)) -1));
    control_widget_->SetPosition(b.min() + diagv * 0.5);
    control_widget_->SetScale(diagv.length()/80.0);
    control_ = b.min() + diagv * 0.5;
  }
  if( control_id_ == -1 ){
    GeomHandle w =control_widget_->GetWidget();
    control_id_ = ogeom_->addObj( w, control_name, &control_lock_);
  }

  cmapmin=(int)cmap->getMin();
  cmapmax=(int)cmap->getMax();
  
  if(td == TypeDescription::SFCXVariable){
    if(need_find.get() == 2)
      need_find.set(1); //or need_find.set(3);
    face_name.set(string("X faces"));
    gui->execute(id + " update_control");
  } else if(td == TypeDescription::SFCYVariable){
      if(need_find.get() == 3)
	need_find.set(1); //or need_find.set(2);
      face_name.set(string("Y faces"));
      gui->execute(id + " update_control");
  } else {
    if(need_find.get() == 1)
	need_find.set(2); //or need_find.set(3);
    face_name.set(string("Z faces"));
    gui->execute(id + " update_control");
  }
    
  int u_num, v_num;
  Point corner(b.min());
  nf = need_find.get();
  Vector u,v;
  bool horiz = false; // false if vert, true if horiz.

  if(td == TypeDescription::SFCXVariable){
    u_num = nx;
    u = ddx_;
    if(nf == 1){ 
      v_num = ny;
      corner.z(control_.z());
      v = ddy_ * ((ny-1.0)/ny);
    } else {
      v_num = nz;
      corner.y(control_.y());
      v = ddz_ * ((ny-1.0)/nz);
    }
    face_name.set("X faces");
  } else if (td == TypeDescription::SFCYVariable){
    if( nf == 1 ){
      u_num = nx;
      v_num = ny;
      u = ddx_ * ((nx-1.0)/nx); v = ddy_;
      corner.z(control_.z());
    } else {
      u_num = ny;
      v_num = nz;
      u = ddy_; v = ddz_ * ((nz-1.0)/nz);
      corner.x(control_.x());
    }
    horiz = true;
    face_name.set("Y faces");
  } else {
    v_num = nz;
    v = ddz_;
    if( nf == 2 ){
      u_num = ny;
      corner.x(control_.x());
      u = ddy_ * ((ny-1.0)/ny);
    } else {
      u_num = nx;
      horiz = true;
      corner.y(control_.y());
      u = ddx_ * ((nx-1.0)/nx);
      }
    face_name.set("Z faces");
  }
  GeomGroup *faces = scinew GeomGroup();

  double sval;
  MaterialHandle matl;
  LatVolMesh::NodeIndex node;
  int i, j, u_, v_;
  u_ = u_num; v_ = v_num;

  for (i = 0; i < u_; i++){
    for (j = 0; j < v_; j++) {
      Point p0, p1;
      if ( horiz ) {
	p0 = corner + u * (i + EPS) +
	  v * j;
	p1 = corner + u * (i+1) +
	  v * j;
      } else {
	p0 = corner + u * i +
	  v * (j + EPS);
	p1 = corner + u * i +
	  v * (j+1);
      }
      GeomLine *line = 0;
      if( mesh_->locate(node, p0 + (p1 - p0) * 0.5)){
	sval = lvf->fdata()[node];
 	matl = cmap->lookup( sval);
	line = new GeomLine(p0,p1);
	float linesz = line_size.get();
	line->setLineWidth((float)line_size.get());
	faces->add( scinew GeomMaterial( line, matl));
      } else {
	matl = outcolor;
	sval = 0;
      }
    }
  }
  
  ogeom_->flushViews();				  
  
  
  // delete the old grid/cutting plane
  if (old_grid_id != 0) {
    ogeom_->delObj( old_grid_id );
  }
  grid_id = ogeom_->addObj(faces,  "Face Cutting Plane");
  old_grid_id = grid_id;
}

template <class Mesh>
bool 
FaceCuttingPlane::get_dimensions(Mesh*, int&, int&, int&)
  {
    return false;
  }

template<> 
bool FaceCuttingPlane::get_dimensions(LatVolMesh* m,
				 int& nx, int& ny, int& nz)
  {
    nx = m->get_ni();
    ny = m->get_nj();
    nz = m->get_nk();
    return true;
  }



} // End namespace Uintah


