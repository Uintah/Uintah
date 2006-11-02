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

#include "FaceCuttingPlane.h"
#include <Core/Datatypes/FieldInterface.h>
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
#include <Dataflow/Network/Ports/ColorMapPort.h>
#include <Dataflow/Network/Ports/GeometryPort.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Dataflow/Widgets/PointWidget.h>
#include <Core/Util/NotFinished.h>
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
  virtual void widget_moved(bool last, BaseWidget*);
  virtual void execute();
  void tcl_command( GuiArgs&, void* );
  void get_minmax(FieldHandle f);

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
  Transform*              trans_;
 MaterialHandle outcolor;
  double                  ddview_;
  pair<double, double> minmax_;
  int grid_id;
  string msg;
  Point control_;
};
} // End namespace Uintah

using namespace Uintah;

DECLARE_MAKER(FaceCuttingPlane)
static string control_name("Control Widget");
  
FaceCuttingPlane::FaceCuttingPlane(GuiContext* ctx) : 
  Module("FaceCuttingPlane", ctx, Filter, "Visualization", "Uintah"), 
  control_lock_("FaceCuttingPlane lock"),
  control_widget_(0),
  control_id_(-1),
  drawX_(get_ctx()->subVar("drawX")),
  drawY_(get_ctx()->subVar("drawY")),
  drawZ_(get_ctx()->subVar("drawZ")),
  need_find(get_ctx()->subVar("need_find")), 
  where(get_ctx()->subVar("where")),
  face_name(get_ctx()->subVar("face_name")), 
  line_size(get_ctx()->subVar("line_size")),
  trans_(0),
  grid_id(0)
{
    need_find.set(1);
    
    outcolor=scinew Material(Color(0,0,0), Color(0,0,0), Color(0,0,0), 0);
    msg="";
}

FaceCuttingPlane::~FaceCuttingPlane()
{
  if( trans_ != 0 ) delete trans_;
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
      widget_moved(true,0);
      ogeom_->flushViews();				  
    }
  } else {
    Module::tcl_command(args, userdata);
  }
}

void FaceCuttingPlane::widget_moved(bool last, BaseWidget*)
{
  if( control_widget_ && trans_ && last && !abort_flag_)
  {
    control_ = trans_->unproject(control_widget_->ReferencePoint());
    abort_flag_=1;
    want_to_execute();
  }
}


void FaceCuttingPlane::execute(void)
{

  infield_ = (FieldIPort *) get_iport("Scalar Field");
  icmap_ = (ColorMapIPort *)get_iport("ColorMap");
  ogeom_ = (GeometryOPort *)get_oport("Geometry");
  
  FieldHandle field;
  if (!(infield_->get( field ) && field.get_rep())) {
    warning("No data on input pipe.");
    return;
  } else if( !field->query_scalar_interface(this).get_rep() ){
    error("Input is not a Scalar field.");
    return;
  }

  int td;
  field->get_property("vartype", td);
  if( td != TypeDescription::SFCXVariable &&
      td != TypeDescription::SFCYVariable &&
      td != TypeDescription::SFCZVariable ){

    warning("Did not receive a Uintah FaceVariable. No action!");
    return;
  }

  MeshHandle mh = field->mesh();

  const SCIRun::TypeDescription *ftd = field->get_type_description();
  if( ftd->get_name().find("LatVolMesh") == string::npos ){
    error("This module only works with a LatVolMesh based field as input. No action!");
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

  CompileInfoHandle ci = FaceCuttingPlaneAlgo::get_compile_info(ftd);
  Handle<FaceCuttingPlaneAlgo> algo;
  if( !module_dynamic_compile(ci, algo) ){
    error("dynamic compile failed.");
    return;
  }
  
  
  if( trans_ == 0 ) trans_ = scinew Transform();
  algo->set_transform( field, *trans_ );

  int nx, ny, nz;
  BBox b;

  if( !algo->get_bounds_and_dimensions(field, nx, ny, nz, b) ){
    error("Something wrong with the mesh type");
    return;
  }
  Vector diagv(b.diagonal());

  int old_grid_id = grid_id;
  int cmapmin, cmapmax;
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
    get_gui()->execute(get_id() + " update_control");
  } else if(td == TypeDescription::SFCYVariable){
      if(need_find.get() == 3)
	need_find.set(1); //or need_find.set(2);
      face_name.set(string("Y faces"));
      get_gui()->execute(get_id() + " update_control");
  } else {
    if(need_find.get() == 1)
	need_find.set(2); //or need_find.set(3);
    face_name.set(string("Z faces"));
    get_gui()->execute(get_id() + " update_control");
  }
  int u_num, v_num;
  Point corner(b.min());
  int nf = need_find.get();
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
      if(algo->get_value(field, p0 + (p1 - p0) * 0.5, sval)) {
 	matl = cmap->lookup( sval);
	line = new GeomLine(p0,p1);
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

CompileInfoHandle
FaceCuttingPlaneAlgo::get_compile_info(const SCIRun::TypeDescription *ftd)
{
  // use cc_to_h if this is in the .cc file, otherwise just __FILE__
  static const string include_path(SCIRun::TypeDescription::cc_to_h(__FILE__));
  static const string template_class_name("FaceCuttingPlaneAlgoT");
  static const string base_class_name("FaceCuttingPlaneAlgo");

  CompileInfo *rval = 
    scinew CompileInfo(template_class_name + "." +
		       ftd->get_filename() + ".",
                       base_class_name, 
                       template_class_name, 
                       ftd->get_name());

  // Add in the include path to compile this obj
  rval->add_include(include_path);
  // add namespace
  rval->add_namespace("Uintah");
  ftd->fill_compile_info(rval);
  return rval;
}



