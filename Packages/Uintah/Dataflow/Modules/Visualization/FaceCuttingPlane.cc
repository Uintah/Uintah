//static char *id="@(#) $Id$";

/*
 *  CuttingPlane.cc:  
 *
 *  Written by:
 *   Kurt Zimmerman
 *   Department of Computer Science
 *   University of Utah
 *   August 2002
 *
 *  Copyright (C) 2002 SCI Group
 */

#include <Core/Datatypes/FieldInterface.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/FieldPort.h>
#include <Core/Datatypes/Field.h>
#include <Dataflow/Ports/ColorMapPort.h>
#include <Core/Geom/GeomGrid.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/GeomLine.h>
#include <Core/Geom/Material.h>
#include <Core/Geom/GeomTransform.h>
#include <Core/Geometry/Point.h>
#include <Core/Math/MinMax.h>
#include <Core/Math/MiscMath.h>
#include <Core/Malloc/Allocator.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Thread/CrowdMonitor.h>
#include <Dataflow/Widgets/FrameWidget.h>
#include <Core/Datatypes/LatVolField.h>
#include <iostream>
using std::cerr;
using std::cin;
using std::endl;

#define CP_PLANE 0
#define CP_SURFACE 1
#define CP_CONTOUR 2

using namespace SCIRun;

namespace Uintah {

#define EPS 1.e-6

/**************************************
CLASS
   FaceCuttingPlane
       FaceCuttingPlane interpolates a planar slice through 
       face centered data.

GENERAL INFORMATION

   FaceCuttingPlane
  
   Author:  Kurt Zimmerman
            SCI Institute
            University of Utah
   Date:    August 2002
   
   C-SAFE
   
   Copyright <C> 2002 SCI Group

KEYWORDS
   Visualization, Widget, GenColormap

DESCRIPTION
   FaceCuttingPlane interpolates a planar slice through 
   face centered data and maps data values to colors on a
   semitransparent surface.  The plane can be manipulated with a
   3D widget to allow the user to look at different
   cross sections of the data.

WARNING
   None

****************************************/

class FaceCuttingPlane : public Module {
  FieldIPort *infield;
  ColorMapIPort *inColorMap;
  GeometryOPort* ogeom;
  CrowdMonitor widget_lock;
  bool init;
  int widget_id;
  FrameWidget *widget;
  virtual void widget_moved(bool last);
  GuiInt need_find;
  MaterialHandle outcolor;
  GuiDouble where;
  GuiString face_name;
  GuiInt line_size;
  int grid_id;
  string msg;
public:
 
  // GROUP:  Constructors:
  ///////////////////////////
  //
  // Constructs an instance of class FaceCuttingPlane
  //
  // Constructor taking
  //    [in] id as an identifier
  //
  FaceCuttingPlane(GuiContext* ctx);
  
  // GROUP:  Destructor:
  ///////////////////////////
  // Destructor
  virtual ~FaceCuttingPlane();
  
  // GROUP:  Access functions:
  ///////////////////////////
  //
  // execute() - execution scheduled by scheduler
  virtual void execute();
  
  void get_minmax(FieldHandle f);
  
  bool get_dimensions(FieldHandle f, int& nx, int& ny, int& nz);
  template <class Mesh>  bool get_dimensions(Mesh, int&, int&, int&);
  //////////////////////////
  //
  // tcl_commands - overides tcl_command in base class Module, takes:
  //                                  findxy, findyz, findxz, 
  //                                  plusx, minusx, plusy, minusy, 
  //                                  plusz, minusz,
  //                                  connectivity
  virtual void tcl_command(GuiArgs&, void*);
private:
  Point iPoint_;
  double result_;
  Vector gradient_;
  bool success_;
  IntVector dim_;
  pair<double, double> minmax_;
};

DECLARE_MAKER(FaceCuttingPlane)
  
static string widget_name("FaceCuttingPlane Widget");

FaceCuttingPlane::FaceCuttingPlane(GuiContext* ctx) :
  Module("FaceCuttingPlane", ctx, Filter, "Visualization", "Uintah"),
  widget_lock("Cutting plane widget lock"),
  init(false),
  widget_id(0),
  need_find(ctx->subVar("need_find")), 
  where(ctx->subVar("where")),
  face_name(ctx->subVar("face_name")), 
  line_size(ctx->subVar("line_size")),
  grid_id(0)
{
    float INIT(.1);

    widget = scinew FrameWidget(this, &widget_lock, INIT);

    need_find.set(1);
    
    outcolor=scinew Material(Color(0,0,0), Color(0,0,0), Color(0,0,0), 0);
    msg="";
}

FaceCuttingPlane::~FaceCuttingPlane()
{
}

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1184
#endif

void FaceCuttingPlane::get_minmax(FieldHandle f)
{
  double mn, mx;
  ScalarFieldInterface* sfi = f->query_scalar_interface();
  sfi->compute_min_max(mn, mx);
  minmax_ = pair<double, double>(mn, mx);
}

void FaceCuttingPlane::execute()
{
  int old_grid_id = grid_id;
  static int find = -1;
  int cmapmin, cmapmax;

  // Create the input ports
  // Need a scalar field and a ColorMap
  infield = (FieldIPort *) get_iport("Scalar Field");
  inColorMap = (ColorMapIPort *) get_iport("ColorMap");
  // Create the output port
  ogeom = (GeometryOPort *) get_oport("Geometry");

  // get the scalar field and ColorMap...if you can
  FieldHandle field;
  if (!infield->get( field ))
    return;

  if(!field->is_scalar()){
    cerr<<"Not a scalar field\n ";
    return;
  }
      
  ColorMapHandle cmap;
  if (!inColorMap->get( cmap ))
    return;

  int td;
  field->get_property("vartype", td);
  if( td != TypeDescription::SFCXVariable &&
      td != TypeDescription::SFCYVariable &&
      td != TypeDescription::SFCZVariable ){
    warning("Did not receive a Uintah FaceVariable, no action.");
    return;
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

  if ( !init || need_find.get() != find) 
  {
    init = true;
    GeomObj *w = widget->GetWidget() ;
    widget_id = ogeom->addObj( w, widget_name, &widget_lock );
    widget->Connect( ogeom );
    widget->SetRatioR( 0.4 );
    widget->SetRatioD( 0.4 );
  } else if (need_find.get() != find){
    if (widget_id != 0) {
      ogeom->delObj( widget_id );
    }
    GeomObj *w = widget->GetWidget() ;
    widget_id = ogeom->addObj( w, widget_name, &widget_lock );
  }

  if (need_find.get() != find)
  {
    BBox box;
    Point min, max;
    box = field->mesh()->get_bounding_box();
    min = box.min(); max = box.max();
    Point center = min + (max-min)/2.0;
    double max_scale;
    double wh=where.get();
    if (need_find.get() == 1)
    {   // Find the field and put in optimal place
      // in xy plane with reasonable frame thickness
      center.z(min.z()*wh+max.z()*(1-wh));
      Point right( max.x(), center.y(), center.z());
      Point down( center.x(), min.y(), center.z());
      max_scale = Max( (max.x() - min.x()), (max.y() - min.y()) );
      widget->SetScale( max_scale/30. );
      widget->SetPosition( center, right, down);
    }
    else if (need_find.get() == 2)
    {   // Find the field and put in optimal place
      // in yz plane with reasonable frame thickness
      center.x(min.x()*wh+max.x()*(1-wh));
      Point right( center.x(), center.y(), max.z());
      Point down( center.x(), min.y(), center.z());	    
      max_scale = Max( (max.z() - min.z()), (max.y() - min.y()) );
      widget->SetScale( max_scale/30. );
      widget->SetPosition( center, right, down);
    }
    else
    {   // Find the field and put in optimal place
      // in xz plane with reasonable frame thickness
      center.y(min.y()*wh+max.y()*(1-wh));
      Point right( max.x(), center.y(), center.z());
      Point down( center.x(), center.y(), min.z());	    
      max_scale = Max( (max.x() - min.x()), (max.z() - min.z()) );
      widget->SetScale( max_scale/30. );
      widget->SetPosition( center, right, down);
    }
    find = need_find.get();
  }

  // advance or decrement along x, y, or z
  if (msg != "" &&
      (field->get_type_name(0) == "LatVolField")){
    int nx, ny, nz;
    get_dimensions(field, nx,ny,nz);
    if(field->data_at() == Field::CELL){ nx--; ny--; nz--; }
    Point center, right, down;
    widget->GetPosition(center, right, down);
    BBox box;
    Point min, max;
    Vector diag;
    box = field->mesh()->get_bounding_box();
    min = box.min(); max = box.max();
    diag=max-min;
    diag.x(diag.x()/(nx-1));
    diag.y(diag.y()/(ny-1));
    diag.z(diag.z()/(nz-1));
    if (msg=="plusx") {
      center.x(center.x()+diag.x());
      right.x(right.x()+diag.x());
      down.x(down.x()+diag.x());
    } else if (msg=="minusx") {
      center.x(center.x()-diag.x());
      right.x(right.x()-diag.x());
      down.x(down.x()-diag.x());
    } else if (msg=="plusy") {
      center.y(center.y()+diag.y());
      right.y(right.y()+diag.y());
      down.y(down.y()+diag.y());
    } else if (msg=="minusy") {
      center.y(center.y()-diag.y());
      right.y(right.y()-diag.y());
      down.y(down.y()-diag.y());
    } else if (msg=="plusz") {
      center.z(center.z()+diag.z());
      right.z(right.z()+diag.z());
      down.z(down.z()+diag.z());
    } else { // if (msg=="minusz")
      center.z(center.z()-diag.z());
      right.z(right.z()-diag.z());
      down.z(down.z()-diag.z());
    }
    widget->SetPosition( center, right, down);
    msg="";
  }

  // get the position of the frame widget
  Point 	corner, center, R, D;
  widget->GetPosition( center, R, D);
  Vector v1 = R - center,
         v2 = D - center;
         
  // calculate the corner and the
  // u and v vectors of the cutting plane
    
  corner = (center - v1) + v2;
  Vector u = v1 * 2.0,
         v = -v2 * 2.0;

    
  // create the grid for the cutting plane
//   double u_fac = widget->GetRatioR(),
//          v_fac = widget->GetRatioD();
  
//   int u_num = (int) (u_fac * 500),
//       v_num = (int) (v_fac * 500);

//   int u_mult = 0,
//       v_mult = 0;
  int u_num, v_num;
  int nx, ny, nz, nf;
  nf = need_find.get();
  get_dimensions(field, nx,ny,nz);
  bool horiz = false; // false if vert, true if horiz.
  if(td == TypeDescription::SFCXVariable){
    u_num = nx;
    if(nf == 1) v_num = ny; else v_num = nz;
    face_name.set("X faces");
  } else if (td == TypeDescription::SFCYVariable){
    if( nf == 1 ){
      u_num = nx; v_num = ny;
    } else {
      u_num = ny; v_num = nz;
    }
      horiz = true;
      face_name.set("Y faces");
  } else {
    v_num = nz;
    if( nf == 2 ){
      u_num = ny;
    } else {
      u_num = nx;
      horiz = true;
    }
      face_name.set("Z faces");
  }

  // Get the scalar values and corresponding
  // colors to put in the cutting plane

  // this wierdness in setting GeomGrid size pays off when filling the grid
//   const int grid_mult = 5;
//   GeomGrid* grid = new GeomGrid( u_num + (u_num * u_mult * (grid_mult-1)),
// 				 v_num + (v_num * v_mult * (grid_mult-1))
// 				 , corner, u, v, 0);
//   GeomGrid* grid = new GeomGrid( u_num, v_num , corner, u, v, 0);
//  TexGeomLines* faces = new TexGeomLines();
  GeomGroup *faces = scinew GeomGroup();
  //int ix = 0;
  int i, j;

  ScalarFieldInterface *sfi = 0;
  if( field->get_type_name(0) == "LatVolField")
    sfi = field->query_scalar_interface();
	
//   int i_inc = Max(grid_mult * u_mult, 1);
//   int j_inc = Max(grid_mult * v_mult, 1);

  
  int u_ = u_num, 
      v_ = v_num;
//   if(horiz) v_++;
//   else u_++;
  for (i = 0; i < u_; i++){
    for (j = 0; j < v_; j++) {
      Point p0, p1;
      
      if ( horiz ) {
	if( j == v_ -1 && i == u_ - 1){
	  p0 = corner + u * ((double) (i-0.00001)/(u_num-1)) + 
	    v * ((double) (j-0.00001)/(v_num-1));
	  p1 = corner + u * ((double) (i+1-0.00001)/(u_num-1)) + 
	    v * ((double) (j-0.00001)/(v_num-1));
	} else 
	if( j == v_ - 1  ){
	  p0 = corner + u * ((double) i/(u_num-1)) + 
	    v * ((double) (j-0.00001)/(v_num-1));
	  p1 = corner + u * ((double)(i+1)/(u_num-1)) + 
	    v * ((double) (j-0.00001)/(v_num-1));
	} else {
	  p0 = corner + u * ((double) i/(u_num-1)) + 
	    v * ((double) j/(v_num-1));
	  p1 = corner + u * ((double)(i+1)/(u_num-1)) + 
	    v * ((double) j/(v_num-1));
	}
      } else {
	if( j == v_ -1 && i == u_ - 1){
	  p0 = corner + u * ((double) (i-0.00001)/(u_num-1)) + 
	    v * ((double) (j-0.00001)/(v_num-1));
	  p1 = corner + u * ((double) (i-0.00001)/(u_num-1)) + 
	    v * ((double)(j+1-0.00001)/(v_num-1));
	} else 
	if( i == u_ - 1  ){
	  p0 = corner + u * ((double) (i-0.00001)/(u_num-1)) + 
	    v * ((double) j/(v_num-1));
	  p1 = corner + u * ((double) (i-0.00001)/(u_num-1)) + 
	    v * ((double)(j+1)/(v_num-1));
	} else {
	  p0 = corner + u * ((double) i/(u_num-1)) + 
	    v * ((double) j/(v_num-1));
	  p1 = corner + u * ((double) i/(u_num-1)) + 
	    v * ((double)(j+1)/(v_num-1));
	}
      }
      
      GeomLine *line = new GeomLine(p0,p1);
      line->setLineWidth(line_size.get());
      double sval;
	    
      // get the color from cmap for p 	    
      MaterialHandle matl;
      if( sfi->interpolate( sval, p0)){
	matl = cmap->lookup( sval);
      }
      else {
	cerr<<"p0 = "<<p0<<" which is out of bounds\n";
	matl = outcolor;
	sval = 0;
      }
//       grid->set(i, j, 0, matl );
      faces->add( scinew GeomMaterial( line, matl));
    }
  }
  // delete the old grid/cutting plane
  if (old_grid_id != 0) {
    ogeom->delObj( old_grid_id );
  }
//   Transform t; t.post_translate( Vector(xt.get(), yt.get(), zt.get()));
//   GeomTransform *gt =
//     scinew GeomTransform( grid, t);
//   grid_id = ogeom->addObj(gt, "Face Cutting Plane");
  grid_id = ogeom->addObj(faces,  "Face Cutting Plane");
  old_grid_id = grid_id;
}


void FaceCuttingPlane::widget_moved(bool last)
{
    if(last && !abort_flag)
    {
	abort_flag=1;
	want_to_execute();
    }
}


void FaceCuttingPlane::tcl_command(GuiArgs& args, void* userdata)
{
    if(args.count() < 2)
    {
	args.error("Streamline needs a minor command");
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
    else if(args[1] == "plusx")
    {
	msg="plusx";
	want_to_execute();
    }
    else if(args[1] == "minusx")
    {
	msg="minusx";
	want_to_execute();
    }
    else if(args[1] == "plusy")
    {
	msg="plusy";
	want_to_execute();
    }
    else if(args[1] == "minusy")
    {
	msg="minusy";
	want_to_execute();
    }
    else if(args[1] == "plusz")
    {
	msg="plusz";
	want_to_execute();
    }
    else if(args[1] == "minusz")
    {
	msg="minusz";
	want_to_execute();
    }
    else if(args[1] == "connectivity")
    {
	
    }
    else
    {
	Module::tcl_command(args, userdata);
    }
}

template <class Mesh>
bool 
FaceCuttingPlane::get_dimensions(Mesh, int&, int&, int&)
  {
    return false;
  }

template<> 
bool FaceCuttingPlane::get_dimensions(LatVolMeshHandle m,
				 int& nx, int& ny, int& nz)
  {
    nx = m->get_nx();
    ny = m->get_ny();
    nz = m->get_nz();
    return true;
  }

bool
FaceCuttingPlane::get_dimensions(FieldHandle texfld_,  int& nx, int& ny, int& nz)
{
  const string type = texfld_->get_type_name(1);
  if(texfld_->get_type_name(0) == "LatVolField"){
    LatVolMeshHandle mesh_;
    if (type == "double") {
      LatVolField<double> *fld =
	dynamic_cast<LatVolField<double>*>(texfld_.get_rep());
      mesh_ = fld->get_typed_mesh();
    } else if (type == "float") {
      LatVolField<float> *fld =
	dynamic_cast<LatVolField<float>*>(texfld_.get_rep());
      mesh_ = fld->get_typed_mesh();
    } else if (type == "long") {
      LatVolField<long> *fld =
	dynamic_cast<LatVolField<long>*>(texfld_.get_rep());
      mesh_ = fld->get_typed_mesh();
    } else if (type == "int") {
      LatVolField<int> *fld =
	dynamic_cast<LatVolField<int>*>(texfld_.get_rep());
      mesh_ = fld->get_typed_mesh();
    } else {
      cerr << "FaceCuttingPlane error - unknown LatVolField type: " << type << endl;
      return false;
    }
    return get_dimensions( mesh_, nx, ny, nz );
  } else {
    return false;
  }
}




} // End namespace Uintah
