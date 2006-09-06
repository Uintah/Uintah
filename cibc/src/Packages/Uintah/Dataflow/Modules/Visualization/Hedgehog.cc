//static char *id="@(#) $Id$";

/*
 *  Hedgehog.cc:  
 *
 *  Written by:
 *   Steven G. Parker
 *   Department of Computer Science
 *   University of Utah
 *   June 1995
 *
 *  Copyright (C) 1995 SCI Group
 */

#include <Core/Containers/Array1.h>
#include <Core/Util/NotFinished.h>
#include <Dataflow/Network/Module.h>
#include <Dataflow/Network/Ports/ColorMapPort.h>
#include <Dataflow/Network/Ports/GeometryPort.h>
#include <Dataflow/Network/Ports/FieldPort.h>
#include <Core/Geom/GeomArrows.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/GeomLine.h>
#include <Core/Geom/Material.h>
#include <Core/Geometry/Point.h>
#include <Core/Math/MinMax.h>
#include <Core/Malloc/Allocator.h>
#include <Dataflow/GuiInterface/GuiVar.h>
#include <Core/Thread/CrowdMonitor.h>
#include <Dataflow/Widgets/BoxWidget.h>
#include <Dataflow/Widgets/FrameWidget.h>
#include <Core/Basis/Constant.h>
#include <Core/Basis/HexTrilinearLgn.h>
#include <Core/Datatypes/LatVolMesh.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Containers/FData.h>
#include <Core/Datatypes/FieldInterface.h>

#include <iostream>
using std::cerr;

#define CP_PLANE 0
#define CP_SURFACE 1
#define CP_CONTOUR 2


namespace Uintah {
/**************************************
CLASS
   Hedgehog
      Hedgehog produces arrows at sample points in a vector field.


GENERAL INFORMATION

   Hedgehog
  
   Author:  Steven G. Parker (sparker@cs.utah.edu)
            
            Department of Computer Science
            
            University of Utah
   
   Date:    June 1995
   
   C-SAFE
   
   Copyright <C> 1995 SCI Group

KEYWORDS
   Visualization, vector_field, GenColormap

DESCRIPTION
   Hedgehog produces arrows at sample points in a vector
   field.  The length of the arrow indicates the magnitude of the
   field at that point, and the orientation indicates the
   direction of the field.  In addition, the shaft of the arrow
   is mapped to a scalar field value using a colormap produced by
   GenColormap.

WARNING
   None



****************************************/

using namespace SCIRun;


class Hedgehog : public Module {
  FieldIPort *invectorfield;
  FieldIPort* inscalarfield;
  ColorMapIPort *inColorMap;
  GeometryOPort* ogeom;
  CrowdMonitor widget_lock;
  int init;
  int widget_id;
  BoxWidget* widget3d;
  FrameWidget *widget2d;
  
  // GROUP:  Widgets:
  //////////////////////
  //
  // widget_moved -  
  virtual void widget_moved(bool last, BaseWidget*);
  GuiDouble length_scale;
  GuiDouble width_scale;
  GuiDouble head_length;
  GuiString type;
  GuiInt exhaustive_flag;
  GuiInt vector_default_color;
  GuiInt drawcylinders;
  GuiDouble shaft_rad;
  MaterialHandle outcolor;
  MaterialHandle shaft;
  MaterialHandle head;
  MaterialHandle back;
  
  int grid_id;
  int need_find2d;
  int need_find3d;

  Point iPoint_;
  IntVector dim_;
  
public:
  typedef LatVolMesh<HexTrilinearLgn<Point> > LVMesh;
  typedef HexTrilinearLgn<Vector>             FDVectorBasis;
  typedef ConstantBasis<Vector>               CFDVectorBasis;
  typedef GenericField<LVMesh, CFDVectorBasis, FData3d<Vector, LVMesh> > CVField;
  typedef GenericField<LVMesh, FDVectorBasis,  FData3d<Vector, LVMesh> > LVField;

  // GROUP:  Constructors:
  ///////////////////////////
  //
  // Constructs an instance of class Hedgehog
  //
  // Constructor taking
  //    [in] id as an identifier
  //
  Hedgehog(GuiContext* ctx);
       
        // GROUP:  Destructor:
        ///////////////////////////
        // Destructor
  virtual ~Hedgehog();
  
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
  virtual void tcl_command(GuiArgs&, void*);

//  bool get_dimensions(FieldHandle f, int& nx, int& ny, int& nz);
//  template <class Mesh>  bool get_dimensions(Mesh, int&, int&, int&);

};

static string module_name("Hedgehog");
static string widget_name("Hedgehog Widget");

  DECLARE_MAKER(Hedgehog)

Hedgehog::Hedgehog(GuiContext* ctx)
: Module("Hedgehog", ctx, Filter, "Visualization", "Uintah"),
  widget_lock("Hedgehog widget lock"),
  length_scale(get_ctx()->subVar("length_scale")),
  width_scale(get_ctx()->subVar("width_scale")),
  head_length(get_ctx()->subVar("head_length")),
  type(get_ctx()->subVar("type")),
  exhaustive_flag(get_ctx()->subVar("exhaustive_flag")),
  vector_default_color(get_ctx()->subVar("vector_default_color")),
  drawcylinders(get_ctx()->subVar("drawcylinders")),
  shaft_rad(get_ctx()->subVar("shaft_rad")),
  shaft(new Material(Color(0,0,0), Color(.6, .6, .6),
		     Color(.6, .6, .6), 10)),
  head(new Material(Color(0,0,0), Color(1,1,1), Color(.6, .6, .6), 10)),
  back(new Material(Color(0,0,0), Color(.6, .6, .6), Color(.6, .6, .6), 10)),
  iPoint_(Point(0,0,0)), dim_(IntVector(1,1,1))
{
    // Create the input ports
    // Need a scalar field and a ColorMap
    init = 1;
    float INIT(.1);

    widget2d = scinew FrameWidget(this, &widget_lock, INIT, true);
    widget3d = scinew BoxWidget(this, &widget_lock, INIT, false, true);
    grid_id=0;

    need_find2d=1;
    need_find3d=1;

    drawcylinders.set(0);
    outcolor=scinew Material(Color(0,0,0), Color(0,0,0), Color(0,0,0), 0);
}

Hedgehog::~Hedgehog()
{
}

void Hedgehog::execute()
{
  int old_grid_id = grid_id;
    // Create the input port
    invectorfield = (FieldIPort *) get_iport("Vector Field");
    inscalarfield = (FieldIPort *) get_iport("Scalar Field");
    inColorMap = (ColorMapIPort *) get_iport("ColorMap");
					
    // Create the output port
    ogeom = (GeometryOPort *) get_oport("Geometry"); 

  // get the scalar field and ColorMap...if you can
  FieldHandle vfield;
  if (!invectorfield->get( vfield )){
    warning("NodeHedgehog::execute() - No data from input vector port.");
    return;
  }

  const TypeDescription *td = vfield->get_type_description(Field::MESH_TD_E);
  if(td->get_name().find("LatVolMesh") == string::npos ) {
    error("First field is not a LatVolMesh based field.");
    return;
  }

  VectorFieldInterfaceHandle vfi = 0;  

  LVMesh *mesh;
  int nx, ny, nz;

  CVField* cfld = dynamic_cast<CVField *>(vfield.get_rep());
  if( !cfld ){
    LVField* lfld = dynamic_cast<LVField *>(vfield.get_rep());
    if(!lfld){
      error("Incoming field does not use a LatVolMesh");
      return;
    } else {
      vfi = lfld->query_vector_interface(this).get_rep() ;

      mesh = lfld->get_typed_mesh().get_rep();
      // use the mesh for the geometry
      nx = mesh->get_ni();
      ny = mesh->get_nj();
      nz = mesh->get_nk();
    }
  } else {
    vfi = cfld->query_vector_interface(this).get_rep() ;
    mesh = cfld->get_typed_mesh().get_rep();
    // use the mesh for the geometry
    // CC data needs to be computed on a mesh 1 size smaller
    nx = mesh->get_ni()-1;
    ny = mesh->get_nj()-1;
    nz = mesh->get_nk()-1;
  }

  if( vfi == 0 ){
    error("First field must be a Vector field.");
    return;
  }

  FieldHandle ssfield;
  int have_sfield=inscalarfield->get( ssfield );
  ScalarFieldInterfaceHandle sfi = 0;
  if( have_sfield ){
    sfi = ssfield->query_scalar_interface(this).get_rep();
    if( sfi == 0 ){
      warning("Second field is not a scalar field.  No Colormapping.");
      have_sfield = 0;
    }
  }
  
  ColorMapHandle cmap;
  int have_cmap=inColorMap->get( cmap );
  if(!have_cmap){
    have_sfield=0;
    if(vector_default_color.get() == 0){
      *(shaft.get_rep()) = Material(Color(0,0,0), Color(.8, .8, .8),
				    Color(.6, .6, .6), 10);
      *(head.get_rep()) = Material(Color(0,0,0), Color(1,1,1),
				   Color(.6, .6, .6), 10);
      *(back.get_rep()) = Material(Color(0,0,0), Color(.8, .8, .8),
				   Color(.6, .6, .6), 10);
    } else if (vector_default_color.get() == 1) {
      *(shaft.get_rep()) = Material(Color(0,0,0), Color(.4, .4, .4),
				    Color(.6, .6, .6), 10);
      *(head.get_rep()) = Material(Color(0,0,0), Color(.4,.4,.4),
				   Color(.6, .6, .6), 10);
      *(back.get_rep()) = Material(Color(0,0,0), Color(.4, .4, .4),
				   Color(.6, .6, .6), 10);
    } else {
      *(shaft.get_rep()) = Material(Color(0,0,0), Color(.1, .1, .1),
				    Color(.6, .6, .6), 10);
      *(head.get_rep()) = Material(Color(0,0,0), Color(.1,.1,.1),
				   Color(.6, .6, .6), 10);
      *(back.get_rep()) = Material(Color(0,0,0), Color(.1, .1, .1),
				   Color(.6, .6, .6), 10);
    }
  }

  if (init == 1) 
  {
    init = 0;
    GeomHandle w2d = widget2d->GetWidget() ;
    GeomHandle w3d = widget3d->GetWidget() ;
    GeomHandle w = new GeomGroup;
    ((GeomGroup*)w.get_rep())->add(w2d);
    ((GeomGroup*)w.get_rep())->add(w3d);
    widget_id = ogeom->addObj( w, widget_name, &widget_lock );

    widget2d->Connect( ogeom );
    widget2d->SetRatioR( 0.2 );
    widget2d->SetRatioD( 0.2 );

    widget3d->Connect( ogeom );
    widget3d->SetRatioR( 0.2 );
    widget3d->SetRatioD( 0.2 );
    widget3d->SetRatioI( 0.2 );
  }
  int do_3d=1;
  if(type.get() == "2D")
    do_3d=0;

  widget2d->SetState(!do_3d);
  widget3d->SetState(do_3d);


  // if field size or location change we need to update the widgets
  BBox box;  Point min, max;
  box = mesh->get_bounding_box();
  min = box.min(); max = box.max();
  if( iPoint_ != min ){
    iPoint_ = min;
    need_find3d = 1;
    need_find2d = 1;
  } else if (   dim_ != IntVector(nx, ny, nz) ){
    dim_ = IntVector(nx, ny, nz);
    need_find3d = 1;
    need_find2d = 1;
  }


  if (do_3d){
    if(need_find3d != 0){
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
    
  // get the position of the frame widget
  Point 	center, R, D, I;
  int u_num, v_num, w_num;
  if(do_3d){
    widget3d->GetPosition( center, R, D, I);
    double u_fac = widget3d->GetRatioR();
    double v_fac = widget3d->GetRatioD();
    double w_fac = widget3d->GetRatioI();
    u_num = (int)(u_fac*100);
    v_num = (int)(v_fac*100);
    w_num = (int)(w_fac*100);
  } else {
    widget2d->GetPosition( center, R, D);
    I = center;
    double u_fac = widget2d->GetRatioR();
    double v_fac = widget2d->GetRatioD();
    u_num = (int)(u_fac*100);
    v_num = (int)(v_fac*100);
    w_num = 2;
  }
  Vector v1 = R - center,
    v2 = D - center,
    v3 = I - center;
    
  //u_num=v_num=w_num=4;

  // calculate the corner and the
  // u and v vectors of the cutting plane
  Point corner = center - v1 - v2 - v3;
  Vector u = v1 * 2.0,
    v = v2 * 2.0,
    w = v3 * 2.0;
    
  // create the grid for the cutting plane
  double lenscale = length_scale.get(),
    widscale = width_scale.get(),
    headlen = head_length.get();
  //int exhaustive = exhaustive_flag.get();
  GeomArrows* arrows = new GeomArrows(widscale, 1.0-headlen, 
				      drawcylinders.get(), shaft_rad.get() );
  for (int i = 0; i < u_num; i++)
    for (int j = 0; j < v_num; j++)
      for(int k = 0; k < w_num; k++){
	Point p = corner + u * ((double) i/(u_num-1)) + 
	  v * ((double) j/(v_num-1)) +
	  w * ((double) k/(w_num-1));

	// Query the vector field...
	Vector vv;
	//int ii=0;
	if ( vfi->interpolate( vv, p) ){
	  if(have_sfield){
	    // get the color from cmap for p 	    
	    MaterialHandle matl;
	    double sval;
	    //			    ii=0;
	    if ( sfi->interpolate( sval, p) )
	      matl = cmap->lookup( sval);
	    else
	    {
	      matl = outcolor;
	    }

	    if(vv.length()*lenscale > 1.e-10){
	      arrows->add(p, vv*lenscale, matl, matl, matl);
            }
	  } else {
	    if(vv.length()*lenscale > 1.e-10){
	      arrows->add(p, vv*lenscale, shaft, back, head);
            }
	  }
	}
      }

  // delete the old grid/cutting plane
  if (old_grid_id != 0)
    ogeom->delObj( old_grid_id );

  grid_id = ogeom->addObj(arrows, module_name);
}

void
Hedgehog::widget_moved(bool last, BaseWidget*)
{
  if(last && !abort_flag_)
    {
      abort_flag_ = true;
      want_to_execute();
    }
}

void
Hedgehog::tcl_command(GuiArgs& args, void* userdata)
{
  if(args.count() < 2)
    {
      args.error("Streamline needs a minor command");
      return;
    }
  if(args[1] == "findxy")
    {
      if(type.get() == "2D")
        need_find2d=1;
      else
        need_find3d=1;
      want_to_execute();
    }
  else if(args[1] == "findyz")
    {
      if(type.get() == "2D")
        need_find2d=2;
      else
        need_find3d=1;
      want_to_execute();
    }
  else if(args[1] == "findxz")
    {
      if(type.get() == "2D")
        need_find2d=3;
      else
        need_find3d=1;
      want_to_execute();
    }
  else
    {
      Module::tcl_command(args, userdata);
    }
}

} // End namespace Uintah
