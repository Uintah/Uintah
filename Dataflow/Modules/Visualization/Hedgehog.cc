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
#include <Core/Datatypes/LatticeVol.h>
#include <Core/Datatypes/LatVolMesh.h>
#include <Dataflow/Widgets/ScaledBoxWidget.h>
#include <Dataflow/Widgets/ScaledFrameWidget.h>

#include <Packages/Uintah/Core/Datatypes/LevelMesh.h>
#include <Packages/Uintah/Core/Datatypes/LevelField.h>
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
  ScaledBoxWidget* widget3d;
  ScaledFrameWidget *widget2d;
  
  // GROUP:  Widgets:
  //////////////////////
  //
  // widget_moved -  
  virtual void widget_moved(int last);
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
  
public:
 
  // GROUP:  Constructors:
  ///////////////////////////
  //
  // Constructs an instance of class Hedgehog
  //
  // Constructor taking
  //    [in] id as an identifier
  //
  Hedgehog(const string& id);
       
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
  virtual void tcl_command(TCLArgs&, void*);
  bool interpolate(FieldHandle f, const Point& p, Vector& val);
  bool interpolate(FieldHandle f, const Point& p, double& val);
//  bool get_dimensions(FieldHandle f, int& nx, int& ny, int& nz);
//  template <class Mesh>  bool get_dimensions(Mesh, int&, int&, int&);

};

static string module_name("Hedgehog");
static string widget_name("Hedgehog Widget");

extern "C" Module* make_Hedgehog(const string& id) {
  return new Hedgehog(id);
}       

Hedgehog::Hedgehog(const string& id)
: Module("Hedgehog", id, Filter, "Visualization", "Uintah"),
  widget_lock("Hedgehog widget lock"),
  length_scale("length_scale", id, this),
  width_scale("width_scale", id, this),
  head_length("head_length", id, this),
  type("type", id, this),
  exhaustive_flag("exhaustive_flag", id, this),
  vector_default_color("vector_default_color", id, this),
  drawcylinders("drawcylinders", id, this),
  shaft_rad("shaft_rad", id, this),
  shaft(new Material(Color(0,0,0), Color(.6, .6, .6),
		     Color(.6, .6, .6), 10)),
  head(new Material(Color(0,0,0), Color(1,1,1), Color(.6, .6, .6), 10)),
  back(new Material(Color(0,0,0), Color(.6, .6, .6), Color(.6, .6, .6), 10))
{
    // Create the input ports
    // Need a scalar field and a ColorMap
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
    GeomObj *w2d = widget2d->GetWidget() ;
    GeomObj *w3d = widget3d->GetWidget() ;
    GeomGroup* w = new GeomGroup;
    w->add(w2d);
    w->add(w3d);
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
    
  cerr << "unum = "<<u_num<<"  vnum="<<v_num<<"  wnum="<<w_num<<"\n";
  //    u_num=v_num=w_num=4;

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
      for(int k = 0; k < w_num; k++)
      {
	Point p = corner + u * ((double) i/(u_num-1)) + 
	  v * ((double) j/(v_num-1)) +
	  w * ((double) k/(w_num-1));

	// Query the vector field...
	Vector vv;
	//int ii=0;
	if ( interpolate( vfield, p, vv)){
	  if(have_sfield){
	    // get the color from cmap for p 	    
	    MaterialHandle matl;
	    double sval;
	    //			    ii=0;
	    if ( interpolate(ssfield, p, sval) )
	      matl = cmap->lookup( sval);
	    else
	    {
	      matl = outcolor;
	    }

	    if(vv.length()*lenscale > 1.e-10)
	      arrows->add(p, vv*lenscale, matl, matl, matl);
	  } else {
	    if(vv.length()*lenscale > 1.e-10)
	      arrows->add(p, vv*lenscale, shaft, back, head);
	    //else
	    //    cerr << "vv.length()="<<vv.length()<<"\n";
	  }
	}
      }

  // delete the old grid/cutting plane
  if (old_grid_id != 0)
    ogeom->delObj( old_grid_id );

  grid_id = ogeom->addObj(arrows, module_name);
}

void Hedgehog::widget_moved(int last)
{
    if(last && !abort_flag)
	{
	    abort_flag=1;
	    want_to_execute();
	}
}


void Hedgehog::tcl_command(TCLArgs& args, void* userdata)
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

// template <class Mesh>
// bool 
// Hedgehog::get_dimensions(Mesh, int&, int&, int&)
// {
//   return false;
// }

// template<> 
// bool Hedgehog::get_dimensions(LevelMeshHandle m,
// 				 int& nx, int& ny, int& nz)
// {
//   nx = m->get_nx();
//   ny = m->get_ny();
//   nz = m->get_nz();
//   return true;
// }

// template<> 
// bool Hedgehog::get_dimensions(LatVolMeshHandle m,
// 				 int& nx, int& ny, int& nz)
// {
//   nx = m->get_nx();
//   ny = m->get_ny();
//   nz = m->get_nz();
//   return true;
// }

// bool
// Hedgehog::get_dimensions(FieldHandle texfld_,  int& nx, int& ny, int& nz)
// {
//   const string type = texfld_->get_type_name(1);
//   if( texfld_->get_type_name(0) == "LevelField" ){
//     LevelMeshHandle mesh_;
//     if (type == "Vector") {
//       LevelField<Vector> *fld =
// 	dynamic_cast<LevelField<Vector>*>(texfld_.get_rep());
//       mesh_ = fld->get_typed_mesh();
//     } else {
//       cerr << "Hedgehog error - unknown LevelField type: " << type << endl;
//       return false;
//     }
//     return get_dimensions( mesh_, nx, ny, nz );
//   } else if(texfld_->get_type_name(0) == "LatticeVol"){
//     LatVolMeshHandle mesh_;
//     if (type == "Vector") {
//       LatticeVol<Vector> *fld =
// 	dynamic_cast<LatticeVol<Vector>*>(texfld_.get_rep());
//       mesh_ = fld->get_typed_mesh();
//     } else {
//       cerr << "Hedgehog error - unknown LatticeVol type: " << type << endl;
//       return false;
//     }
//     return get_dimensions( mesh_, nx, ny, nz );
//   } else {
//     return false;
//   }
// }

bool  
Hedgehog::interpolate(FieldHandle vfld, const Point& p, Vector& val)
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
      cerr << "Uintah::Hedgehog::interpolate:: error - unknown Field type: " << type << endl;
      return false;
    }
  } else {
    if( field_type == "LatticeVol"){
    // use virtual field interpolation
      VectorFieldInterface *vfi;
      if( (vfi = vfld->query_vector_interface())){
	return vfi->interpolate( val, p);
      } 
    }    
    return false;
  }
  return false;
}


bool  
Hedgehog::interpolate(FieldHandle sfld, const Point& p, double& val)
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
      cerr << "Uintah::Hedgehog::interpolate:: error - unimplemented Field type: " << type << endl;
      return false;
    }
  } else if( sfld->get_type_name(0) == "LatticeVol" ){
    // use virtual field interpolation
    ScalarFieldInterface *sfi;
    if(( sfi = sfld->query_scalar_interface())){
      return sfi->interpolate( val, p);
    }
  } else {
    return false;
  }
  return false;
}

} // End namespace Uintah
