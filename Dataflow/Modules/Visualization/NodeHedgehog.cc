
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
#include <Core/Datatypes/FieldInterface.h>
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
#include <Core/Thread/Thread.h>
#include <Core/Thread/Semaphore.h>
#include <Core/Thread/Mutex.h>
#include <Packages/Uintah/Core/Grid/Box.h>

#include <Dataflow/Widgets/BoxWidget.h>
#include <Dataflow/Widgets/FrameWidget.h>
#include <Core/Datatypes/LatVolField.h>
#include <Core/Datatypes/LatVolMesh.h>

#include <iostream>
#include <vector>

using std::cerr;
using std::vector;

#define CP_PLANE 0
#define CP_SURFACE 1
#define CP_CONTOUR 2

namespace Uintah {

using namespace SCIRun;
class NodeHedgehogWorker;
  //#define USE_HOG_THREADS
  
/**************************************
CLASS
   NodeHedgehog
      NodeHedgehog produces arrows at node locations in a vector field.


GENERAL INFORMATION

   NodeHedgehog
  
   Author:  James Bigler (bigler@cs.utah.edu)
            
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
  // widget_moved -  
  virtual void widget_moved(bool last);

  // GROUP: Private Function:

  // struct to pass into add_arrow
  struct ArrowInfo {
    double arrow_length_scale;
    double minlen;
    double maxlen;
    bool have_sfield;
    ScalarFieldInterfaceHandle sf_interface;
    bool have_cmap;
    ColorMapHandle cmap;
  };
  
  // GROUP: Private Function:
  //////////////////////
  // add_arrow -
  void add_arrow(Point &v_origin, Vector &vf_value, GeomArrows *arrows,
		 ArrowInfo &info);
		   
  GuiDouble length_scale;
  GuiDouble min_crop_length;
  GuiDouble max_crop_length;
  GuiDouble width_scale;
  GuiDouble head_length;
  GuiString type;
  GuiInt drawcylinders;
  GuiInt norm_head;
  GuiInt skip_node;
  GuiDouble shaft_rad;
  MaterialHandle outcolor;
  // This number is used to delete only the arrow geometry without destroying
  // the widget geometry
  int geom_id;
  int need_find2d;
  int need_find3d;

  // max Vector stuff
  Vector max_vector;
  double max_length;
  GuiDouble max_vector_x;
  GuiDouble max_vector_y;
  GuiDouble max_vector_z;
  GuiDouble max_vector_length;

  Mutex add_arrows;

#ifdef USE_HOG_THREADS
  friend class NodeHedgehogWorker;
#endif
public:
 
  // GROUP:  Constructors:
  ///////////////////////////
  // Constructs an instance of class NodeHedgehog
  // Constructor taking
  //    [in] id as an identifier
  NodeHedgehog(GuiContext* ctx);
  
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
  virtual void tcl_command(GuiArgs&, void*);

  // used to get the values.
  //  bool interpolate(FieldHandle f, const Point& p, Vector& val);
  //  bool interpolate(FieldHandle f, const Point& p, double& val);
};

#ifdef USE_HOG_THREADS
  
class NodeHedgehogWorker: public Runnable {
public:
  NodeHedgehogWorker(Patch *patch, LatVolField<Vector> *field,
		     FieldHandle ssfield,
		     bool have_sfield, ColorMapHandle cmap, bool have_cmap,
		     Box boundaryRegion,
		     NodeHedgehog *hog, GeomGroup *arrow_group,
		     double *max_length, Vector* max_vector,
		     Mutex *amutex, Semaphore *sema):
    patch(patch), field(field), ssfield(ssfield), have_sfield(have_sfield),
    cmap(cmap), have_cmap(have_cmap), boundaryRegion(boundaryRegion), hog(hog),
    arrow_group(arrow_group),
    max_length(max_length), max_vector(max_vector), amutex(amutex), sema(sema),
    my_max_length(0), my_max_vector(0,0,0)
  {
    arrows = scinew GeomArrows(hog->width_scale.get(),
			       1.0 - hog->head_length.get(),
			       hog->drawcylinders.get(),
			       hog->shaft_rad.get(),
			       hog->norm_head.get());
  }
  
  void run() {
    // extract the vectors and add them to arrows
    int skip = hog->skip_node.get();
    if (skip < 1)
      skip = 1;
    if( field->data_at() == Field::NODE) {
      //------------------------------------
      // for each node in the patch
      for(NodeIterator iter = patch->getNodeIterator(boundaryRegion); !iter.done(); iter+=skip){
	IntVector idx = *iter;
	Point p = patch->nodePosition(idx);
	Vector vv = field->fdata().get_data_by_patch_and_index(patch, idx);
	add_arrow(vv, p, hog->length_scale.get());
      } // end for loop
    } else if( field->data_at() == Field::CELL) {
      for(CellIterator iter = patch->getCellIterator(boundaryRegion); !iter.done(); iter+=skip){
	IntVector idx = *iter;
	Point p = patch->cellPosition(idx);
	Vector vv = field->fdata().get_data_by_patch_and_index(patch, idx);
	add_arrow(vv, p, hog->length_scale.get());
      } // end for loop
    } // end if NODE

    // now add the arrows to arrow_group and update the max_length
    // and max_vector
    amutex->lock();
    //    cerr << "NodeHedgehogWorker is about to add the goods\n";
    if (my_max_length > *max_length) {
      *max_length = my_max_length;
      *max_vector = my_max_vector;
    }
    arrow_group->add((GeomObj*)arrows);
    amutex->unlock();
    sema->up();
  }
private:
  // passed in
  Patch *patch;
  LatVolField<Vector> *field;
  FieldHandle ssfield;
  bool have_sfield;
  ColorMapHandle cmap;
  bool have_cmap;
  Box boundaryRegion;
  NodeHedgehog *hog;
  GeomGroup *arrow_group;
  double *max_length;
  Vector *max_vector;
  Mutex *amutex;
  Semaphore *sema;

  // local to thread
  double my_max_length;
  Vector my_max_vector;
  GeomArrows *arrows;

  // functions
  void add_arrow(Vector vv, Point p, double lenscale) {
    double length = vv.length();
    
    // crop the vector based on it's length
    // we only want to check against max_crop_length if it's greater than 0
    if (hog->max_crop_length.get() > 0)
      if (length > hog->max_crop_length.get())
	return;
    if (length < hog->min_crop_length.get())
      return;

    // calculate the maximum based on vectors not cropped
    if (length > my_max_length) {
      my_max_length = length;
      my_max_vector = vv;
    }

    if(have_cmap) {
      if(have_sfield) {
	// get the color from cmap for p 	    
	MaterialHandle matl;
	double sval;
	if (hog->interpolate( ssfield, p, sval))
	  matl = cmap->lookup( sval);
	else {
	  matl = hog->outcolor;
	}
      
	arrows->add(p, vv*lenscale, matl, matl, matl);
      } else {
	MaterialHandle matl = cmap->lookup( 0);
	arrows->add(p, vv*lenscale, matl, matl, matl);
      }
    }
    else {
      arrows->add(p, vv*lenscale);
    }
  } // end add_arrow
  
};
#endif // USE_HOG_THREADS
  
static string module_name("NodeHedgehog");
static string widget_name("NodeHedgehog Widget");
DECLARE_MAKER(NodeHedgehog)

NodeHedgehog::NodeHedgehog(GuiContext* ctx):
  Module("NodeHedgehog", ctx, Filter, "Visualization", "Uintah"),
  widget_lock("NodeHedgehog widget lock"),
  length_scale(ctx->subVar("length_scale")),
  min_crop_length(ctx->subVar("min_crop_length")),
  max_crop_length(ctx->subVar("max_crop_length")),
  width_scale(ctx->subVar("width_scale")),
  head_length(ctx->subVar("head_length")),
  type(ctx->subVar("type")),
  drawcylinders(ctx->subVar("drawcylinders")),
  norm_head(ctx->subVar("norm_head")),
  skip_node(ctx->subVar("skip_node")),
  shaft_rad(ctx->subVar("shaft_rad")),
  max_vector(0,0,0), max_length(0),
  max_vector_x(ctx->subVar("max_vector_x")),
  max_vector_y(ctx->subVar("max_vector_y")),
  max_vector_z(ctx->subVar("max_vector_z")),
  max_vector_length(ctx->subVar("max_vector_length")),
  add_arrows("NodeHedgehog add_arrows mutex")
{
  init = 1;
  float INIT(.1);
  
  widget2d = scinew FrameWidget(this, &widget_lock, INIT, false);
  widget3d = scinew BoxWidget(this, &widget_lock, INIT, true, false);
  geom_id=0;
  
  need_find2d=1;
  need_find3d=1;
  
  drawcylinders.set(0);
  norm_head.set(0);
  outcolor=scinew Material(Color(0,0,0), Color(0,0,0), Color(0,0,0), 0);
}

NodeHedgehog::~NodeHedgehog()
{
}

void NodeHedgehog::add_arrow(Point &v_origin, Vector &vf_value,
			     GeomArrows *arrows, ArrowInfo &info) {
  //cout << "v_origin = "<<v_origin<<", vf_value = "<<vf_value<<endl;

  // crop the vector based on it's length
  // we only want to check against max_crop_length if it's greater than SMALL_NUM
  // it locks up if the vector is too small.
  double length = vf_value.length();
  double SMALL_NUM = 1.0e-50;    //--Todd
  //cout << "vf_value"<<vf_value<<"length("<<length<<")";flush(cout);

  if (length <= SMALL_NUM)
    // zero length vectors are not added
    return;
  if (info.maxlen > SMALL_NUM)
    if (length > info.maxlen)
      return;
  if (length < info.minlen)
    return;

  // Keep track of the vector with the maximun length and that length
  if (length > max_length) {
    max_length = length;
    max_vector = vf_value;
  }

  MaterialHandle arrow_color;
  if (info.have_cmap) {
    if (info.have_sfield) {
      // query the scalar field
      double sf_value;
      double minout = info.sf_interface->find_closest(sf_value, v_origin);
      cout << "minout = "<<minout<<", sf_value = "<<sf_value<<", v_origin = "<<v_origin<<endl;
      arrow_color = info.cmap->lookup( sf_value );
    } else {
      // Grab a value from the color map.
      // lookup2 is used, so we can indicate that we want the middle color.
      // lookup2 takes a double from 0 to 1 and then indexes into the ColorMap.
      arrow_color = info.cmap->lookup2( 0.5 );
    }
    arrows->add(v_origin, vf_value * info.arrow_length_scale,
		arrow_color, arrow_color, arrow_color);
  } else {
    arrows->add(v_origin, vf_value * info.arrow_length_scale);
  }
}

void NodeHedgehog::execute()
{
  cerr << "NodeHedgehog::execute: start\n";
  int old_geom_id = geom_id;

  // Create the input port
  inscalarfield = (FieldIPort *) get_iport("Scalar Field");
  invectorfield = (FieldIPort *) get_iport("Vector Field");
  inColorMap = (ColorMapIPort *) get_iport("ColorMap");
					
  // Create the output port
  ogeom = (GeometryOPort *) get_oport("Geometry"); 

  // Must have a vector field, otherwise exit
  cerr << "NodeHedgehog::execute:attempting to extract vector field from port.\n";
  FieldHandle vfield;
  if (!invectorfield->get( vfield ))
    return;

  if( vfield->get_type_name(1) != "Vector" ){
    cerr<<"First field must be a Vector field.\n";
    return;
  }
  
  if(vfield->get_type_name(0) != "LatVolField") {
    cerr<<"Not a LatVolField\n";
    return;
  }

  LatVolField<Vector> *fld =
    dynamic_cast<LatVolField<Vector>*>(vfield.get_rep());
  // if fld == NULL then the cast didn't work, so bail
  if (!fld) {
    cerr << "Cannot cast field into a LatVolField\n";
    return;
  }
  
  // Get the scalar field and ColorMap...if you can
  FieldHandle ssfield;
  int have_sfield=inscalarfield->get( ssfield );
  ScalarFieldInterfaceHandle sf_interface;
  if( have_sfield ){
    if( !ssfield->is_scalar() ){
      cerr<<"Second field is not a scalar field.  No Colormapping.\n";
      have_sfield = 0;
    } else {
      sf_interface = ssfield->query_scalar_interface();
      if (sf_interface.get_rep() == 0)
	have_sfield = 0;
    }
  }
  ColorMapHandle cmap;
  int have_cmap=inColorMap->get( cmap );
  if (have_cmap) {
    if (cmap->IsScaled())
      cout << "cmap is scaled.\n";
    else 
      cout << "cmap is not scaled.\n";
    cout << "cmap.getMin() = "<<cmap->getMin()<<", getMax() = "<<cmap->getMax()<<endl;
  }
  
  cout << "NodeHedgehog::execute:initializing phase\n";
  if (init == 1) {
    init = 0;
    GeomHandle w2d = widget2d->GetWidget() ;
    GeomHandle w3d = widget3d->GetWidget() ;
    GeomHandle w = new GeomGroup;
    ((GeomGroup*)w.get_rep())->add(w2d);
    ((GeomGroup*)w.get_rep())->add(w3d);
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
  widget2d->SetCurrentMode(3);
  widget3d->SetCurrentMode(6);

  BBox mesh_boundary = vfield->mesh()->get_bounding_box();
  
  // draws the widget based on the vfield's size and position
  if (do_3d){
    if(need_find3d != 0){
      Point min, max;
      min = mesh_boundary.min(); max = mesh_boundary.max();
      cout << "mesh::min = "<<min<<", max = "<<max<<endl;
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
      Point min, max;
      min = mesh_boundary.min(); max = mesh_boundary.max();
      cout << "mesh::min = "<<min<<", max = "<<max<<endl;
      
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

  cout << "NodeHedgehog::execute:calculating frame widget boundaries\n";
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
  cout << "lower = "<<lower<<", upper = "<<upper<<endl;
//   upper = Min(upper, mesh_boundary.max());
//   lower = Max(lower, mesh_boundary.min());
//   cout << "lower = "<<lower<<", upper = "<<upper<<endl;
  
  cout << "NodeHedgehog::execute:add arrows\n";
  // create the group for the arrows to be added to
#ifdef USE_HOG_THREADS
  GeomGroup *arrows = scinew GeomGroup;
#else
  GeomArrows* arrows = scinew GeomArrows(width_scale.get(), 1.0-head_length.get(), drawcylinders.get(), shaft_rad.get(), norm_head.get() );
#endif

  // reset max_length and max_vector
  max_length = 0;
  max_vector = Vector(0,0,0);
  
  // We need a few pieces of information.
  // 1. All the Node/Cell locations (we'll call that v_origin)
  // 2. The value of the vector field at v_origin ( vf_value )
  // 3. The color to use for the arrow ( arrow_color ).  This is:
  //    A. If there is a color map
  //       1. If there is a scalar field
  //          a. Get the value of the scalar field at v_origin ( sf_value )
  //          b. The color by indexing into color map by sf_value
  //       2. If there is no scalar field, the value of cman.lookup(0)
  //    B. Without a color map, use a default color ( green ).

  // get the mesh for the geometry
  LatVolMesh *mesh = fld->get_typed_mesh().get_rep();
  // Access length_scale once, because accessing a tcl variable can be
  // expensive if inside of a loop.
  ArrowInfo info;
  info.arrow_length_scale = length_scale.get();
  info.minlen = min_crop_length.get();
  info.maxlen = max_crop_length.get();
  info.have_sfield = have_sfield;
  info.sf_interface = sf_interface;
  info.have_cmap = have_cmap;
  info.cmap = cmap;
  
#ifdef USE_HOG_THREADS
  // break up the volume into smaller pieces and then loop over each piece
#else
  // Make a switch based on the data location
  if( fld->data_at() == Field::CELL) {
    // Create the sub-mesh that represents where the widget is
#if 0
    LatVolMesh::Cell::index_type min_index, max_index;
    bool min_valid = mesh->locate(min_index, lower);
    bool max_valid = mesh->locate(max_index, upper);
    cout << "min_index = ["<<min_index.i_<<", "<<min_index.j_<<", "<<min_index.k_<<"]";
    if (min_valid) cout << " valid!\n"; else cout << " not valid\n";
    cout << "max_index = ["<<max_index.i_<<", "<<max_index.j_<<", "<<max_index.k_<<"]";
    if (max_valid) cout << " valid!\n"; else cout << " not valid\n";
#endif
    LatVolMesh::Cell::range_iter iter;
    LatVolMesh::Cell::iterator end;
    mesh->get_cell_range(iter, end, BBox(lower, upper));
#if 0
    cout << "begin["<<(*iter).i_<<", "<<(*iter).j_<<", "<<(*iter).k_<<"]\n";
    cout << "end["<<(*end).i_<<", "<<(*end).j_<<", "<<(*end).k_<<"]\n";
    LatVolMesh::Cell::iterator mesh_begin; mesh->begin(mesh_begin);
    LatVolMesh::Cell::iterator mesh_end;   mesh->end(mesh_end);
    cout << "mesh_begin["<<(*mesh_begin).i_<<", "<<(*mesh_begin).j_<<", "<<(*mesh_begin).k_<<"]\n";
    cout << "mesh_end["<<(*mesh_end).i_<<", "<<(*mesh_end).j_<<", "<<(*mesh_end).k_<<"]\n";
    // Cropping upper and lower to the boundaries
    cout << "mesh_boundary.min = "<<mesh_boundary.min()<<", max = "<<mesh_boundary.max()<<endl;
    // crop by min boundary
    upper = Max(upper, mesh_boundary.min()); 
    lower = Max(lower, mesh_boundary.min());
    // crop by max boundary
    upper = Min(upper, mesh_boundary.max()); 
    lower = Min(lower, mesh_boundary.max());
    cout << "Cropping upper and lower to the boundaries\n";
    cout << "lower = "<<lower<<", upper = "<<upper<<endl;
    min_valid = mesh->locate(min_index, lower);
    max_valid = mesh->locate(max_index, upper);
    cout << "min_index = ["<<min_index.i_<<", "<<min_index.j_<<", "<<min_index.k_<<"]";
    if (min_valid) cout << " valid!\n"; else cout << " not valid\n";
    cout << "max_index = ["<<max_index.i_<<", "<<max_index.j_<<", "<<max_index.k_<<"]";
    if (max_valid) cout << " valid!\n"; else cout << " not valid\n";
#endif


#if 0
    for(; iter != end; ++iter) {     //  O L D   S T Y L E 
          Point v_origin;
          mesh->get_center(v_origin, *iter);
          //cout << "iter["<<(*iter).i_<<", "<<(*iter).j_<<", "<<(*iter).k_<<"], ";
          //cout << ", v_origin = "<<v_origin<<endl;
          Vector vf_value = fld->value(*iter);
          add_arrow(v_origin, vf_value, arrows, info); 
    }
#endif

    //__________________________________
    //  Find the upper and lower limits of the bounding
    // box.  There has to be a cleaner way -- Todd
    int max_i = (*iter).i_, max_j = (*iter).j_, max_k = (*iter).k_;
    int min_i = (*iter).i_, min_j = (*iter).j_, min_k = (*iter).k_;
    for(; iter != end; ++iter) {
      max_i = std::max( max_i, (int)(*iter).i_ );
      max_j = std::max( max_j, (int)(*iter).j_ );
      max_k = std::max( max_k, (int)(*iter).k_ );

      min_i = std::min( min_i, (int)(*iter).i_ );
      min_j = std::min( min_j, (int)(*iter).j_ );
      min_k = std::min( min_k, (int)(*iter).k_ );
    }
    IntVector lo(min_i, min_j, min_k);
    IntVector hi(max_i, max_j, max_k);
//    cout << "low " << lo << " Hi " << hi << endl;
    
    //__________________________________
    //  setup the number of skips in each dir
    int skip_x = skip;
    int skip_y = skip;
    int skip_z = skip;
   
    if ( hi.x() - lo.x() < 4) {  // If we're looking at a 2D slice
      skip_x = 1;                // don't skip the data in the slice
    } 
    if ( hi.y() - lo.y() < 4) {
      skip_y = 1;
    }
    if ( hi.z() - lo.z() < 4) {
      skip_z = 1;
    }  
 
    for (int i = lo.x(); i <= hi.x(); i += skip_x) {
      for (int j = lo.y(); j <= hi.y(); j += skip_y) {
        for (int k = lo.z(); k <= hi.z(); k += skip_z) {         
          LatVolMesh::Cell::index_type  idx(mesh, i, j, k); 
                           
          Point v_origin;
          mesh->get_center(v_origin, idx);
          
          Vector vf_value = fld->fdata()[ idx ];
          add_arrow(v_origin, vf_value, arrows, info);
         }
       }
     }
  } else if( fld->data_at() == Field::NODE) {
#if 0
    LatVolMesh::Node::index_type min_index, max_index;
    bool min_valid = mesh->locate(min_index, lower);
    bool max_valid = mesh->locate(max_index, upper);
    cout << "min_index = ["<<min_index.i_<<", "<<min_index.j_<<", "<<min_index.k_<<"]";
    if (min_valid) cout << " valid!\n"; else cout << " not valid\n";
    cout << "max_index = ["<<max_index.i_<<", "<<max_index.j_<<", "<<max_index.k_<<"]";
    if (max_valid) cout << " valid!\n"; else cout << " not valid\n";
#endif
    LatVolMesh::Node::range_iter iter;
    LatVolMesh::Node::iterator end;
    mesh->get_node_range(iter, end, BBox(lower, upper));
    

#if 0
    cout << "begin["<<(*iter).i_<<", "<<(*iter).j_<<", "<<(*iter).k_<<"]\n";
    cout << "end["<<(*end).i_<<", "<<(*end).j_<<", "<<(*end).k_<<"]\n";
    LatVolMesh::Node::iterator mesh_begin; mesh->begin(mesh_begin);
    LatVolMesh::Node::iterator mesh_end;   mesh->end(mesh_end);
    cout << "mesh_begin["<<(*mesh_begin).i_<<", "<<(*mesh_begin).j_<<", "<<(*mesh_begin).k_<<"]\n";
    cout << "mesh_end["<<(*mesh_end).i_<<", "<<(*mesh_end).j_<<", "<<(*mesh_end).k_<<"]\n";
    // Cropping upper and lower to the boundaries
    cout << "mesh_boundary.min = "<<mesh_boundary.min()<<", max = "<<mesh_boundary.max()<<endl;
    // crop by min boundary
    upper = Max(upper, mesh_boundary.min()); 
    lower = Max(lower, mesh_boundary.min());
    // crop by max boundary
    upper = Min(upper, mesh_boundary.max()); 
    lower = Min(lower, mesh_boundary.max());
    cout << "Cropping upper and lower to the boundaries\n";
    cout << "lower = "<<lower<<", upper = "<<upper<<endl;
    min_valid = mesh->locate(min_index, lower);
    max_valid = mesh->locate(max_index, upper);
    cout << "min_index = ["<<min_index.i_<<", "<<min_index.j_<<", "<<min_index.k_<<"]";
    if (min_valid) cout << " valid!\n"; else cout << " not valid\n";
    cout << "max_index = ["<<max_index.i_<<", "<<max_index.j_<<", "<<max_index.k_<<"]";
    if (max_valid) cout << " valid!\n"; else cout << " not valid\n";
#endif
    
    for(; iter != end; ++iter) {
      //cout << "iter["<<(*iter).i_<<", "<<(*iter).j_<<", "<<(*iter).k_<<"], ";
      //cout << "+";
     
      
      Point v_origin;
      mesh->get_center(v_origin, *iter);
      //cout << ", v_origin = "<<v_origin<<endl;
      Vector vf_value = fld->value(*iter);
      add_arrow(v_origin, vf_value, arrows, info);
    }
  }
#endif // ifdef USE_HOG_THREADS
  cout << "\nNodeHedgehog::execute:finished adding arrows\n";
  
  // This needs to happen so that we don't destroy the widget.
  // Delete the old geometry.
  if (old_geom_id != 0)
    ogeom->delObj( old_geom_id );
  
  geom_id = ogeom->addObj(arrows, module_name);

  // set the max vector stuff in the tcl code
  max_vector_length.set(max_length);
  max_vector_x.set(max_vector.x());
  max_vector_y.set(max_vector.y());
  max_vector_z.set(max_vector.z());

  cerr << "NodeHedgehog::execute: end"<<endl;
}

void NodeHedgehog::widget_moved(bool last)
{
  if(last && !abort_flag) {
    abort_flag=1;
    want_to_execute();
  }
}


void NodeHedgehog::tcl_command(GuiArgs& args, void* userdata)
{
#if 1
  if(args.count() < 2) {
    args.error("NodeHedgehog needs a minor command");
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
#endif
  cerr <<"NodeHedgehog::tcl_command: args\n";
  for(int i = 0; i < args.count(); i++)
    cerr << "args["<<i<<"] = "<<args[i]<<endl;
  cerr << "NodeHedgehog::tcl_command: was called."<<endl;
}

} // End namespace Uintah


