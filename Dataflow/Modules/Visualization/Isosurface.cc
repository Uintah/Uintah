/*
 *  Isosurface.cc:  
 *
 *   \authur Yarden Livnat
 *   Department of Computer Science
 *   University of Utah
 *
 *   \date Feb 2001
 *
 *  Copyright (C) 2001 SCI Institute
 */

#include <map>
#include <iostream>
using std::cerr;
using std::cin;
using std::endl;
#include <sstream>
using std::ostringstream;

#include <Core/Malloc/Allocator.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/Material.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Datatypes/TetVol.h>
#include <Core/Datatypes/LatticeVol.h>

#include <Core/Algorithms/Loader/Loader.h>
#include <Core/Algorithms/Visualization/TetMC.h>
#include <Core/Algorithms/Visualization/HexMC.h>
#include <Core/Algorithms/Visualization/MarchingCubes.h>
#include <Core/Algorithms/Visualization/Noise.h>

#include <Dataflow/Network/Module.h>
#include <Dataflow/Ports/ColorMapPort.h>
#include <Dataflow/Ports/GeometryPort.h>
#include <Dataflow/Ports/FieldPort.h>
//#include <Dataflow/Ports/SurfacePort.h>


namespace SCIRun {

class MinmaxFunctor {
public:
  virtual bool get( Field *, pair<double,double>& ) = 0;
};

template<class F>
class Minmax : public MinmaxFunctor {
public:
  virtual bool get( Field *field, pair<double,double> &p ) {
    F *f = dynamic_cast<F *>(field);
    if ( !f ) return false;
    cerr << "compute minmax...\n";
    return field_minmax( *f, p );
  }
};

class Isosurface : public Module {

  // Input Ports
  FieldIPort* infield;
  FieldIPort* incolorfield;
  ColorMapIPort* inColorMap;

  // Output Ports
  GeometryOPort* ogeom;
  FieldOPort* osurf;
  

  //! GUI variables
  GuiDouble gui_iso_value;
  GuiInt    extract_from_new_field;
  GuiInt    use_algorithm;

  //! 
  double iso_value;
  FieldHandle field_;
  GeomObj *surface;
  FieldHandle colorfield;
  ColorMapHandle cmap;

  //! status variables
  int init;
  int geom_id;
  double prev_min, prev_max;
  int last_generation;
  bool have_colorfield;
  bool have_ColorMap;

  MarchingCubesAlg *mc_alg;
  NoiseAlg *noise_alg;
  Loader loader;
  Loader minmax_loader;

  MaterialHandle matl;

public:
  Isosurface(const clString& id);
  virtual ~Isosurface();
  virtual void execute();

  void initialize();
  void new_field( FieldHandle & );
  void send_results();
};


extern "C" Module* make_Isosurface(const clString& id) {
  return new Isosurface(id);
}

//static clString module_name("Isosurface");
static clString surface_name("Isosurface");
static clString widget_name("Isosurface");

Isosurface::Isosurface(const clString& id)
  : Module("Isosurface", id, Filter), 
    gui_iso_value("isoval", id, this),
    extract_from_new_field("extract-from-new-field", id, this ),
    use_algorithm("algorithm", id, this)
{
    // Create the input ports
  infield=scinew FieldIPort(this, "Field", FieldIPort::Atomic);
  add_iport(infield);
  incolorfield=scinew FieldIPort(this, "Color Field", FieldIPort::Atomic);
  add_iport(incolorfield);
  inColorMap=scinew ColorMapIPort(this, "Color Map", ColorMapIPort::Atomic);
  add_iport(inColorMap);
  

  // Create the output port
  ogeom=scinew GeometryOPort(this, "Geometry", GeometryIPort::Atomic);
  add_oport(ogeom);
//   osurf=scinew FieldOPort(this, "Surface", FieldIPort::Atomic);
//   add_oport(osurf);
  
  matl = scinew Material(Color(0,.3,0), Color(0,.6,0), Color(.7,.7,.7), 50);

  geom_id=0;
  
  prev_min=prev_max=0;
  last_generation = -1;
  mc_alg = 0;
  noise_alg = 0;
  init = true;
}

Isosurface::~Isosurface()
{
}

void Isosurface::execute()
{
  update_state(NeedData);

  FieldHandle field;
  infield->get(field);
  if(!field.get_rep()){
    return;
  }

  update_state(JustStarted);

  if( init ) {
    initialize();
    init = false;
  }
  
  if ( field->generation != last_generation ) {
    // new field
    new_field( field );
    last_generation = field->generation;
    if ( !extract_from_new_field.get() )
      return;

    // fall through and extract isosurface from the new field
  }

  // Color the surface
  have_colorfield=incolorfield->get(colorfield);
  have_ColorMap=inColorMap->get(cmap);
  
  iso_value = gui_iso_value.get();
  switch ( use_algorithm.get() ) {
  case 0:  // Marching Cubes
    // for now, use a trivial MC
    if ( !mc_alg ) {
      string type = string("MC::") + field->get_type_name();
      cerr << "look for alg = " << type << endl;
      if ( !loader.get( type, mc_alg ) ) {
	error( "can not work with this field\n");
	return;
      }
      mc_alg->set_field( field.get_rep() );
    }
    // mc_alg should be set now
    surface = mc_alg->search( iso_value );

    break;
  case 1:  // Noise
    //error("Noise not implemented\n");
    if ( !noise_alg ) {
      string type = string("Noise::") + field->get_type_name();
      cerr <<"look for alg = " << type << endl;
      if ( !loader.get( type, noise_alg ) ) {
	error( "NOISE can not work with this field\n");
	return;
      }
      noise_alg->set_field( field.get_rep() );
    }
    surface = noise_alg->search( iso_value );
    break;
  case 2:  // View Dependent
    error("View dependent not implemented\n");
    surface = 0;
    break;
  default: // Error
    error("Unknow Algorithm requested\n");
    return;
    break;
  }

  send_results();
}

void
Isosurface::send_results()
{
  // stop showing the prev. surface
  if ( geom_id )
    ogeom->delObj( geom_id );

  // if no surface than we are done
  if( !surface ) {
    geom_id=0;
    
    return;
  }

  GeomObj *geom;
  
  if(have_ColorMap && !have_colorfield){
    // Paint entire surface based on ColorMap
    geom=scinew GeomMaterial( surface , cmap->lookup(iso_value) );
  } else if(have_ColorMap && have_colorfield){
    geom = surface;     // Nothing - done per vertex
  } else {
    geom=scinew GeomMaterial( surface, matl); // Default material
  }

  // send to viewer
  geom_id=ogeom->addObj( geom, surface_name);

  // output surface
//   if (emit_surface.get()) {
//     //osurf->send(TSurfaceHandle(surf));
//     //osurf->send(FieldHandle(f));
//   }
}

void
Isosurface::initialize()
{
  minmax_loader.store("TetVol<double>", new Minmax<TetVol<double> > );
  loader.store("MC::TetVol<double>", 
	       new MarchingCubes<Module,TetMC<TetVol<double> > >(this) );
  loader.store("Noise::TetVol<double>", 
	       new Noise<Module,TetMC<TetVol<double> > >(this) );

  minmax_loader.store("LatticeVol<double>", new Minmax<LatticeVol<double> > );
  loader.store("MC::LatticeVol<double>", 
	       new MarchingCubes<Module,HexMC<LatticeVol<double> > >(this) );
  loader.store("Noise::LatticeVol<double>", 
	       new Noise<Module,HexMC<LatticeVol<double> > >(this) );

  
//   widget_id = ogeom->addObj(widget->GetWidget(), widget_name, &widget_lock);
//   widget->Connect(ogeom);
}


void
Isosurface::new_field( FieldHandle &field )
{
  string type = field->get_type_name();
  cerr << "field type = " << type << endl;

  if ( !field->is_scalar() ) {
    cerr << "Isosurface: not a scalar field\n";
    return;
  }

  MinmaxFunctor *functor;
  if ( !minmax_loader.get( type, functor ) ) {
    cerr << "isosurface module: can not compute minmax for field\n";
    return;
  }

  pair<double, double> minmax;
  if ( !functor->get( field.get_rep(), minmax ) ) {
    cerr << "field does not have minmax\n";
    return;
  }
  
  // reset the GUI

  // 1: field info
  ostringstream info;
  info << id << " set_info " << type << " " << field->generation;
  TCL::execute(info.str().c_str());

  // 2: min/max
  if(minmax.first != prev_min || minmax.second != prev_max){
    ostringstream str;
    str << id << " set_minmax " << minmax.first << " " << minmax.second;
    TCL::execute(str.str().c_str());
    prev_min = minmax.first;
    prev_max = minmax.second;
  }

  // delete any algorithms created for the previous field.
  if ( mc_alg ) { delete mc_alg; mc_alg = 0;}
  if ( noise_alg ) { delete noise_alg; noise_alg = 0;}
}


} // End namespace SCIRun
