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
using std::cin;
using std::endl;
#include <sstream>
using std::ostringstream;

#include <Dataflow/Modules/Visualization/Isosurface.h>

//#include <typeinfo>
#include <Core/Malloc/Allocator.h>
#include <Core/Geom/GeomGroup.h>
#include <Core/Geom/Material.h>
#include <Core/GuiInterface/GuiVar.h>
#include <Core/Datatypes/TetVol.h>
#include <Core/Datatypes/MaskedTetVol.h>
#include <Core/Datatypes/LatticeVol.h>
#include <Core/Datatypes/MaskedLatticeVol.h>
#include <Core/Datatypes/TriSurf.h>

#include <Core/Algorithms/Loader/Loader.h>
#include <Core/Algorithms/Visualization/TetMC.h>
#include <Core/Algorithms/Visualization/HexMC.h>
#include <Core/Algorithms/Visualization/MarchingCubes.h>
#include <Core/Algorithms/Visualization/Noise.h>
#include <Core/Algorithms/Visualization/Sage.h>

#include <Dataflow/Network/Module.h>
//#include <Dataflow/Ports/SurfacePort.h>


namespace SCIRun {



extern "C" Module* make_Isosurface(const string& id) {
  return new Isosurface(id);
}

//static string module_name("Isosurface");
static string surface_name("Isosurface");
static string widget_name("Isosurface");

Isosurface::Isosurface(const string& id)
  : Module("Isosurface", id, Filter, "Visualization", "SCIRun"), 
    gui_iso_value("isoval", id, this),
    extract_from_new_field("extract-from-new-field", id, this ),
    use_algorithm("algorithm", id, this),
    build_trisurf_("build_trisurf", id, this),
    np_("np", id, this)  
{
  matl = scinew Material(Color(0,.3,0), Color(0,.6,0), Color(.7,.7,.7), 50);

  geom_id=0;
  
  prev_min=prev_max=0;
  last_generation = -1;
  mc_alg = 0;
  noise_alg = 0;
  sage_alg = 0;
  init = true;
}

Isosurface::~Isosurface()
{
}

void Isosurface::execute()
{
  update_state(NeedData);
  infield = (FieldIPort *)get_iport("Field");
  inColorMap = (ColorMapIPort *)get_iport("Color Map");
  osurf = (FieldOPort *)get_oport("Surface");
  ogeom = (GeometryOPort *)get_oport("Geometry");
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
  //   have_colorfield=incolorfield->get(colorfield);
  have_ColorMap=inColorMap->get(cmap);
  
  iso_value = gui_iso_value.get();
  trisurf_mesh_ = 0;
  build_trisurf = build_trisurf_.get();
  switch ( use_algorithm.get() ) {
  case 0:  // Marching Cubes
    if ( !mc_alg ) {
      const string type = "MC::" + field->get_type_name();
      if ( !loader.get( type, mc_alg ) ) {
	error( "Marching Cubes can not work with this field.");
	return;
      }
    }
    // mc_alg should be set now
    mc_alg->set_np( np_.get() ); 
    if ( np_.get() > 1 )
      build_trisurf = false;
    mc_alg->set_field( field.get_rep() );
    mc_alg->search( iso_value, build_trisurf);
    surface = mc_alg->get_geom();
    trisurf_mesh_ = mc_alg->get_trisurf();

    break;
  case 1:  // Noise
    //error("Noise not implemented.");
    if ( !noise_alg ) {
      const string type = "Noise::" + field->get_type_name();
      remark("look for alg = " + type);
      if ( !loader.get( type, noise_alg ) ) {
	error( "NOISE can not work with this field.");
	return;
      }
      noise_alg->set_field( field.get_rep() );
    }
    surface = noise_alg->search( iso_value );
    break;
  case 2:  // View Dependent
    if ( !sage_alg ) {
      const string type = "Sage::" + field->get_type_name();
      if ( !loader.get( type, sage_alg ) ) {
	error( "SAGE can not work with this field.");
	return;
      }
      sage_alg->set_field( field.get_rep() );
    }
    {
      GeomGroup *group = new GeomGroup;
      GeomPts *points = new GeomPts(1000);
      sage_alg->search( iso_value, group, points );
      surface = group;
    }
    break;
  default: // Error
    error("Unknow Algorithm requested.");
    return;
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
  
  if(have_ColorMap /* && !have_colorfield */){
    // Paint entire surface based on ColorMap
    geom=scinew GeomMaterial( surface , cmap->lookup(iso_value) );
//   } else if(have_ColorMap && have_colorfield){
//     geom = surface;     // Nothing - done per vertex
  } else {
    geom=scinew GeomMaterial( surface, matl); // Default material
  }

  // send to viewer
  geom_id=ogeom->addObj( geom, surface_name);

  // output surface
  if (build_trisurf) {
    TriSurf<double> *ts = new TriSurf<double>(trisurf_mesh_, Field::NODE);
    vector<double>::iterator iter = ts->fdata().begin();
    while (iter != ts->fdata().end()) { (*iter)=iso_value; ++iter; }
    FieldHandle fH(ts);
    osurf->send(fH);
  }
}



void
Isosurface::new_field( FieldHandle &field )
{
  const string type = field->get_type_name();

  if ( !field->is_scalar() ) {
    error("Not a scalar input field.");
    return;
  }

  MinmaxFunctor *functor;
  if ( !minmax_loader.get( type, functor ) ) {
    error("Can not compute minmax for input field.");
    return;
  }

  pair<double, double> minmax;
  if ( !functor->get( field.get_rep(), minmax ) ) {
    error("Field does not have minmax.");
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
  if ( mc_alg ) { mc_alg->release(); mc_alg = 0;}
  if ( noise_alg ) { noise_alg->release(); noise_alg = 0;}
  if ( sage_alg ) { sage_alg->release(); sage_alg = 0;}
}

void
Isosurface::initialize()
{
  // min max
  minmax_loader.store("TetVol<unsigned_char>", 
		      new Minmax<TetVol<unsigned char> > );
  minmax_loader.store("TetVol<short>",  
		      new Minmax<TetVol<short> > );
  minmax_loader.store("TetVol<int>",    
		      new Minmax<TetVol<int> > );
  minmax_loader.store("TetVol<double>", 
		      new Minmax<TetVol<double> > );

  minmax_loader.store("MaskedTetVol<unsigned_char>", 
		      new Minmax<MaskedTetVol<unsigned char> > );
  minmax_loader.store("MaskedTetVol<short>",  
		      new Minmax<MaskedTetVol<short> > );
  minmax_loader.store("MaskedTetVol<int>",    
		      new Minmax<MaskedTetVol<int> > );
  minmax_loader.store("MaskedTetVol<double>", 
		      new Minmax<MaskedTetVol<double> > );

  minmax_loader.store("LatticeVol<unsigned_char>",   
		      new Minmax<LatticeVol<unsigned char> > );
  minmax_loader.store("LatticeVol<short>",  
		      new Minmax<LatticeVol<short> > );
  minmax_loader.store("LatticeVol<int>",    
		      new Minmax<LatticeVol<int> > );
  minmax_loader.store("LatticeVol<double>", 
		      new Minmax<LatticeVol<double> > );

  minmax_loader.store("MaskedLatticeVol<unsigned_char>",   
		      new Minmax<MaskedLatticeVol<unsigned char> > );
  minmax_loader.store("MaskedLatticeVol<short>",  
		      new Minmax<MaskedLatticeVol<short> > );
  minmax_loader.store("MaskedLatticeVol<int>",    
		      new Minmax<MaskedLatticeVol<int> > );
  minmax_loader.store("MaskedLatticeVol<double>", 
		      new Minmax<MaskedLatticeVol<double> > );
  // MC::TetVol
  loader.store("MC::TetVol<unsigned_char>", 
	       new MarchingCubes<Module,TetMC<TetVol<unsigned char> > >(this));
  loader.store("MC::TetVol<short>", 
	       new MarchingCubes<Module,TetMC<TetVol<short> > >(this) );
  loader.store("MC::TetVol<int>", 
	       new MarchingCubes<Module,TetMC<TetVol<int> > >(this) );
  loader.store("MC::TetVol<double>", 
	       new MarchingCubes<Module,TetMC<TetVol<double> > >(this) );

  // MC::MaskedTetVol
  loader.store("MC::MaskedTetVol<unsigned_char>", 
	       new MarchingCubes<Module,TetMC<MaskedTetVol<unsigned char> > >(this));
  loader.store("MC::MaskedTetVol<short>", 
	       new MarchingCubes<Module,TetMC<MaskedTetVol<short> > >(this) );
  loader.store("MC::MaskedTetVol<int>", 
	       new MarchingCubes<Module,TetMC<MaskedTetVol<int> > >(this) );
  loader.store("MC::MaskedTetVol<double>", 
	       new MarchingCubes<Module,TetMC<MaskedTetVol<double> > >(this) );

  // Noise::TetVol
  loader.store("Noise::TetVol<unsigned_char>", 
	       new Noise<Module,TetMC<TetVol<unsigned char> > >(this) );
  loader.store("Noise::TetVol<short>", 
	       new Noise<Module,TetMC<TetVol<short> > >(this) );
  loader.store("Noise::TetVol<int>", 
	       new Noise<Module,TetMC<TetVol<int> > >(this) );
  loader.store("Noise::TetVol<double>", 
	       new Noise<Module,TetMC<TetVol<double> > >(this) );


  // MC:LatticeVol
  loader.store("MC::LatticeVol<unsigned_char>", 
	  new MarchingCubes<Module,HexMC<LatticeVol<unsigned char> > >(this) );
  loader.store("MC::LatticeVol<short>", 
	       new MarchingCubes<Module,HexMC<LatticeVol<short> > >(this) );
  loader.store("MC::LatticeVol<int>", 
	       new MarchingCubes<Module,HexMC<LatticeVol<int> > >(this) );
  loader.store("MC::LatticeVol<double>", 
	       new MarchingCubes<Module,HexMC<LatticeVol<double> > >(this) );

  // MC:MaskedLatticeVol
  loader.store("MC::MaskedLatticeVol<unsigned_char>", 
	  new MarchingCubes<Module,HexMC<MaskedLatticeVol<unsigned char> > >(this) );
  loader.store("MC::MaskedLatticeVol<short>", 
	       new MarchingCubes<Module,HexMC<MaskedLatticeVol<short> > >(this) );
  loader.store("MC::MaskedLatticeVol<int>", 
	       new MarchingCubes<Module,HexMC<MaskedLatticeVol<int> > >(this) );
  loader.store("MC::MaskedLatticeVol<double>", 
	       new MarchingCubes<Module,HexMC<MaskedLatticeVol<double> > >(this) );

  // Noise::LatticeVol
  loader.store("Noise::LatticeVol<unsigned_char>", 
	       new Noise<Module,HexMC<LatticeVol<unsigned char> > >(this) );
  loader.store("Noise::LatticeVol<short>", 
	       new Noise<Module,HexMC<LatticeVol<short> > >(this) );
  loader.store("Noise::LatticeVol<int>", 
	       new Noise<Module,HexMC<LatticeVol<int> > >(this) );
  loader.store("Noise::LatticeVol<double>", 
	       new Noise<Module,HexMC<LatticeVol<double> > >(this) );

  // Sage::LatticeVol
#if 0
  loader.store("Sage::LatticeVol<unsigned_char>", 
	       new Sage<Module,LatticeVol<unsigned char> >(this) );
  loader.store("Sage::LatticeVol<short>", 
	       new Sage<Module,LatticeVol<short> >(this) );
  loader.store("Sage::LatticeVol<int>", 
	       new Sage<Module,LatticeVol<int> >(this) );
  loader.store("Sage::LatticeVol<double>", 
	       new Sage<Module,LatticeVol<double> >(this) );
#endif
}

} // End namespace SCIRun
