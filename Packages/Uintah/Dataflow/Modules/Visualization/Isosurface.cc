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

#include <Packages/Uintah/Core/Datatypes/LevelField.h>

#include <Dataflow/Modules/Visualization/Isosurface.h>

#include <Dataflow/Network/Module.h>
#include <Core/Datatypes/FieldAlgo.h>
#include <Core/Algorithms/Loader/Loader.h>
#include <Core/Algorithms/Visualization/MarchingCubes.h>
#include <Core/Algorithms/Visualization/Noise.h>
#include <Core/Algorithms/Visualization/Sage.h>



namespace Uintah {

using SCIRun::Module;
using SCIRun::Field;
using SCIRun::FieldHandle;
using SCIRun::Loader;
using SCIRun::MarchingCubes;
using SCIRun::Noise;
using SCIRun::Sage;
using SCIRun::Minmax;
using SCIRun::field_minmax;



class Isosurface : public SCIRun::Isosurface {
public:
  Isosurface(const string& id);
  virtual ~Isosurface();

  virtual void initialize();
  void new_field( FieldHandle & );
  void send_results();
};


extern "C" Module* make_Isosurface(const string& id) {
  return new Isosurface(id);
}

//static string module_name("Isosurface");
static string surface_name("Isosurface");
static string widget_name("Isosurface");

Isosurface::Isosurface(const string& id)
  : SCIRun::Isosurface( id ) 
{
  packageName = "Uintah";
}
Isosurface::~Isosurface()
{
}


void
Isosurface::initialize()
{
  SCIRun::Isosurface::initialize();
  minmax_loader.store("LevelField<double>",
		      new Minmax<LevelField<double> > );
  minmax_loader.store("LevelField<double>",
		      new Minmax<LevelField<float> > );
  minmax_loader.store("LevelField<long>",
		      new Minmax<LevelField<long> > );

//   loader.store("MC::LevelField<double>", 
//  	       new MarchingCubes<Module,HexMC<LevelField<double> > >(this) );
  loader.store("Noise::LevelField<double>", 
	       new Noise<Module,HexMC<LevelField<double> > >(this) );
//   loader.store("Sage::LevelField<double>",  
//                new Sage<Module,LevelField<double> >(this) ); 

//   loader.store("MC::LevelField<float>", 
//  	       new MarchingCubes<Module,HexMC<LevelField<float> > >(this) );
  loader.store("Noise::LevelField<float>", 
	       new Noise<Module,HexMC<LevelField<float> > >(this) );
//   loader.store("Sage::LevelField<float>",  
//                new Sage<Module,LevelField<float> >(this) ); 

//   loader.store("MC::LevelField<long>", 
//  	       new MarchingCubes<Module,HexMC<LevelField<long> > >(this) );
  loader.store("Noise::LevelField<long>", 
	       new Noise<Module,HexMC<LevelField<long> > >(this) );
//   loader.store("Sage::LevelField<long>",  
//                new Sage<Module,LevelField<long> >(this) ); 





}


} // End namespace SCIRun
