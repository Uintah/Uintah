#ifndef TENSORPARTICLES_H
#define TENSORPARTICLES_H

#include "PSet.h"
#include <SCICore/Datatypes/Datatype.h>
#include <SCICore/Containers/LockingHandle.h>
#include <Uintah/Interface/DataArchive.h>
#include <Uintah/Components/MPM/Util/Matrix3.h>
#include <Uintah/Grid/ParticleVariable.h>
#include <Uintah/Grid/Grid.h>
#include <Uintah/Grid/LevelP.h>
#include <Uintah/Grid/Patch.h>
#include <SCICore/Persistent/Persistent.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/Vector.h>
#include <iostream>
#include <vector>
using std::vector;

namespace Uintah {
namespace Datatypes {

using namespace Uintah;
using SCICore::Datatypes::Datatype;
using SCICore::Containers::LockingHandle;
using SCICore::PersistentSpace::Piostream;
using SCICore::PersistentSpace::PersistentTypeID;
using SCICore::Geometry::Point;
using SCICore::Geometry::Vector;

/**************************************

CLASS
   TensorParticles
   
   Simple TensorParticles Class.

GENERAL INFORMATION

   TensorParticles.h

   Kurt Zimmerman
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Texture

DESCRIPTION
   TensorParticles class.
  
WARNING
  
****************************************/
class TensorParticles;
typedef LockingHandle<TensorParticles> TensorParticlesHandle;

class TensorParticles : public Datatype {

public:
  // GROUP: Constructors:
  //////////
  // Constructor
  TensorParticles();
  //////////
  // Constructor
  TensorParticles(const vector <ParticleVariable<Matrix3> >& tensors,
		  PSet* pset );

  // GROUP: Destructors
  //////////
  // Destructor
  virtual ~TensorParticles();
 
  // GROUP: Access
  //////////
  // return the Tensors
  vector<ParticleVariable<Matrix3> >& get(){ return tensors; }
  PSet* getParticleSet(){ return psetH.get_rep(); }


  // GROUP: Modify
  //////////  
  // Set the Particle Set Handle
  void Set(PSetHandle psh){ psetH = psh;}
  //////////  
  // Set the Tensors
  void Set(vector <ParticleVariable<Matrix3> >& s){ tensors = s; }

  void AddVar( const ParticleVariable<Matrix3> parts );


  void SetName( string vname ) { _varname = vname; }
  void SetMaterial( int index) { _matIndex = index; }
	       

  // Persistant representation
  virtual void io(Piostream&);
  static PersistentTypeID type_id;

  void get_minmax(double& v0, double& v1);
  void get_bounds(Point& p0, Point& p1){ psetH->get_bounds(p0,p1);}
protected:

  bool have_minmax;
  double data_min;
  double data_max;
  void compute_minmax();


private:
  PSetHandle psetH;


  string _varname;
  int _matIndex;
  vector<ParticleVariable<Matrix3> >  tensors;

};

} // end namespace Datatypes
} // end namespace Uintah
#endif
