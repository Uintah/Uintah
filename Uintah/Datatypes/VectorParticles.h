#ifndef VECTORPARTICLES_H
#define VECTORPARTICLES_H

#include <SCICore/Datatypes/Datatype.h>
#include <SCICore/Containers/LockingHandle.h>
#include <Uintah/Interface/DataArchive.h>
//#include <Uintah/Components/MPM/Util/Matrix3D.h>
#include <Uintah/Grid/ParticleVariable.h>
#include <SCICore/Persistent/Persistent.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/Vector.h>
#include <iostream>

namespace Uintah {
namespace Datatypes {

using Uintah::DataArchive;
using Uintah::ParticleVariable;
using SCICore::Datatypes::Datatype;
using SCICore::Containers::LockingHandle;
using SCICore::PersistentSpace::Piostream;
using SCICore::PersistentSpace::PersistentTypeID;
using SCICore::Geometry::Point;
using SCICore::Geometry::Vector;

/**************************************

CLASS
   VectorParticles
   
   Simple VectorParticles Class.

GENERAL INFORMATION

   VectorParticles.h

   Kurt Zimmerman
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Texture

DESCRIPTION
   VectorParticles class.
  
WARNING
  
****************************************/
class VectorParticles;
typedef LockingHandle<VectorParticles> VectorParticlesHandle;

class VectorParticles : public Datatype {

public:
  // GROUP: Constructors:
  //////////
  // Constructor
  VectorParticles();
  //////////
  // Constructor
  VectorParticles(const ParticleVariable<Point>& positions,
		 const ParticleVariable<Vector>& vectors,
		 void* callbackClass);

  // GROUP: Destructors
  //////////
  // Destructor
  virtual ~VectorParticles();
 
  // GROUP: Access
  //////////
  // return the Points
  ParticleVariable<Point>&  getPositions(){ return positions;}
  //////////
  // return the Vectors
  ParticleVariable<Vector>& get(){ return vectors; }
  //////////
  // return the callback
  void* getCallbackClass(){ return cbClass; }
  // GROUP: Modify
  //////////  
  // Set the Vectors
  void Set(const ParticleVariable<Vector>& s){ vectors = s; }
  //////////
  // Set the particle Positions
  void SetPositions(const ParticleVariable<Point>& p){ positions = p; }
  //////////
  // Set callback class
  void SetCallbackClass( void* cbc){ cbClass = cbc; }

  // Persistant representation
  virtual void io(Piostream&);
  static PersistentTypeID type_id;

private:


  void* cbClass;

  ParticleVariable<Point> positions;
  ParticleVariable<Vector> vectors;
};

} // end namespace Datatypes
} // end namespace Kurt
#endif
