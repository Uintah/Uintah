#ifndef VISPARTICLESET_H
#define VISPARTICLESET_H

#include <SCICore/Datatypes/Datatype.h>
#include <SCICore/Containers/LockingHandle.h>
#include <Uintah/Interface/DataArchive.h>
#include <Uintah/Grid/ParticleVariable.h>
#include <SCICore/Persistent/Persistent.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/Vector.h>
#include <iostream>

namespace Kurt {
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
   VisParticleSet
   
   Simple VisParticleSet Class.

GENERAL INFORMATION

   VisParticleSet.h

   Kurt Zimmerman
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Texture

DESCRIPTION
   VisParticleSet class.
  
WARNING
  
****************************************/
class VisParticleSet;
typedef LockingHandle<VisParticleSet> VisParticleSetHandle;

class VisParticleSet : public Datatype {

public:
  // GROUP: Constructors:
  //////////
  // Constructor
  VisParticleSet();
  //////////
  // Constructor
  VisParticleSet(const ParticleVariable<Point>& positions,
		 const ParticleVariable<double>& scalars,
		 const ParticleVariable<Vector>& vectors,
		 void* callbackClass);

  // GROUP: Destructors
  //////////
  // Destructor
  virtual ~VisParticleSet();
 
  // GROUP: Access
  //////////
  // return the Points
  ParticleVariable<Point>&  getPositions(){ return positions;}
  //////////
  // return the Vectors
  ParticleVariable<Vector>& getVectors(){ return vectors; }
  //////////
  // return the Scalars
  ParticleVariable<double>& getScalars(){ return scalars; }

  //////////
  // return the Scalars
  void* getCallbackClass(){ return cbClass; }
  // GROUP: Modify
  //////////  
  // Set the Scalars
  void SetScalars(const ParticleVariable<double>& s){ scalars = s; }
  //////////
  // Set the Vectors
  void SetVectors(const ParticleVariable<Vector>& v){ vectors = v; }
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
  ParticleVariable<double> scalars;
  ParticleVariable<Vector> vectors;
};

} // end namespace Datatypes
} // end namespace Kurt
#endif
