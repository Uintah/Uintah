#ifndef VISPARTICLESET_H
#define VISPARTICLESET_H

#include <Core/Datatypes/Datatype.h>
#include <Core/Containers/LockingHandle.h>
#include <Packages/Uintah/Interface/DataArchive.h>
#include <Packages/Uintah/Grid/ParticleVariable.h>
#include <Core/Persistent/Persistent.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <iostream>

namespace Kurt {
using Uintah::DataArchive;
using Uintah::ParticleVariable;
using namespace SCIRun;

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
} // End namespace Kurt

#endif
