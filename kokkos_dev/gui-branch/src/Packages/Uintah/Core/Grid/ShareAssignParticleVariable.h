#ifndef UINTAH_HOMEBREW_SHAREASSIGNPARTICLEVARIABLE_H
#define UINTAH_HOMEBREW_SHAREASSIGNPARTICLEVARIABLE_H

#include <Packages/Uintah/Core/Grid/ParticleVariable.h>

#include <unistd.h>
#include <errno.h>

namespace Uintah {

using namespace SCIRun;

class TypeDescription;

/**************************************

CLASS
ShareAssignParticleVariable
  
Short description...

GENERAL INFORMATION

ShareAssignParticleVariable.h

Wayne Witzel
Department of Computer Science
University of Utah

Center for the Simulation of Accidental Fires and Explosions (C-SAFE)

Copyright (C) 2001 SCI Group

KEYWORDS
Particle_Variable, Assignment

DESCRIPTION
Long description...
  
WARNING
  
****************************************/

template<class T>
class ShareAssignParticleVariable : public ParticleVariable<T> {
public:
  ShareAssignParticleVariable()
    : ParticleVariable<T>() {}
  ShareAssignParticleVariable(const ParticleVariable<T>& pv)
    : ParticleVariable<T>(pv) {}

  virtual ~ShareAssignParticleVariable() {}
  
  ShareAssignParticleVariable<T>& operator=(const ParticleVariable<T>& pv)
  { copyPointer(const_cast<ParticleVariable<T>&>(pv)); return *this; }

  ShareAssignParticleVariable<T>& operator=(const ShareAssignParticleVariable<T>& pv)
  { copyPointer(const_cast<ShareAssignParticleVariable<T>&>(pv)); return *this; }
};

} // End namespace Uintah

#endif
