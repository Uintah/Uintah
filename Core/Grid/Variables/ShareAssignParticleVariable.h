/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#ifndef UINTAH_HOMEBREW_SHAREASSIGNPARTICLEVARIABLE_H
#define UINTAH_HOMEBREW_SHAREASSIGNPARTICLEVARIABLE_H

#include <Core/Grid/Variables/ParticleVariable.h>

#ifndef _WIN32
#include <unistd.h>
#endif
#include <cerrno>

namespace Uintah {

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
  { this->copyPointer(const_cast<ParticleVariable<T>&>(pv)); return *this; }

  ShareAssignParticleVariable<T>& operator=(const ShareAssignParticleVariable<T>& pv)
  { this->copyPointer(const_cast<ShareAssignParticleVariable<T>&>(pv)); return *this; }
};

} // End namespace Uintah

#endif
