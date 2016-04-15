

#include <Core/Grid/Variables/StaticInstantiate.h>

#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Variables/NCVariable.h>
#include <Core/Grid/Variables/SFCXVariable.h>
#include <Core/Grid/Variables/SFCYVariable.h>
#include <Core/Grid/Variables/SFCZVariable.h>
#include <Core/Grid/Variables/ParticleVariable.h>

namespace Uintah {


void instantiateVariableTypes()
{
  CCVariable<double>::getTypeDescription();
  CCVariable<float>::getTypeDescription();
  CCVariable<Stencil7>::getTypeDescription();
  CCVariable<Vector>::getTypeDescription();
  // CCVariable<cutcel>::getTypeDescription();
  CCVariable<int>::getTypeDescription();

  NCVariable<Matrix3>::getTypeDescription();
  NCVariable<Stencil7>::getTypeDescription();
  NCVariable<Vector>::getTypeDescription();
  NCVariable<double>::getTypeDescription();
  NCVariable<float>::getTypeDescription();
  NCVariable<int>::getTypeDescription();

  // ParticleVariable<CMData>::getTypeDescription();
  ParticleVariable<FILE*>::getTypeDescription();
  ParticleVariable<Matrix3>::getTypeDescription();
  ParticleVariable<Point>::getTypeDescription();
  // ParticleVariable<Short27>::getTypeDescription();
  ParticleVariable<Point>::getTypeDescription();
  ParticleVariable<Vector>::getTypeDescription();
  ParticleVariable<double>::getTypeDescription();
  ParticleVariable<float>::getTypeDescription();
  ParticleVariable<int>::getTypeDescription();
  ParticleVariable<long64>::getTypeDescription();

  SFCXVariable<Stencil7>::getTypeDescription();
  SFCXVariable<Vector>::getTypeDescription();
  SFCXVariable<double>::getTypeDescription();
  SFCXVariable<float>::getTypeDescription();
  SFCXVariable<int>::getTypeDescription();

  SFCYVariable<Stencil7>::getTypeDescription();
  SFCYVariable<Vector>::getTypeDescription();
  SFCYVariable<double>::getTypeDescription();
  SFCYVariable<float>::getTypeDescription();
  SFCYVariable<int>::getTypeDescription();

  SFCZVariable<Stencil7>::getTypeDescription();
  SFCZVariable<Vector>::getTypeDescription();
  SFCZVariable<double>::getTypeDescription();
  SFCZVariable<float>::getTypeDescription();
  SFCZVariable<int>::getTypeDescription();
  
}

} // end namespace Uintah
