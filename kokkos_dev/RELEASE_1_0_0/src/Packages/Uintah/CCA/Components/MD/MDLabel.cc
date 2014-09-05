
#include <Packages/Uintah/CCA/Components/MD/MDLabel.h>
#include <Packages/Uintah/Core/Grid/ParticleVariable.h>
#include <Packages/Uintah/Core/Grid/CCVariable.h>
#include <Packages/Uintah/Core/Grid/NCVariable.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/Core/Grid/VarTypes.h>
#include <Packages/Uintah/Core/Grid/PerPatch.h>
#include <Core/Malloc/Allocator.h>
using namespace Uintah;
using namespace Uintah::MD;

MDLabel::MDLabel()
{
  // Particle Variables

  atomMassLabel = scinew VarLabel("atom.mass",
			ParticleVariable<double>::getTypeDescription() );
  
  atomXLabel = scinew VarLabel( "atom.x", ParticleVariable<Point>::getTypeDescription(),
			VarLabel::PositionVariable);

  // Reduction variables

  delTLabel = scinew VarLabel( "delT", delt_vartype::getTypeDescription() );
  
} 

const MDLabel* MDLabel::getLabels()
{
  static MDLabel* instance=0;
  if(!instance)
    instance=scinew MDLabel();
  return instance;
}

