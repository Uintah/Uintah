/* REFERENCED */
static char *id="@(#) $Id$";

#include <Uintah/Components/MD/MDLabel.h>
#include <Uintah/Grid/ParticleVariable.h>
#include <Uintah/Grid/CCVariable.h>
#include <Uintah/Grid/NCVariable.h>
#include <Uintah/Grid/VarLabel.h>
#include <Uintah/Grid/VarTypes.h>
#include <Uintah/Grid/PerPatch.h>
#include <SCICore/Malloc/Allocator.h>
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

// $Log$
// Revision 1.2  2000/08/09 03:17:58  jas
// Changed new to scinew and added deletes to some of the destructors.
//
// Revision 1.1  2000/06/10 04:10:46  tan
// Added MDLabel class.
//
