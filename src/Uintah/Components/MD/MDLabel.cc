/* REFERENCED */
static char *id="@(#) $Id$";

#include <Uintah/Components/MD/MDLabel.h>
#include <Uintah/Grid/ParticleVariable.h>
#include <Uintah/Grid/CCVariable.h>
#include <Uintah/Grid/NCVariable.h>
#include <Uintah/Grid/VarLabel.h>
#include <Uintah/Grid/VarTypes.h>
#include <Uintah/Grid/PerPatch.h>

using namespace Uintah;
using namespace Uintah::MD;

MDLabel::MDLabel()
{
  // Particle Variables

  atomMassLabel = new VarLabel("atom.mass",
			ParticleVariable<double>::getTypeDescription() );
  
  atomXLabel = new VarLabel( "atom.x", ParticleVariable<Point>::getTypeDescription(),
			VarLabel::PositionVariable);

  // Reduction variables

  delTLabel = new VarLabel( "delT", delt_vartype::getTypeDescription() );
  
} 

const MDLabel* MDLabel::getLabels()
{
  static MDLabel* instance=0;
  if(!instance)
    instance=new MDLabel();
  return instance;
}

// $Log$
// Revision 1.1  2000/06/10 04:10:46  tan
// Added MDLabel class.
//
