
#include <Uintah/Components/ICE/ICE.h>
#include <Uintah/Interface/CFDInterface.h>
#include <Uintah/Grid/VarLabel.h>
#include <Uintah/Grid/CCVariable.h>
#include <SCICore/Geometry/Vector.h>
#include <Uintah/Parallel/ProcessorGroup.h>

using SCICore::Geometry::Vector;

using namespace Uintah;
using namespace Uintah::ICESpace;


ICE::ICE(const ProcessorGroup* myworld) 
  : UintahParallelComponent(myworld)
{
  lb = new ICELabel();

}

ICE::~ICE()
{
}

void ICE::problemSetup(const ProblemSpecP& prob_spec, GridP& grid,
		       SimulationStateP& sharedState)
{
}


//
// $Log$
// Revision 1.22  2000/10/04 20:15:27  jas
// Start to bring ICE into UCF.
//
