#include <Packages/Uintah/CCA/Components/Schedulers/GhostOffsetVarMap.h>
#include <Packages/Uintah/Core/Grid/ComputeSet.h>

#include <iostream>

using namespace std;
using namespace Uintah;
using namespace SCIRun;

void GhostOffsetVarMap::includeOffsets(const VarLabel* var,
				       const MaterialSubset* matls,
				       const PatchSubset* patches,
				       Ghost::GhostType gtype,
				       int numGhostCells)
{
  IntVector lowOffset, highOffset;
  TypeDescription::Type varType = var->typeDescription()->getType();
  Patch::getGhostOffsets(varType, gtype, numGhostCells,
			 lowOffset, highOffset);
  for (int p = 0; p < patches->size(); p++) {
    const Patch* patch = patches->get(p);
    for (int m = 0; m < matls->size(); m++) {
      VarLabelMatlPatch vmp(var, matls->get(m), patch);
      Offsets& offsets = map_[vmp];  
      offsets.encompassOffsets(lowOffset, highOffset);
    }
  }
}

void GhostOffsetVarMap::
getExtents(const VarLabelMatlPatch& vmp,
	   Ghost::GhostType requestedGType, int requestedNumGhostCells,
	   IntVector& requiredLow, IntVector& requiredHigh,	   
	   IntVector& requestedLow, IntVector& requestedHigh) const
{
  IntVector lowOffset, highOffset;
  const VarLabel* var = vmp.label_;
  const Patch* patch = vmp.patch_;  
  Patch::VariableBasis basis =
    Patch::translateTypeToBasis(var->typeDescription()->getType(), true);
  Offsets offsets; // defaults to (0,0,0), (0,0,0)
  Map::const_iterator foundIter = map_.find(vmp);
  if (foundIter != map_.end()) {
    offsets = (*foundIter).second;
  }

  // before taking the requested ghost cells into consideration
  // (just those required by later tasks).
  offsets.getOffsets(lowOffset, highOffset);
  patch->computeExtents(basis, var->getBoundaryLayer(), lowOffset, highOffset,
			requiredLow, requiredHigh);  

  Patch::getGhostOffsets(var->typeDescription()->getType(),
			 requestedGType, requestedNumGhostCells,
			 lowOffset, highOffset);
  offsets.encompassOffsets(lowOffset, highOffset);

  // after taking the requested ghost cells into consideratio n 
  offsets.getOffsets(lowOffset, highOffset);
  patch->computeExtents(basis, var->getBoundaryLayer(), lowOffset, highOffset,
			requestedLow, requestedHigh);  
}

void GhostOffsetVarMap::Offsets::
encompassOffsets(IntVector lowOffset, IntVector highOffset)
{
  lowOffset_ = Max(lowOffset_, lowOffset); // low offset to be subtracted
  highOffset_ = Max(highOffset_, highOffset);
}

