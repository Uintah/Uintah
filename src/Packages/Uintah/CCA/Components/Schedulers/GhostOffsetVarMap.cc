#include <Packages/Uintah/CCA/Components/Schedulers/GhostOffsetVarMap.h>

#include <iostream>
using namespace std;


using namespace Uintah;

void GhostOffsetVarMap::includeOffsets(const VarLabel* var,
					Ghost::GhostType gtype,
					int numGhostCells)
{
  Offsets& offsets = map_[var];
  IntVector lowOffset, highOffset;
  TypeDescription::Type varType = var->typeDescription()->getType();
  Patch::getGhostOffsets(varType, gtype, numGhostCells,
			 lowOffset, highOffset); 
  offsets.encompassOffsets(lowOffset, highOffset);
}

void GhostOffsetVarMap::getExtents(const VarLabel* var, const Patch* patch,
				   Ghost::GhostType gtype, int numGhostCells,
				   IntVector& lowIndex,
				   IntVector& highIndex) const
{
  IntVector lowOffset, highOffset;
  Patch::VariableBasis basis =
    Patch::translateTypeToBasis(var->typeDescription()->getType(), true);
  Offsets offsets; // defaults to (0,0,0), (0,0,0)
  Map::const_iterator foundIter = map_.find(var);
  if (foundIter != map_.end()) {
    offsets = (*foundIter).second;
  }

  Patch::getGhostOffsets(var->typeDescription()->getType(), gtype,
			 numGhostCells, lowOffset, highOffset);
  offsets.encompassOffsets(lowOffset, highOffset);
  
  offsets.getOffsets(lowOffset, highOffset);
  patch->computeExtents(basis, lowOffset, highOffset, lowIndex, highIndex);  
}

void GhostOffsetVarMap::Offsets::
encompassOffsets(IntVector lowOffset, IntVector highOffset)
{
  lowOffset_ = Max(lowOffset_, lowOffset); // low offset to be subtracted
  highOffset_ = Max(highOffset_, highOffset);
}

