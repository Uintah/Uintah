#include <Packages/Uintah/CCA/Components/Schedulers/LocallyComputedPatchVarMap.h>
#include <Core/Exceptions/InternalError.h>

#include <iostream>
using namespace std;

using namespace Uintah;

SuperPatchContainer LocallyComputedPatchVarMap::emptySuperPatchContainer;

class PatchRangeQuerier
{
public:
  typedef Level::selectType ResultContainer;
public:
  PatchRangeQuerier(const Level* level)
    : level_(level) {}
  
  inline void query(IntVector low, IntVector high, ResultContainer& result)
  {
    level_->selectPatches(low, high, result);
  }

  void queryNeighbors(IntVector low, IntVector high, ResultContainer& result);
private:
  const Level* level_;
};

void LocallyComputedPatchVarMap::
addComputedPatchSet(const VarLabel* label,
		    const PatchSubset* patches)
{
  if (patches == 0)
    return; // don't worry about reduction variables

  set<const Patch*> patchSet;  
  if (map_.find(label) != map_.end()) {
    patchSet = *map_[label];
  }

  for (int i = 0; i < patches->size(); i++) {
    patchSet.insert((*patches)[i]);
  }

  ConnectedPatchGrouper* nullGrouper = 0;
  ConnectedPatchGrouperMap::iterator insertedIter =
    connectedPatchGroupers_.insert(make_pair(patchSet, nullGrouper)).first;
  map_[label] = &((*insertedIter).first);
}

LocallyComputedPatchVarMap::ConnectedPatchGrouper* LocallyComputedPatchVarMap::
getConnectedPatchGrouper(const VarLabel* label) const
{
  Map::const_iterator foundIter = map_.find(label);
  if (foundIter == map_.end())
    return 0;
  const set<const Patch*>& patchSet = *(*foundIter).second;
  ConnectedPatchGrouper*& connectedPatchGrouper =
    connectedPatchGroupers_[patchSet];
  if (connectedPatchGrouper != 0)
    return connectedPatchGrouper;
  else
    return connectedPatchGrouper = scinew ConnectedPatchGrouper(patchSet);
}

const SuperPatch*
LocallyComputedPatchVarMap::getConnectedPatchGroup(const VarLabel* label,
						   const Patch* patch) const
{
  ConnectedPatchGrouper* connectedPatchGrouper =
    getConnectedPatchGrouper(label);
  return (connectedPatchGrouper == 0) ? 0 :
    connectedPatchGrouper->getConnectedPatchGroup(patch);
}

const SuperPatchContainer*
LocallyComputedPatchVarMap::getSuperPatches(const VarLabel* label) const
{
  ConnectedPatchGrouper* connectedPatchGrouper =
    getConnectedPatchGrouper(label);
  return (connectedPatchGrouper == 0) ? 0 :
    &connectedPatchGrouper->getSuperPatches();
}

LocallyComputedPatchVarMap::~LocallyComputedPatchVarMap()
{
  ConnectedPatchGrouperMap::iterator iter;
  for (iter = connectedPatchGroupers_.begin();
       iter != connectedPatchGroupers_.end(); iter++) {
    delete iter->second;
  }
}

LocallyComputedPatchVarMap::ConnectedPatchGrouper::
ConnectedPatchGrouper(const set<const Patch*>& patchSet)
{
  if (patchSet.size() < 1)
    throw InternalError("LocallyComputedPatchVarMap::ConnectedPatchGroupMap::ConnectedPatchGroupMap, empty patch set");
  const Level* level = (*patchSet.begin())->getLevel();

  if (patchSet.size() > 1)
    cerr << "patchSet size: " << patchSet.size() << endl;
  
  PatchRangeQuerier patchRangeQuerier(level);
  connectedPatchGroups_ =
    SuperPatchSet::makeNearOptimalSuperBoxSet(patchSet.begin(),
					      patchSet.end(),
					      patchRangeQuerier);

  //cerr << "ConnectedPatchGroups: " << endl;
  //cerr << *connectedPatchGroups_ << endl;
  // map each patch to its SuperBox
  const SuperPatchContainer& superBoxes =
    connectedPatchGroups_->getSuperBoxes();
  SuperPatchContainer::const_iterator iter;
  for (iter = superBoxes.begin(); iter != superBoxes.end(); iter++) {
    const SuperPatch* superBox = *iter;
    vector<const Patch*>::const_iterator SBiter;
    for (SBiter = superBox->getBoxes().begin();
	 SBiter != superBox->getBoxes().end(); SBiter++) {
      map_[*SBiter] = superBox;
    }
  }
}

void
PatchRangeQuerier::queryNeighbors(IntVector low, IntVector high,
				  PatchRangeQuerier::ResultContainer& result)
{
  back_insert_iterator<ResultContainer> result_ii(result);
  
  ResultContainer tmp;
  // query on each of 6 sides (done in pairs of opposite sides)
  for (int i = 0; i < 3; i++) {
    IntVector sideLow = low;
    IntVector sideHigh = high;
    sideHigh[i] = sideLow[i]--;
    tmp.resize(0);
    level_->selectPatches(sideLow, sideHigh, tmp);
    copy(tmp.begin(), tmp.end(), result_ii);

    sideHigh = high;
    sideLow = low;
    sideLow[i] = sideHigh[i]++;
    tmp.resize(0);
    level_->selectPatches(sideLow, sideHigh, tmp);
    copy(tmp.begin(), tmp.end(), result_ii);
  }
}
