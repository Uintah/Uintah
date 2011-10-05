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


#include <Core/Grid/Variables/LocallyComputedPatchVarMap.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/Level.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Exceptions/InternalError.h>
#include <iostream>

using namespace std;
using namespace SCIRun;
using namespace Uintah;

class PatchRangeQuerier
{
public:
  typedef Patch::selectType ResultContainer;
public:
  PatchRangeQuerier(const Level* level, set<const Patch*>& patches)
    : level_(level), patches_(patches) {}
  
  void query(IntVector low, IntVector high, ResultContainer& result);

  void queryNeighbors(IntVector low, IntVector high, ResultContainer& result);
private:
  const Level* level_;
  set<const Patch*>& patches_;
};

void PatchRangeQuerier::query(IntVector low, IntVector high,
                              ResultContainer& result)
{
  back_insert_iterator<ResultContainer> result_ii(result);
  
  ResultContainer tmp;
  //add the extra cells to the low and subtract them from the high
    //this adjusts for the extra cells that may have been included in the low and high
    //this assumes the minimum patch size is greater than 2*extra_cells
  IntVector offset=level_->getExtraCells();
  level_->selectPatches(low+offset, high-offset, tmp);  
  
  for(ResultContainer::iterator iter = tmp.begin(); iter != tmp.end(); iter++){
    if(patches_.find(*iter) != patches_.end())
      *result_ii++ = *iter;
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
    for(ResultContainer::iterator iter = tmp.begin(); iter != tmp.end(); iter++){
      if(patches_.find(*iter) != patches_.end())
        *result_ii++ = *iter;
    }

    sideHigh = high;
    sideLow = low;
    sideLow[i] = sideHigh[i]++;
    tmp.resize(0);
    level_->selectPatches(sideLow, sideHigh, tmp);
    for(ResultContainer::iterator iter = tmp.begin(); iter != tmp.end(); iter++){
      if(patches_.find(*iter) != patches_.end())
        *result_ii++ = *iter;
    }
  }
}

LocallyComputedPatchVarMap::LocallyComputedPatchVarMap()
{
  reset();
}

LocallyComputedPatchVarMap::~LocallyComputedPatchVarMap()
{
  reset();
}

void
LocallyComputedPatchVarMap::reset()
{
  groupsMade=false;
  for (unsigned i = 0; i < sets_.size(); i++)
    delete sets_[i];
  sets_.clear();
}

void
LocallyComputedPatchVarMap::addComputedPatchSet(const PatchSubset* patches)
{
  ASSERT(!groupsMade);
  if (!patches || !patches->size())
    return; // don't worry about reduction variables

  const Level* level = patches->get(0)->getLevel();
#if SCI_ASSERTION_LEVEL >= 1
  // Each call to this should contain only one level (one level at a time)
  for(int i=1;i<patches->size();i++){
    const Patch* patch = patches->get(i);
    ASSERT(patch->getLevel() == level);
  }
#endif

  if ((int) sets_.size() <= level->getIndex())
    sets_.resize(level->getIndex()+1);

  LocallyComputedPatchSet* lcpatches = sets_[level->getIndex()];
  if(lcpatches == 0){
    lcpatches = scinew LocallyComputedPatchSet();
    sets_[level->getIndex()] = lcpatches;
  }
  lcpatches->addPatches(patches);
}

const SuperPatch*
LocallyComputedPatchVarMap::getConnectedPatchGroup(const Patch* patch) const
{
  ASSERT(groupsMade);
  int l = patch->getLevel()->getIndex();
  if (sets_.size() == 0 || sets_[l] == 0)
    return 0;
  return sets_[l]->getConnectedPatchGroup(patch);
}

const SuperPatchContainer*
LocallyComputedPatchVarMap::getSuperPatches(const Level* level) const
{
  ASSERT(groupsMade);
  int l = level->getIndex();
  if (sets_.size() == 0 || sets_[l] == 0) {
    return 0;
  }
  return sets_[l]->getSuperPatches();
}

void LocallyComputedPatchVarMap::makeGroups()
{
  ASSERT(!groupsMade);
  for (unsigned l = 0; l < sets_.size(); l++)
    if (sets_[l]) {
      sets_[l]->makeGroups();
    }
  groupsMade=true;
}

LocallyComputedPatchVarMap::LocallyComputedPatchSet::LocallyComputedPatchSet()
{
  connectedPatchGroups_=0;
}

LocallyComputedPatchVarMap::LocallyComputedPatchSet::~LocallyComputedPatchSet()
{
  if(connectedPatchGroups_)
    delete connectedPatchGroups_;
}

void
LocallyComputedPatchVarMap::LocallyComputedPatchSet::addPatches(const PatchSubset* patches)
{
  ASSERT(connectedPatchGroups_ == 0);
  for (int i = 0; i < patches->size(); i++){
    if(map_.find(patches->get(i)) == map_.end())
      map_.insert(make_pair(patches->get(i), static_cast<SuperPatch*>(0)));
  }
}

const SuperPatchContainer*
LocallyComputedPatchVarMap::LocallyComputedPatchSet::getSuperPatches() const
{
  ASSERT(connectedPatchGroups_ != 0);
  return &connectedPatchGroups_->getSuperBoxes();
}

const SuperPatch*
LocallyComputedPatchVarMap::LocallyComputedPatchSet::getConnectedPatchGroup(const Patch* patch) const
{
  ASSERT(connectedPatchGroups_ != 0);
  PatchMapType::const_iterator iter = map_.find(patch);
  if(iter == map_.end())
    return 0;
  return iter->second;
}

void
LocallyComputedPatchVarMap::LocallyComputedPatchSet::makeGroups()
{
  ASSERT(connectedPatchGroups_ == 0);
  // Need to copy the patch list into a vector (or a set, but a
  // vector would do), since the grouper cannot deal with a map
  // We know that it is a unique list, because it is a map
  set<const Patch*> patches;
  for(PatchMapType::iterator iter = map_.begin(); iter != map_.end(); ++iter)
    patches.insert(iter->first);

  ASSERT(patches.begin() != patches.end());
  const Level* level = (*patches.begin())->getLevel();
#if SCI_ASSERTION_LEVEL >= 1
  for(set<const Patch*>::iterator iter = patches.begin(); iter != patches.end(); iter++){
    ASSERT((*iter)->getLevel() == level);
  }
#endif

  PatchRangeQuerier patchRangeQuerier(level, patches);
  connectedPatchGroups_ =
    SuperPatchSet::makeNearOptimalSuperBoxSet(patches.begin(),
                                              patches.end(),
                                              patchRangeQuerier);
  //cerr << "ConnectedPatchGroups: \n";
  //cerr << *connectedPatchGroups_ << "\n";

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
#if SCI_ASSERTION_LEVEL >= 1
  for(PatchMapType::iterator iter = map_.begin(); iter != map_.end(); ++iter)
    ASSERT(iter->second != 0);
#endif
}
