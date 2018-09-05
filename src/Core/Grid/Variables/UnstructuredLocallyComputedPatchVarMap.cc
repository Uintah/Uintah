/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#include <Core/Grid/Variables/UnstructuredLocallyComputedPatchVarMap.h>

#include <Core/Grid/UnstructuredGrid.h>
#include <Core/Grid/UnstructuredLevel.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Exceptions/InternalError.h>

#include <iostream>

using namespace Uintah;

class UnstructuredPatchRangeQuerier {

  public:

    typedef UnstructuredPatch::selectType ResultContainer;

    UnstructuredPatchRangeQuerier( const UnstructuredLevel*                  level,
                             std::set<const UnstructuredPatch*>& patches )
        : level_(level), patches_(patches)
    {
    }

    void query( IntVector        low,
                IntVector        high,
                ResultContainer& result );

    void queryNeighbors( IntVector        low,
                         IntVector        high,
                         ResultContainer& result) ;

  private:

    const UnstructuredLevel*                 level_;
          std::set<const UnstructuredPatch*>& patches_;
};

void
UnstructuredPatchRangeQuerier::query( IntVector        low,
                          IntVector        high,
                          ResultContainer& result )
{
  std::back_insert_iterator<ResultContainer> result_ii( result );

  ResultContainer tmp;
  //add the extra cells to the low and subtract them from the high
  //this adjusts for the extra cells that may have been included in the low and high
  //this assumes the minimum patch size is greater than 2*extra_cells
  IntVector offset = level_->getExtraCells();
  level_->selectPatches(low + offset, high - offset, tmp);

  for (ResultContainer::iterator iter = tmp.begin(); iter != tmp.end(); iter++) {
    if (patches_.find(*iter) != patches_.end())
      *result_ii++ = *iter;
  }
}

void
UnstructuredPatchRangeQuerier::queryNeighbors( IntVector                           low,
                                   IntVector                           high,
                                   UnstructuredPatchRangeQuerier::ResultContainer& result )
{
  std::back_insert_iterator<ResultContainer> result_ii(result);

  ResultContainer tmp;
  // query on each of 6 sides (done in pairs of opposite sides)
  for (int i = 0; i < 3; i++) {
    IntVector sideLow = low;
    IntVector sideHigh = high;
    sideHigh[i] = sideLow[i]--;
    tmp.resize(0);
    level_->selectPatches(sideLow, sideHigh, tmp);

    for (ResultContainer::iterator iter = tmp.begin(); iter != tmp.end(); iter++) {
      if (patches_.find(*iter) != patches_.end())
        *result_ii++ = *iter;
    }

    sideHigh = high;
    sideLow = low;
    sideLow[i] = sideHigh[i]++;
    tmp.resize(0);
    level_->selectPatches(sideLow, sideHigh, tmp);

    for (ResultContainer::iterator iter = tmp.begin(); iter != tmp.end(); iter++) {
      if (patches_.find(*iter) != patches_.end())
        *result_ii++ = *iter;
    }
  }
}

UnstructuredLocallyComputedPatchVarMap::UnstructuredLocallyComputedPatchVarMap()
{
  reset();
}

UnstructuredLocallyComputedPatchVarMap::~UnstructuredLocallyComputedPatchVarMap()
{
  reset();
}

void
UnstructuredLocallyComputedPatchVarMap::reset()
{
  groupsMade = false;
  for (unsigned i = 0; i < sets_.size(); i++) {
    delete sets_[i];
  }
  sets_.clear();
}

void
UnstructuredLocallyComputedPatchVarMap::addComputedPatchSet( const UnstructuredPatchSubset* patches )
{
  ASSERT(!groupsMade);
  if (!patches || !patches->size()) {
    return;  // don't worry about reduction variables
  }

  const UnstructuredLevel* level = patches->get(0)->getLevel();
#if SCI_ASSERTION_LEVEL >= 1
  // Each call to this should contain only one level (one level at a time)
  for (int i = 1; i < patches->size(); i++) {
    const UnstructuredPatch* patch = patches->get(i);
    ASSERT(patch->getLevel() == level);
  }
#endif

  if ((int)sets_.size() <= level->getIndex()) {
    sets_.resize(level->getIndex() + 1);
  }

  UnstructuredLocallyComputedPatchSet* lcpatches = sets_[level->getIndex()];
  if (lcpatches == 0) {
    lcpatches = scinew UnstructuredLocallyComputedPatchSet();
    sets_[level->getIndex()] = lcpatches;
  }
  lcpatches->addPatches(patches);
}

const SuperPatch*
UnstructuredLocallyComputedPatchVarMap::getConnectedPatchGroup( const UnstructuredPatch* patch ) const
{
  ASSERT(groupsMade);
  int l = patch->getLevel()->getIndex();
  if (sets_.size() == 0 || sets_[l] == 0) {
    return 0;
  }
  return sets_[l]->getConnectedPatchGroup(patch);
}

const SuperPatchContainer*
UnstructuredLocallyComputedPatchVarMap::getSuperPatches( const UnstructuredLevel* level ) const
{
  ASSERT(groupsMade);
  int l = level->getIndex();
  if (sets_.size() == 0 || sets_[l] == 0) {
    return 0;
  }
  return sets_[l]->getSuperPatches();
}

void UnstructuredLocallyComputedPatchVarMap::makeGroups()
{
  ASSERT(!groupsMade);
  for (unsigned l = 0; l < sets_.size(); l++)
    if (sets_[l]) {
      sets_[l]->makeGroups();
    }
  groupsMade = true;
}

UnstructuredLocallyComputedPatchVarMap::UnstructuredLocallyComputedPatchSet::UnstructuredLocallyComputedPatchSet()
{
  connectedPatchGroups_ = 0;
}

UnstructuredLocallyComputedPatchVarMap::UnstructuredLocallyComputedPatchSet::~UnstructuredLocallyComputedPatchSet()
{
  if (connectedPatchGroups_) {
    delete connectedPatchGroups_;
  }
}

void UnstructuredLocallyComputedPatchVarMap::UnstructuredLocallyComputedPatchSet::addPatches(const UnstructuredPatchSubset* patches)
{
  ASSERT(connectedPatchGroups_ == 0);
  for (int i = 0; i < patches->size(); i++) {
    if (map_.find(patches->get(i)) == map_.end()) {
      map_.insert(std::make_pair(patches->get(i), static_cast<SuperPatch*>(0)));
    }
  }
}

const SuperPatchContainer*
UnstructuredLocallyComputedPatchVarMap::UnstructuredLocallyComputedPatchSet::getSuperPatches() const
{
  ASSERT(connectedPatchGroups_ != 0);
  return &connectedPatchGroups_->getSuperBoxes();
}

const SuperPatch*
UnstructuredLocallyComputedPatchVarMap::UnstructuredLocallyComputedPatchSet::getConnectedPatchGroup( const UnstructuredPatch* patch ) const
{
  ASSERT(connectedPatchGroups_ != 0);
  UnstructuredPatchMapType::const_iterator iter = map_.find(patch);
  if (iter == map_.end())
    return 0;
  return iter->second;
}

void
UnstructuredLocallyComputedPatchVarMap::UnstructuredLocallyComputedPatchSet::makeGroups()
{
  ASSERT(connectedPatchGroups_ == 0);
  // Need to copy the patch list into a vector (or a set, but a
  // vector would do), since the grouper cannot deal with a map
  // We know that it is a unique list, because it is a map
  std::set<const UnstructuredPatch*> patches;
  for (UnstructuredPatchMapType::iterator iter = map_.begin(); iter != map_.end(); ++iter) {
    patches.insert(iter->first);
  }

  ASSERT(patches.begin() != patches.end());
  const UnstructuredLevel* level = (*patches.begin())->getLevel();
#if SCI_ASSERTION_LEVEL >= 1
  for (std::set<const UnstructuredPatch*>::iterator iter = patches.begin(); iter != patches.end(); iter++) {
    ASSERT((*iter)->getLevel() == level);
  }
#endif

  UnstructuredPatchRangeQuerier patchRangeQuerier(level, patches);
  connectedPatchGroups_ = SuperPatchSet::makeNearOptimalSuperBoxSet(patches.begin(), patches.end(), patchRangeQuerier);

//  std::cerr << "ConnectedPatchGroups: \n" << *connectedPatchGroups_ << "\n";

  // map each patch to its SuperBox
  const SuperPatchContainer& superBoxes = connectedPatchGroups_->getSuperBoxes();
  SuperPatchContainer::const_iterator iter;
  for (iter = superBoxes.begin(); iter != superBoxes.end(); iter++) {
    const SuperPatch* superBox = *iter;
    std::vector<const UnstructuredPatch*>::const_iterator SBiter;
    for (SBiter = superBox->getBoxes().begin(); SBiter != superBox->getBoxes().end(); SBiter++) {
      map_[*SBiter] = superBox;
    }
  }
#if SCI_ASSERTION_LEVEL >= 1
  for (UnstructuredPatchMapType::iterator iter = map_.begin(); iter != map_.end(); ++iter) {
    ASSERT(iter->second != 0);
  }
#endif
}
