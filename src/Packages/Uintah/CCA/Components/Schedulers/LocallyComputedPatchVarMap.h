//  The contents of this file are subject to the University of Utah Public
//  License (the "License"); you may not use this file except in compliance
//  with the License.
//  
//  Software distributed under the License is distributed on an "AS IS"
//  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
//  License for the specific language governing rights and limitations under
//  the License.
//  
//  The Original Source Code is SCIRun, released March 12, 2001.
//  
//  The Original Source Code was developed by the University of Utah.
//  Portions created by UNIVERSITY are Copyright (C) 2001, 1994
//  University of Utah. All Rights Reserved.
//  
//    File   : LocallyComputedPatchVarMap.h
//    Author : Wayne Witzel
//    Date   : Mon Jan 28 17:40:35 2002

#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Core/Containers/SuperBox.h>

namespace Uintah {

using namespace SCIRun;

inline int getVolume(IntVector low, IntVector high)
{ return Patch::getVolume(low, high); }

typedef SuperBox<const Patch*, IntVector, int, int,
  InternalAreaSuperBoxEvaluator<const Patch*, int> > SuperPatch;

typedef SuperBoxSet<const Patch*, IntVector, int, int,
  InternalAreaSuperBoxEvaluator<const Patch*, int> > SuperPatchSet;
typedef SuperPatchSet::SuperBoxContainer SuperPatchContainer;

class LocallyComputedPatchVarMap
{
public:
  LocallyComputedPatchVarMap() {}
  ~LocallyComputedPatchVarMap();

  void addComputedPatchSet(const VarLabel* label, const PatchSubset* patches);

  const SuperPatch* getConnectedPatchGroup(const VarLabel* label,
					   const Patch* patch) const;

  const SuperPatchContainer* getSuperPatches(const VarLabel* label) const;

public:
  class ConnectedPatchGrouper
  {
  public:
    ConnectedPatchGrouper(const set<const Patch*>& patchSet);
    ~ConnectedPatchGrouper()
    { delete connectedPatchGroups_; }

    const SuperPatch* getConnectedPatchGroup(const Patch* patch) const
    {
      map<const Patch*, const SuperPatch*>::const_iterator findIt =
	map_.find(patch);
      return (findIt != map_.end()) ? findIt->second : 0;
    }

    const SuperPatchContainer& getSuperPatches() const
    { return connectedPatchGroups_->getSuperBoxes(); }
  private:
    SuperPatchSet* connectedPatchGroups_;
    map<const Patch*, const SuperPatch*> map_;
  };

private:
  ConnectedPatchGrouper* getConnectedPatchGrouper(const VarLabel* label) const;
  
  typedef map< set<const Patch*>, ConnectedPatchGrouper* >
  ConnectedPatchGrouperMap;

  // mutable because it creates ConnectedPatchGroupMap's as needed.
  mutable ConnectedPatchGrouperMap connectedPatchGroupers_;
  
  typedef map<const VarLabel*, const set<const Patch*>*, VarLabel::Compare >
  Map;
  Map map_;

  static SuperPatchContainer emptySuperPatchContainer;
};

} // End namespace Uintah
