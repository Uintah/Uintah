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

#ifndef LOCALLYCOMPUTEDPATCHVARMAP
#define LOCALLYCOMPUTEDPATCHVARMAP

#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Core/Containers/SuperBox.h>

#include <sgi_stl_warnings_off.h>
#include <map>
#include <sgi_stl_warnings_on.h>

#include <Packages/Uintah/Core/Grid/share.h>
namespace Uintah {

  inline int getVolume(const IntVector& low, const IntVector& high)
  { return Patch::getVolume(low, high); }

  typedef SCIRun::SuperBox<const Patch*, IntVector, int, int,
    SCIRun::InternalAreaSuperBoxEvaluator<const Patch*, int> > SuperPatch;

  typedef SCIRun::SuperBoxSet<const Patch*, IntVector, int, int,
    SCIRun::InternalAreaSuperBoxEvaluator<const Patch*, int> > SuperPatchSet;
  typedef SuperPatchSet::SuperBoxContainer SuperPatchContainer;

  class SCISHARE LocallyComputedPatchVarMap {
  public:
    LocallyComputedPatchVarMap();
    ~LocallyComputedPatchVarMap();

    void reset();
    void addComputedPatchSet(const PatchSubset* patches);

    const SuperPatch* getConnectedPatchGroup(const Patch* patch) const;
    const SuperPatchContainer* getSuperPatches(const Level* level) const;
    void makeGroups();

    class LocallyComputedPatchSet {
    public:
      LocallyComputedPatchSet();
      ~LocallyComputedPatchSet();
      void addPatches(const PatchSubset* patches);
      const SuperPatch* getConnectedPatchGroup(const Patch* patch) const;
      const SuperPatchContainer* getSuperPatches() const;
      void makeGroups();
    private:
      typedef std::map<const Patch*, const SuperPatch*> PatchMapType;
      PatchMapType map_;
      SuperPatchSet* connectedPatchGroups_;
    };
  private:

    typedef std::vector<LocallyComputedPatchSet*> DataType;
    DataType sets_;
    bool groupsMade;
  };

} // End namespace Uintah

#endif
