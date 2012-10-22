/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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


//  
//    File   : LocallyComputedPatchVarMap.h
//    Author : Wayne Witzel
//    Date   : Mon Jan 28 17:40:35 2002

#ifndef LOCALLYCOMPUTEDPATCHVARMAP
#define LOCALLYCOMPUTEDPATCHVARMAP

#include <Core/Grid/Patch.h>
#include <Core/Grid/Level.h>
#include <Core/Containers/SuperBox.h>

#include <map>

namespace Uintah {

  inline int getVolume(const IntVector& low, const IntVector& high)
  { return Patch::getVolume(low, high); }

  typedef SCIRun::SuperBox<const Patch*, IntVector, int, int,
    SCIRun::InternalAreaSuperBoxEvaluator<const Patch*, int> > SuperPatch;

  typedef SCIRun::SuperBoxSet<const Patch*, IntVector, int, int,
    SCIRun::InternalAreaSuperBoxEvaluator<const Patch*, int> > SuperPatchSet;
  typedef SuperPatchSet::SuperBoxContainer SuperPatchContainer;

  class LocallyComputedPatchVarMap {
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
