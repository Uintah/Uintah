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


//  
//    File   : LocallyComputedPatchVarMap.h
//    Author : Wayne Witzel
//    Date   : Mon Jan 28 17:40:35 2002

#ifndef UNSTRUCTUREDLOCALLYCOMPUTEDPATCHVARMAP
#define UNSTRUCTUREDLOCALLYCOMPUTEDPATCHVARMAP

#include <Core/Grid/UnstructuredPatch.h>
#include <Core/Grid/UnstructuredLevel.h>
#include <Core/Containers/UnstructuredSuperBox.h>

#include <map>

namespace Uintah {

  inline int getVolume(const IntVector& low, const IntVector& high)
  { return UnstructuredPatch::getVolume(low, high); }

  typedef Uintah::SuperBox<const UnstructuredPatch*, IntVector, int, int,
    Uintah::InternalAreaSuperBoxEvaluator<const UnstructuredPatch*, int> > SuperPatch;

  typedef Uintah::SuperBoxSet<const UnstructuredPatch*, IntVector, int, int,
    Uintah::InternalAreaSuperBoxEvaluator<const UnstructuredPatch*, int> > SuperPatchSet;
  typedef SuperPatchSet::SuperBoxContainer SuperPatchContainer;

  class UnstructuredLocallyComputedPatchVarMap {
  public:
    UnstructuredLocallyComputedPatchVarMap();
    ~UnstructuredLocallyComputedPatchVarMap();

    void reset();
    void addComputedPatchSet(const UnstructuredPatchSubset* patches);

    const SuperPatch* getConnectedPatchGroup(const UnstructuredPatch* patch) const;
    const SuperPatchContainer* getSuperPatches(const UnstructuredLevel* level) const;
    void makeGroups();

    class UnstructuredLocallyComputedPatchSet {
    public:
      UnstructuredLocallyComputedPatchSet();
      ~UnstructuredLocallyComputedPatchSet();
      void addPatches(const UnstructuredPatchSubset* patches);
      const SuperPatch* getConnectedPatchGroup(const UnstructuredPatch* patch) const;
      const SuperPatchContainer* getSuperPatches() const;
      void makeGroups();
    private:
      typedef std::map<const UnstructuredPatch*, const SuperPatch*> UnstructuredPatchMapType;
      UnstructuredPatchMapType map_;
      SuperPatchSet* connectedPatchGroups_;
    };
  private:

    typedef std::vector<UnstructuredLocallyComputedPatchSet*> DataType;
    DataType sets_;
    bool groupsMade;
  };

} // End namespace Uintah

#endif
