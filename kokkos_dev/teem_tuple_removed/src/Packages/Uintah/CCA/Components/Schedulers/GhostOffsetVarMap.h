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
//    File   : GhostOffsetVarMap.h
//    Author : Wayne Witzel
//    Date   : Fri May 18 14:36:19 2001

#include <Packages/Uintah/Core/Grid/Patch.h>
#include <Packages/Uintah/Core/Grid/ComputeSet.h>
#include <Packages/Uintah/Core/Grid/VarLabelMatlPatch.h>
#include <Packages/Uintah/Core/Grid/Ghost.h>
#include <map>

namespace Uintah {

class GhostOffsetVarMap
{
public:
  GhostOffsetVarMap() {}

  void clear()
  { map_.erase(map_.begin(), map_.end()); }
  
  void includeOffsets(const VarLabel* var,
		      const MaterialSubset* matls, const PatchSubset* patches,
		      Ghost::GhostType gtype, int numGhostCells);

  // requiredLow and requiredHigh will be the extents that cover the
  // expected ghost cells from all requires.  requestedLow and requestedHigh
  // will cover that as well as the requested ghost cells (which may be more
  // than what will be required by future tasks).
  void getExtents(const VarLabelMatlPatch& vmp,
		  Ghost::GhostType requestedGType, int requestedNumGhostCells,
		  IntVector& requiredLow, IntVector& requiredHigh,
		  IntVector& requestedLow, IntVector& requestedHigh) const;
private:
  // Note:  The offsets should be >= 0 in each dimension.  You should subtract,
  // not add, the low offset.
  class Offsets
  {
  public:
    Offsets()
      : lowOffset_(0,0,0), highOffset_(0,0,0) {}

    Offsets& operator=(const Offsets& other)
    {
      lowOffset_ = other.lowOffset_; highOffset_ = other.highOffset_;
      return *this;
    }

    bool operator==(const Offsets& other) const
    { return lowOffset_ == other.lowOffset_ &&
	highOffset_ == other.highOffset_; }

    void encompassOffsets(IntVector lowOffset, IntVector highOffset);

    void getOffsets(IntVector& lowOffset, IntVector& highOffset) const
    { lowOffset = lowOffset_; highOffset = highOffset_; }
  private:
    IntVector lowOffset_;
    IntVector highOffset_;
  };

  typedef std::map<VarLabelMatlPatch, Offsets> Map;
  Map map_;
};

} // End namespace Uintah
