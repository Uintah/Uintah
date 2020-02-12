/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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

// The purpose of this is to provide tools to allow better tracking of what will be queued
// into the GPU, and to manage that queued process.  For example, the engine may see that
// for variable phi, patch 2, material 0, level 0, that it needs to send the CPU var data
// into the GPU.  This helps collect this and all other forthcoming copies.

#ifndef CCA_COMPONENTS_SCHEDULERS_GPUGRIDVARIABLEINFO_H
#define CCA_COMPONENTS_SCHEDULERS_GPUGRIDVARIABLEINFO_H

#include <CCA/Components/Schedulers/GPUDataWarehouse.h>

#include <Core/Datatypes/TypeName.h>
#include <Core/Disclosure/TypeDescription.h>
#include <Core/Grid/Patch.h>
#include <Core/Grid/Task.h>
#include <Core/Grid/Variables/GridVariableBase.h>

#include <map>
#include <vector>

using namespace Uintah;

class GpuUtilities {

public:

  struct LabelPatchMatlLevelDw {

    LabelPatchMatlLevelDw( const char * label
                         ,       int    patchID
                         ,       int    matlIndx
                         ,       int    levelIndx
                         ,       int    dataWarehouse
                         )
      : m_label{label}
      , m_patchID{patchID}
      , m_matlIndx{matlIndx}
      , m_levelIndx{levelIndx}
      , m_dataWarehouse{dataWarehouse}
    {}

    // this is so it can be used in an STL map
    bool operator<(const LabelPatchMatlLevelDw& right) const
    {
      if (m_label < right.m_label) {
        return true;
      }
      else if (m_label == right.m_label && (m_patchID < right.m_patchID)) {
        return true;
      }
      else if (m_label == right.m_label && (m_patchID == right.m_patchID) && (m_matlIndx < right.m_matlIndx)) {
        return true;
      }
      else if (m_label == right.m_label && (m_patchID == right.m_patchID) && (m_matlIndx == right.m_matlIndx)
               && (m_levelIndx < right.m_levelIndx)) {
        return true;
      }
      else if (m_label == right.m_label && (m_patchID == right.m_patchID) && (m_matlIndx == right.m_matlIndx)
               && (m_levelIndx == right.m_levelIndx) && (m_dataWarehouse < right.m_dataWarehouse)) {
        return true;
      }
      else {
        return false;
      }
    }

    std::string m_label;
    int         m_patchID;
    int         m_matlIndx;
    int         m_levelIndx;
    int         m_dataWarehouse;
  };

  struct GhostVarsTuple {

    GhostVarsTuple( std::string label
                  , int         matlIndx
                  , int         levelIndx
                  , int         fromPatch
                  , int         toPatch
                  , int         dataWarehouse
                  , IntVector   sharedLowCoordinates
                  , IntVector   sharedHighCoordinates
                  )
      : m_label{label}
      , m_matlIndx{matlIndx}
      , m_levelIndx{levelIndx}
      , m_dataWarehouse{dataWarehouse}
      , m_fromPatch{fromPatch}
      , m_toPatch{toPatch}
      , m_sharedLowCoordinates{sharedLowCoordinates}
      , m_sharedHighCoordinates{sharedHighCoordinates}
    {}

    //This is so it can be used in an STL map
    bool operator<(const GhostVarsTuple& right) const
    {
      if (m_label < right.m_label) {
        return true;
      }
      else if (m_label == right.m_label && (m_matlIndx < right.m_matlIndx)) {
        return true;
      }
      else if (m_label == right.m_label && (m_matlIndx == right.m_matlIndx) && (m_levelIndx < right.m_levelIndx)) {
        return true;
      }
      else if (m_label == right.m_label && (m_matlIndx == right.m_matlIndx) && (m_levelIndx == right.m_levelIndx)
               && (m_fromPatch < right.m_fromPatch)) {
        return true;
      }
      else if (m_label == right.m_label && (m_matlIndx == right.m_matlIndx) && (m_levelIndx == right.m_levelIndx)
               && (m_fromPatch == right.m_fromPatch) && (m_toPatch < right.m_toPatch)) {
        return true;
      }
      else if (m_label == right.m_label && (m_matlIndx == right.m_matlIndx) && (m_levelIndx == right.m_levelIndx)
               && (m_fromPatch == right.m_fromPatch) && (m_toPatch == right.m_toPatch)
               && (m_dataWarehouse < right.m_dataWarehouse)) {
        return true;
      }
      else if (m_label == right.m_label && (m_matlIndx == right.m_matlIndx) && (m_levelIndx == right.m_levelIndx)
               && (m_fromPatch == right.m_fromPatch) && (m_toPatch == right.m_toPatch)
               && (m_dataWarehouse == right.m_dataWarehouse) && (m_sharedLowCoordinates < right.m_sharedLowCoordinates)) {
        return true;
      }
      else if (m_label == right.m_label && (m_matlIndx == right.m_matlIndx) && (m_levelIndx == right.m_levelIndx)
               && (m_fromPatch == right.m_fromPatch) && (m_toPatch == right.m_toPatch)
               && (m_dataWarehouse == right.m_dataWarehouse) && (m_sharedLowCoordinates == right.m_sharedLowCoordinates)
               && (m_sharedHighCoordinates < right.m_sharedHighCoordinates)) {
        return true;
      }
      else {
        return false;
      }
    }

    std::string m_label;
    int         m_matlIndx;
    int         m_levelIndx;
    int         m_dataWarehouse;
    int         m_fromPatch;
    int         m_toPatch;
    IntVector   m_sharedLowCoordinates;
    IntVector m_sharedHighCoordinates;
  };

  enum DeviceVarDestination {
      sameDeviceSameMpiRank    = 0
    , anotherDeviceSameMpiRank = 1
    , anotherMpiRank           = 2
    , unknown                  = 3
  };

  static void assignPatchesToGpus(const GridP& grid);

  static int getGpuIndexForPatch(const Patch* patch);

};


class DeviceGridVariableInfo {

using DeviceVarDest = GpuUtilities::DeviceVarDestination;

public:

  DeviceGridVariableInfo() = default;

  DeviceGridVariableInfo(       Variable          * var
                        ,       DeviceVarDest       dest
                        ,       bool                staging
                        ,       IntVector           sizeVector
                        ,       size_t              sizeOfDataType
                        ,       size_t              varMemSize
                        ,       IntVector           offset
                        ,       int                 matlIndx
                        ,       int                 levelIndx
                        , const Patch             * patchPointer
                        , const Task::Dependency  * dep
                        ,       Ghost::GhostType    gtype
                        ,       int                 numGhostCells
                        ,       unsigned int        whichGPU
                        );

  // For PerPatch vars
  DeviceGridVariableInfo(       Variable         * var
                        ,       DeviceVarDest      dest
                        ,       bool               staging
                        ,       size_t             sizeOfDataType
                        ,       size_t             varMemSize
                        ,       int                matlIndx
                        ,       int                levelIndx
                        , const Patch            * patchPointer
                        , const Task::Dependency * dep
                        ,       unsigned int       whichGPU
                        );

  bool operator==( DeviceGridVariableInfo& rhs )
  {
    return (m_sizeVector == rhs.m_sizeVector
           && m_sizeOfDataType == rhs.m_sizeOfDataType
           && m_varMemSize == rhs.m_varMemSize
           && m_offset == rhs.m_offset
           && m_matlIndx == rhs.m_matlIndx
           && m_levelIndx == rhs.m_levelIndx
           && m_patchPointer == rhs.m_patchPointer
           && m_gtype == rhs.m_gtype
           && m_numGhostCells == rhs.m_numGhostCells
           && m_whichGPU == rhs.m_whichGPU
           && m_dest == rhs.m_dest
           && m_staging == rhs.m_staging);
  }


  Variable               * m_var              {nullptr}; //Holds onto the var in case it needs to query more information
  GridVariableBase       * m_tempVarToReclaim {nullptr}; //Holds onto the result of an OnDemandDW getGridVar() call until an async device-to-host copy completes
  DeviceVarDest            m_dest;
  bool                     m_staging;
  IntVector                m_sizeVector;
  size_t                   m_sizeOfDataType;
  size_t                   m_varMemSize;
  IntVector                m_offset;
  int                      m_matlIndx;
  int                      m_levelIndx;
  const Patch            * m_patchPointer;
  const Task::Dependency * m_dep;
  Ghost::GhostType         m_gtype;
  int                      m_numGhostCells;
  unsigned int             m_whichGPU;
  TypeDescription::Type    m_type;

};



class DeviceInfo {

using DeviceVarDest = GpuUtilities::DeviceVarDestination;

public:

  DeviceInfo()
  {
    totalSize = 0;
    for (int i = 0; i < Task::TotalDWs; i++) {
      totalSizeForDataWarehouse[i] = 0;
      totalVars[i]                 = 0;
      totalMaterials[i]            = 0;
      totalLevels[i]               = 0;
    }
  }

  size_t totalSize;
  size_t totalSizeForDataWarehouse[Task::TotalDWs];
  unsigned int totalVars[Task::TotalDWs];
  unsigned int totalMaterials[Task::TotalDWs];
  unsigned int totalLevels[Task::TotalDWs];
};


class DeviceGridVariables {

using DeviceVarDest = GpuUtilities::DeviceVarDestination;

public:

  DeviceGridVariables() = default;

  void clear();

  // for grid vars
  void add( const Patch            * patchPointer
          ,       int                matlIndx
          ,       int                levelIndx
          ,       bool               staging
          ,       IntVector          sizeVector
          ,       size_t             sizeOfDataType
          ,       size_t             varMemSize
          ,       IntVector          offset
          , const Task::Dependency * dep
          ,       Ghost::GhostType   gtype
          ,       int                numGhostCells
          ,       unsigned int       whichGPU
          ,       Variable         * var
          ,       DeviceVarDest      dest
          );

  // for PerPatch vars and reduction vars.  They don't use ghost cells
  void add( const Patch             * patchPointer
          ,       int                 matlIndx
          ,       int                 levelIndx
          ,       size_t              varMemSize
          ,       size_t              sizeOfDataType
          , const Task::Dependency  * dep
          ,       unsigned int        whichGPU
          ,       Variable          * var
          ,       DeviceVarDest       dest
          );

  bool varAlreadyExists( const VarLabel * label
                       , const Patch    * patchPointer
                       ,       int        matlIndx
                       ,       int        levelIndx
                       ,       int        dataWarehouse
                       );

  bool stagingVarAlreadyExists( const VarLabel * label
                              , const Patch    * patchPointer
                              , int              matlIndx
                              , int              levelIndx
                              , IntVector        low
                              , IntVector        high
                              , int              dataWarehouse
                              );

  // for regular task vars.
  void addTaskGpuDWVar( const Patch            * patchPointer
                      ,       int                matlIndx
                      ,       int                levelIndx
                      ,       size_t             varMemSize
                      , const Task::Dependency * dep
                      ,       unsigned int       whichGPU
                      );

  // as the name suggests, to keep track of all vars for which this task is managing ghost cells.
  void addVarToBeGhostReady( const std::string      & taskName
                           ,       const Patch      * patchPointer
                           ,       int                matlIndx
                           ,       int                levelIndx
                           , const Task::Dependency * dep
                           ,       unsigned int       whichGPU
                           );

  // for staging contiguous arrays
  void addTaskGpuDWStagingVar( const Patch            * patchPointer
                             ,       int                matlIndx
                             ,       int                levelIndx
                             ,       IntVector          offset
                             ,       IntVector          sizeVector
                             ,       size_t             sizeOfDataType
                             , const Task::Dependency * dep
                             ,       unsigned int       whichGPU
                             );

  DeviceGridVariableInfo getStagingItem( const std::string& label
                                       , const Patch* patch
                                       , const int matlIndx
                                       , const int levelIndx
                                       , const IntVector low
                                       , const IntVector size
                                       , const int dataWarehouseIndex
                                       ) const;

  size_t getTotalSize( const unsigned int whichGPU ) ;

  size_t getSizeForDataWarehouse( const unsigned int whichGPU, const int dwIndex );

  unsigned int getTotalVars( const unsigned int whichGPU, const int DWIndex ) const;

  unsigned int getTotalMaterials( const unsigned int whichGPU, const int DWIndex ) const;

  unsigned int getTotalLevels( const unsigned int whichGPU, const int DWIndex ) const;

  std::multimap<GpuUtilities::LabelPatchMatlLevelDw, DeviceGridVariableInfo>& getMap()
  {
    return vars;
  }

private:

  // This multimap acts essentially contains objects
  // which are first queued up, and then processed in a group.  These DeviceGridVariableInfo objects
  // can 1) Tell the host-side GPU DW what variables need to be created on the GPU and what copies need
  // to be made host to deivce.  2) Tell a task GPU DW which variables it needs to know about from
  // the host-side GPU DW (this task GPU DW gets sent into the GPU).
  // This is a multimap because it can hold staging/foreign vars in it as well.  A regular variable can have
  // zero to many staging/foreign variables associated with it.
  std::multimap<GpuUtilities::LabelPatchMatlLevelDw, DeviceGridVariableInfo> vars;

  std::map<unsigned int, DeviceInfo> deviceInfoMap;

};

static std::map<const Patch *, int> patchAcceleratorLocation;

#endif // End CCA_COMPONENTS_SCHEDULERS_GPUGRIDVARIABLEINFO_H
