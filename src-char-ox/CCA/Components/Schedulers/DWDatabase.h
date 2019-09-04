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

#ifndef CCA_COMPONENTS_SCHEDULERS_DWDATABASE_H
#define CCA_COMPONENTS_SCHEDULERS_DWDATABASE_H

#ifdef BRADS_NEW_DWDATABASE

#include <CCA/Components/Schedulers/MemoryLog.h>

#include <Core/Grid/UnknownVariable.h>
#include <Core/Grid/Variables/ReductionVariableBase.h>
#include <Core/Grid/Variables/ScrubItem.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarLabelMatlMemspace.h>
#include <Core/Containers/FastHashTable.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Parallel/MasterLock.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Util/FancyAssert.h>
#include <Core/Util/ConcurrentHash/cuckoohash_map.hh>
#include <sci_hash_map.h>

#include <ostream>
#include <sstream>
#include <string>
#include <vector>
#include <memory>


/**************************************

  CLASS
    DWDatabase

  GENERAL INFORMATION

    DWDatabase.h

    Steven G. Parker
    Department of Computer Science
    University of Utah

    Center for the Simulation of Accidental Fires and Explosions (C-SAFE)


  KEYWORDS
    DWDatabase

  DESCRIPTION


****************************************/
namespace {


//struct keyDB_tag{};
//struct mvars_tag{};
//
//using  keyDB_monitor = Uintah::CrowdMonitor<keyDB_tag>;
//using  mvars_monitor = Uintah::CrowdMonitor<mvars_tag>;
//
//
//Uintah::MasterLock g_keyDB_mutex{};
//Uintah::MasterLock g_mvars_mutex{};

}

//This status is for concurrency.  This enum largely follows a model of "action -> state".
//For example, allocating -> allocated.  The idea is that only one thread should be able
//to claim moving into an action, and that winner should be responsible for setting it into the state.
//When it hits the state, other threads can utilize the variable.


typedef int atomicDataStatus;
//    0                   1                   2                   3
//    0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
//   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//   |    16-bit reference counter   |  unused           |G|B|V|B|A|B|
//   +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+

// Left sixteen bits is a 16-bit integer reference counter.

enum status { UNALLOCATED               = 0x00000000,
              BECOMING_ALLOCATED        = 0x00000001,
              ALLOCATED                 = 0x00000002,
              BECOMING_VALID            = 0x00000004,     // Could be due to computing/modifying, copying in, resizing, or superpatching.
              VALID                     = 0x00000008,     // For when a variable has its data, this excludes any knowledge of ghost cells.
              BECOMING_GHOST_VALID      = 0x00000010,     // For when when we know a variable is awaiting ghost cell data
                                                          // It is possible for VALID bit set to 0 or 1 with this bit set,
                                                          // meaning we can know a variable is awaiting ghost copies but we
                                                          // don't know from this bit alone if the variable is valid yet.
              GHOST_VALID               = 0x00000020      // For when a variable has its data and it has its ghost cells
                                                          // Note: Change to just GHOST_VALID?  Meaning ghost cells could be valid but the
                                                          // non ghost part is unknown?

              //LEFT_SIXTEEN_BITS                         // Use the other 16 bits as a usage to track how many tasks
                                                          // are currently using the variable.
};

namespace Uintah {


//template<class DomainType>
//class KeyDatabase {
//
//  template<class T> friend class DWDatabase;
//
//public:
//
//  KeyDatabase() {};
//
//  ~KeyDatabase() {};
//
//  void clear();
//
//  void insert( const VarLabel   * label
//             ,       int          matlIndex
//             , const DomainType * dom
//             );
//
//  int lookup( const VarLabel   * label
//            ,       int          matlIndex
//            , const DomainType * dom
//            );
//
//  void merge( const KeyDatabase<DomainType>& newDB );
//
//  void print( std::ostream & out, int rank ) const;
//
//private:
//
//
//  //using keyDBtype = hashmap<VarLabelMatlMemspace<DomainType, int>, int>;
//  using keyDBtype = std::unordered_map<VarLabelMatlMemspace<DomainType, MemorySpace>,
//                                       int,
//                                       Uintah::VarLabelMatlMemspaceHasher<DomainType, MemorySpace>newdi
//                                      >;
//  keyDBtype m_keys;
//
//  int m_key_count { 0 };
//
//};


template<class DomainType>
class DWDatabase {
  public:

    DWDatabase() {};

    ~DWDatabase();

    void clear();

    // void doReserve( KeyDatabase<DomainType>* keydb );

    bool exists( const VarLabel   * label
               ,       int          matlIndex
               , const DomainType * dom
               ) const;


    void put( const VarLabel   * label
            ,       int          matlindex
            , const DomainType * dom
            ,       Variable   * var
            ,       bool         init
            ,       bool         replace
            );


    void putReduce( const VarLabel              * label
                  ,       int                     matlindex
                  , const DomainType            * dom
                  ,       ReductionVariableBase * var
                  ,       bool init);


    void putForeign( const VarLabel   * label
                   ,       int          matlindex
                   , const DomainType * dom
                   ,       Variable   * var
                   ,       bool         init
                   );


    void get( const VarLabel   * label
            ,       int          matlindex
            , const DomainType * dom
            ,       Variable   & var
            ) const;


    void getlist( const VarLabel               * label
                ,       int                      matlIndex
                , const DomainType             * dom
                ,       std::vector<Variable*> & varlist
                ) const;


    inline Variable* get( const VarLabel   * label
                        ,       int          matlindex
                        , const DomainType * dom
                        ) const;


    void print( std::ostream & out, int rank ) const;


    void cleanForeign();

    // Scrub counter manipulator functions -- when the scrub count goes to
    // zero, the data is scrubbed.  Return remaining count

    // How Scrubbing works:  If a variable is in the OldDW, at the beginning of the timestep
    // initializeScrubs will be called for each of those variables.  For each variable computed
    // or copied via MPI, setScrubCount will be called on it, based on the scrubCountTable in
    // DetailedTasks.  Then, when the variable is used, decrementScrubCount is called on it
    // and if the count reaches zero, it is scrubbed.
    int decrementScrubCount( const VarLabel   * label
                           ,       int          matlindex
                           , const DomainType * dom
                           );


    void setScrubCount( const VarLabel   * label
                      ,       int          matlindex
                      , const DomainType * dom
                      ,       int          count
                      );


    void scrub( const VarLabel   * label
              ,       int          matlindex
              , const DomainType * dom
              );


    // add means increment the scrub count instead of setting it.  This is for when a DW
    // can act as a CoarseOldDW as well as an OldDW
    void initializeScrubs(       int                        dwid
                         , const FastHashTable<ScrubItem> * scrubcounts
                         ,       bool                       add
                         );


    void logMemoryUse(       std::ostream  & out
                     ,       unsigned long & total
                     , const std::string   & tag
                     ,       int             dwid
                     );

    void getVarLabelMatlTriples(std::vector<VarLabelMatlMemspace<DomainType, MemorySpace> >& vars) const;


    bool requestAllocationPrivilege( const VarLabel   * label,
                                     int                matlIndex,
                                     const DomainType * dom,
                                     MemorySpace        memorySpace);

    int lookup( const VarLabel   * label,
                                     int                matlIndex,
                                     const DomainType * dom,
                                     MemorySpace        memorySpace);




  private:

    bool setAllocated( atomicDataStatus& status );

    struct DataItem {

        DataItem() {}

        ~DataItem()
        {
          if (m_next) {
            delete m_next;
          }
          ASSERT(m_var);
          delete m_var;
        }
        Variable        * m_var  { nullptr };
        struct DataItem * m_next { nullptr };
        atomicDataStatus statusInMemory {0};

    };

    DataItem* getDataItem( const VarLabel   * label
                         ,       int          matlindex
                         , const DomainType * dom
                         ) const;


    //A concurrent hash map.  The key is a 4-tuple (label, material, domain(usually patch), memory space).
    //The concurrent hash map

    using varDBtype = cuckoohash_map<VarLabelMatlMemspace<DomainType, MemorySpace>,
                                   std::shared_ptr<DataItem>,
                                   Uintah::VarLabelMatlMemspaceHasher<DomainType, MemorySpace>
                                    >;
    varDBtype varDB;


//    KeyDatabase<DomainType>* m_keyDB {};
//
//    //The array of variable objects
//    using varDBtype = std::vector<DataItem*>;
//    varDBtype m_vars;
//
//    using scrubDBtype = std::vector<int>;
//    scrubDBtype m_scrubs;

    // eliminate copy, assignment and move
    DWDatabase( const DWDatabase & )            = delete;
    DWDatabase& operator=( const DWDatabase & ) = delete;
    DWDatabase( DWDatabase && )                 = delete;
    DWDatabase& operator=( DWDatabase && )      = delete;
};

//______________________________________________________________________
//
template<class DomainType>
DWDatabase<DomainType>::~DWDatabase()
{
  clear();
}

//______________________________________________________________________
//
template<class DomainType>
void DWDatabase<DomainType>::clear()
{
//  mvars_monitor mvars_lock{ Uintah::CrowdMonitor<mvars_tag>::WRITER };
//  for (auto iter = m_vars.begin(); iter != m_vars.end(); ++iter) {
//    if (*iter) {
//      delete *iter;
//    }
//    *iter = nullptr;
//  }
//  m_vars.clear();
  {
    auto lt = varDB.lock_table();
    for (auto &item : lt) {
      if (item.second) {
        delete item.second;
        item.second = nullptr;
      }
    }
  }
  varDB.clear();

}

//______________________________________________________________________
//
template<class DomainType>
void
DWDatabase<DomainType>::cleanForeign()
{

//  mvars_monitor mvars_lock{ Uintah::CrowdMonitor<mvars_tag>::WRITER };
//
//  for (auto iter = m_vars.begin(); iter != m_vars.end(); ++iter) {
//    if (*iter && (*iter)->m_var->isForeign()) {
//      delete (*iter);
//      (*iter) = nullptr;
//    }
//  }

  {
    auto lt = varDB.lock_table();
    for (auto &item : lt) {
      if (item.second && item.second->m_var->isForeign()) {
        delete item.second;

        item.second = nullptr;
      }
    }
  }
  varDB.clear();

}

//______________________________________________________________________
//
template<class DomainType>
int
DWDatabase<DomainType>::decrementScrubCount( const VarLabel   * label
                                           ,       int          matlIndex
                                           , const DomainType * dom
                                           )
{
  // Dav's conjectures on how this works:
  //   setScrubCount is called the first time with "count" set to some X.  
  //   This X represents the number of tasks that will use the var.  Later,
  //   after a task has used the var, it will call decrementScrubCount
  //   If scrubCount then is equal to 0, the var is scrubbed.

  // Brad's conjecture on scrub counters
  // They provide far too much work to maintain and offer little value in return.

//  ASSERT(matlIndex >= -1);
//  int idx = m_keyDB->lookup(label, matlIndex, dom);
//  if (idx == -1) {
//    return 0;
//  }
//
//  mvars_monitor mvars_lock{ Uintah::CrowdMonitor<mvars_tag>::WRITER };
//
//  if (!m_vars[idx]) {
//    return 0;
//  }
//  int rt = __sync_sub_and_fetch(&(m_scrubs[idx]), 1);
//  if (rt == 0) {
//    delete m_vars[idx];
//    m_vars[idx] = nullptr;
//  }
//  return rt;
}

//______________________________________________________________________
//
template<class DomainType>
void
DWDatabase<DomainType>::setScrubCount( const VarLabel   * label
                                     ,       int          matlIndex
                                     , const DomainType * dom
                                     ,       int          count
                                     )
{

//  int idx = m_keyDB->lookup(label, matlIndex, dom);
//  if (idx == -1) {
//    SCI_THROW(UnknownVariable(label->getName(), -99, dom, matlIndex, "DWDatabase::setScrubCount", __FILE__, __LINE__));
//  }
//  m_scrubs[idx] = count;
//
//  // TODO do we need this - APH 03/01/17
////  if (!__sync_bool_compare_and_swap(&(m_scrubs[idx]), 0, count)) {
////      SCI_THROW(InternalError("overwriting non-zero scrub counter", __FILE__, __LINE__));
////  }
}

//______________________________________________________________________
//
template<class DomainType>
void
DWDatabase<DomainType>::scrub( const VarLabel   * label
                             ,       int          matlIndex
                             , const DomainType * dom
                             )
{
//  ASSERT(matlIndex >= -1);
//  int idx = m_keyDB->lookup(label, matlIndex, dom);
//
//#if 0
//  if (m_vars[idx] == nullptr) {  // scrub not found
//    std::ostringstream msgstr;
//    msgstr << label->getName() << ", matl " << matlIndex << ", patch/level " << dom->getID() << " not found for scrubbing.";
//    SCI_THROW(InternalError(msgstr.str(), __FILE__, __LINE__));
//  }
//#endif
//  mvars_monitor mvars_lock{ Uintah::CrowdMonitor<mvars_tag>::WRITER };
//  if (idx != -1 && m_vars[idx]) {
//    delete m_vars[idx];
//    m_vars[idx] = nullptr;
//  }
}

//______________________________________________________________________
//
template<class DomainType>
void
DWDatabase<DomainType>::initializeScrubs(       int                        dwid
                                        , const FastHashTable<ScrubItem> * scrubcounts
                                        ,       bool                       add
                                        )
{
//  // loop over each variable, probing the scrubcount map. Set the scrubcount appropriately.
//  // If the variable has no entry in the scrubcount map, delete it
//  keyDB_monitor keyDB_lock{ Uintah::CrowdMonitor<keyDB_tag>::READER };
//
//  for (auto keyiter = m_keyDB->m_keys.begin(); keyiter != m_keyDB->m_keys.end();) {
//
//    mvars_monitor mvars_lock{ Uintah::CrowdMonitor<mvars_tag>::WRITER };
//
//    if (m_vars[keyiter->second]) {
//      VarLabelMatlMemspace<DomainType, MemorySpace> vlm = keyiter->first;
//      // See if it is in the scrubcounts map.
//      ScrubItem key(vlm.label_, vlm.matlIndex_, vlm.domain_, dwid);
//      ScrubItem* result = scrubcounts->lookup(&key);
//      if (!result && !add) {
//        delete m_vars[keyiter->second];
//        m_vars[keyiter->second] = nullptr;
//      }
//      else {
//        if (result) {
//          if (add) {
//            __sync_add_and_fetch(&(m_scrubs[keyiter->second]), result->m_count);
//          }
//          else {
//            if (!__sync_bool_compare_and_swap(&(m_scrubs[keyiter->second]), 0, result->m_count)) {
//              SCI_THROW(InternalError("initializing non-zero scrub counter", __FILE__, __LINE__));
//            }
//          }
//        }
//        keyiter++;
//      }
//    }
//    else {
//      keyiter++;
//    }
//  }
}

////______________________________________________________________________
////
//template<class DomainType>
//int
//KeyDatabase<DomainType>::lookup( const VarLabel   * label
//                               ,       int          matlIndex
//                               , const DomainType * dom
//                               )
//{
//
//
//  //std::lock_guard<Uintah::MasterLock> lookup_lock(g_keyDB_mutex);
//  keyDB_monitor keyDB_lock{ Uintah::CrowdMonitor<keyDB_tag>::READER };
//
//  VarLabelMatlMemspace<DomainType, MemorySpace> v(label, matlIndex, getRealDomain(dom), MemorySpace::HostSpace);
//  typename keyDBtype::const_iterator const_iter = m_keys.find(v);
//  if (const_iter == m_keys.end()) {
//    return -1;
//  }
//  else {
//    return const_iter->second;
//  }
//}
//
////______________________________________________________________________
////
//template<class DomainType>
//void
//KeyDatabase<DomainType>::merge( const KeyDatabase<DomainType> & newDB )
//{
//  keyDB_monitor keyDB_lock{ Uintah::CrowdMonitor<keyDB_tag>::WRITER };
//
//  for (typename keyDBtype::const_iterator const_keyiter = newDB.m_keys.begin(); const_keyiter != newDB.m_keys.end();
//      const_keyiter++) {
//    typename keyDBtype::const_iterator const_db_iter = m_keys.find(const_keyiter->first);
//    if (const_db_iter == m_keys.end()) {
//      m_keys.insert(std::pair<VarLabelMatlMemspace<DomainType, MemorySpace>, int>(const_keyiter->first, m_key_count++));
//    }
//  }
//}
//
////______________________________________________________________________
////
//template<class DomainType>
//void
//KeyDatabase<DomainType>::insert( const VarLabel   * label
//                               ,       int          matlIndex
//                               , const DomainType * dom
//                               )
//{
//  keyDB_monitor keyDB_lock{ Uintah::CrowdMonitor<keyDB_tag>::WRITER };
//
//  VarLabelMatlMemspace<DomainType, MemorySpace> v(label, matlIndex, getRealDomain(dom), MemorySpace::HostSpace);
//  typename keyDBtype::const_iterator const_iter = m_keys.find(v);
//  if (const_iter == m_keys.end()) {
//    m_keys.insert(std::pair<VarLabelMatlMemspace<DomainType, MemorySpace>, int>(v, m_key_count++));
//  }
//}
//
////______________________________________________________________________
////
//template<class DomainType>
//void
//KeyDatabase<DomainType>::clear()
//{
//  keyDB_monitor keyDB_lock{ Uintah::CrowdMonitor<keyDB_tag>::WRITER };
//
//  m_keys.clear();
//  m_key_count = 0;
//}
//
////______________________________________________________________________
////
//template<class DomainType>
//void
//KeyDatabase<DomainType>::print( std::ostream & out, int rank ) const
//{
//
//  keyDB_monitor keyDB_lock{ Uintah::CrowdMonitor<keyDB_tag>::READER };
//
//  for (auto keyiter = m_keys.begin(); keyiter != m_keys.end(); keyiter++) {
//    const VarLabelMatlMemspace<DomainType, MemorySpace>& vlm = keyiter->first;
//    const DomainType* dom = vlm.domain_;
//    if (dom) {
//      out << rank << " Name: " << vlm.label_->getName() << "  domain: " << *dom << "  matl:" << vlm.matlIndex_ << '\n';
//    }
//    else {
//      out << rank << " Name: " << vlm.label_->getName() << "  domain: N/A  matl: " << vlm.matlIndex_ << '\n';
//    }
//  }
//}

//______________________________________________________________________
//
//template<class DomainType>
//void
//DWDatabase<DomainType>::doReserve( KeyDatabase<DomainType> * keydb )
//{
//  //Note, we could check the capacity and only set writer when the capacity changes
//  //(vectors usually give themselves lots of extra buffer room so resizing by 1 is easy)
//  mvars_monitor mvars_lock{ Uintah::CrowdMonitor<mvars_tag>::WRITER };
//  keyDB_monitor keyDB_lock{ Uintah::CrowdMonitor<keyDB_tag>::WRITER };
//
//  m_keyDB = keydb;
//  m_vars.resize(m_keyDB->m_key_count + 1, (DataItem*)nullptr);
//  m_scrubs.resize(m_keyDB->m_key_count + 1, 0);
//}

//______________________________________________________________________
//
template<class DomainType>
bool
DWDatabase<DomainType>::exists( const VarLabel   * label,
                                int                matlIndex,
                                const DomainType * dom ) const
{
  // lookup is lock_guard protected

//  int idx = m_keyDB->lookup(label, matlIndex, dom);
//
//  {
//    mvars_monitor mvars_lock{ Uintah::CrowdMonitor<mvars_tag>::READER };
//
//    if (idx == -1) {
//      return false;
//    }
//    if (m_vars[idx] == nullptr) {
//      return false;
//    }
//    return true;
//  }

  return (varDB.contains(VarLabelMatlMemspace<DomainType, MemorySpace>(label, matlIndex, getRealDomain(dom), MemorySpace::HostSpace)));

}

//______________________________________________________________________
//
// Note: This method should be called only via legacy tasks that don't preload variables.
// From a concurrency perspective, these assume allocation has already happened.
// The preloading approach instead assumes the scheduler must get rights to preallocate.
template<class DomainType>
void
DWDatabase<DomainType>::put( const VarLabel   * label
                           ,       int          matlIndex
                           , const DomainType * dom
                           ,       Variable   * var
                           ,       bool         init
                           ,       bool         replace
                           )
{

  ASSERT(matlIndex >= -1);
//
//  if (init) {
//    keyDB_monitor keyDB_lock{ Uintah::CrowdMonitor<keyDB_tag>::WRITER };
//    m_keyDB->insert(label, matlIndex, dom);
//    this->doReserve(m_keyDB);
//  }
//
//  // lookup is lock_guard protected
//  int idx = m_keyDB->lookup(label, matlIndex, dom);
//  if (idx == -1) {
//    SCI_THROW(UnknownVariable(label->getName(), -1, dom, matlIndex, "check task computes", __FILE__, __LINE__));
//  }
//
//  if (m_vars[idx]) {
//    if (m_vars[idx]->m_next) {
//      SCI_THROW(InternalError("More than one vars on this label", __FILE__, __LINE__));
//    }
//    if (!replace) {
//      SCI_THROW(InternalError("Put replacing old vars", __FILE__, __LINE__));
//    }
//    ASSERT(m_vars[idx]->m_var != var);
//    delete m_vars[idx];
//  }
//  DataItem* newdi = new DataItem();
//  newdi->m_var = var;
//  m_vars[idx] = newdi;

  // Note: We can assume this entry always remains as the current plan is to only remove variables at the end of the timestep, and
  // not putting happens at that point.  In other words, we never mix putting and removing simultaneously.
  // If a need arises to mix putting and removing simultaneously, I believe upsert or update_fn allows us to pass a functor, and
  // then we could also add a reference counter within the DataItem to ensure something can't get removed until we are done with it.

  std::shared_ptr<DataItem> dataItem = std::make_shared<DataItem>();
  dataItem->m_var = var;
  setAllocated( dataItem->statusInMemory );

  if ( replace ) {
    // The replace parameter was designed long ago,and it assumes the user manages concurrency correctly.
    // From a concurrency standpoint, replace is messy.  It is effectively a combined erase and put.  To do erasing concurrently,
    // we would need to start doing referencing counting on tasks using the variable.  For now, we're going to just replace and pray the
    // user isn't invoking replace while other tasks are simultaneously using this variable.
    varDB.insert_or_assign(
        VarLabelMatlMemspace<DomainType, MemorySpace>( label, matlIndex, getRealDomain(dom), MemorySpace::HostSpace ),
        dataItem);
  } else {
    bool success = varDB.insert(
        VarLabelMatlMemspace<DomainType, MemorySpace>( label, matlIndex, getRealDomain(dom), MemorySpace::HostSpace ),
        dataItem);
    if ( !success ) {
      //The key was already in here.  That shouldn't happen with this legacy put method.
      SCI_THROW(InternalError("Put replacing old vars", __FILE__, __LINE__));
    }
  }
}

//______________________________________________________________________
//
template<class DomainType>
void
DWDatabase<DomainType>::putReduce( const VarLabel              * label
                                 ,       int                     matlIndex
                                 , const DomainType            * dom
                                 ,       ReductionVariableBase * var
                                 ,       bool                    init
                                 )
{
  ASSERT(matlIndex >= -1);

  // Attempt to insert it.
  // If not successful on inserting, then one is already in there.  Obtain the var,
  //  set the flag from VALID back to BECOMING_VALID (when possible), perform the
  //  reduction operation on it, then set the flag back to VALID.
  std::shared_ptr<DataItem> dataItem = std::make_shared<DataItem>();
  dataItem->m_var = var;
  dataItem->statusInMemory = ALLOCATED & VALID;

  auto key = VarLabelMatlMemspace<DomainType, MemorySpace>( label, matlIndex, getRealDomain(dom), MemorySpace::HostSpace );

  bool success = varDB.insert(key, dataItem);
  if (!success) {
    // Set the data item's var back to nullptr so the var doesn't get deleted when dataItem gets deallocated.
    dataItem->m_var = nullptr;
    std::shared_ptr<DataItem> existingDataItem = varDB.find(key);
    //Attempt to set it back to BECOMING_VALID to get our lock on the reduction variable.
    do {
      // Do an atomic read.
      atomicDataStatus oldVarStatus = __sync_or_and_fetch(&existingDataItem->statusInMemory, 0);
      if ( (( oldVarStatus & ALLOCATED ) == 0 ) || (( oldVarStatus & VALID ) == 0 )) {
        //A sanity check
        std::ostringstream msgstr;
        msgstr << label->getName() << ", material " << matlIndex << " is not listed as allocated or valid.";
        SCI_THROW(InternalError(msgstr.str(), __FILE__, __LINE__));
      } else {
        //Attempt to claim we get to modify it.  Turn off the VALID bit and turn on the BECOMING_VALID bit.
        atomicDataStatus newVarStatus = oldVarStatus & ~VALID;
        newVarStatus = newVarStatus | BECOMING_VALID;
        success = __sync_bool_compare_and_swap(&existingDataItem->statusInMemory, oldVarStatus, newVarStatus);
      }
    } while (!success);

    //We now have the lock.  Perform the reduction on the var
    ReductionVariableBase* existingVar = dynamic_cast<ReductionVariableBase*>(existingDataItem->m_var);
    existingVar->reduce(*var);

    //Release the lock.  Turn off BECOMING_VALID and turn on VALID.
    do {
      atomicDataStatus oldVarStatus = __sync_or_and_fetch(&existingDataItem->statusInMemory, 0);
      atomicDataStatus newVarStatus = oldVarStatus & ~BECOMING_VALID;
      newVarStatus = newVarStatus | VALID;
      success = __sync_bool_compare_and_swap(&existingDataItem->statusInMemory, oldVarStatus, newVarStatus);
    } while(!success);

  }
}

//______________________________________________________________________
//
template<class DomainType>
void
DWDatabase<DomainType>::putForeign( const VarLabel   * label
                                  ,       int          matlIndex
                                  , const DomainType * dom
                                  ,       Variable   * var
                                  ,       bool         init
                                  )
{
  ASSERT(matlIndex >= -1);

  if (init) {
    m_keyDB->insert(label, matlIndex, dom);
    this->doReserve(m_keyDB);
  }

  // lookup is lock_guard protected
  int idx = m_keyDB->lookup(label, matlIndex, dom);
  if (idx == -1) {
    SCI_THROW(UnknownVariable(label->getName(), -1, dom, matlIndex, "check task computes", __FILE__, __LINE__));
  }

  DataItem* newdi = new DataItem();
  newdi->m_var = var;
  do {
    newdi->m_next = m_vars[idx];
  }
  while (!__sync_bool_compare_and_swap(&m_vars[idx], newdi->m_next, newdi));  // vars[iter->second] = newdi;
}

//______________________________________________________________________
//
template<class DomainType>
typename DWDatabase<DomainType>::DataItem*
DWDatabase<DomainType>::getDataItem( const VarLabel   * label
                                   ,       int          matlIndex
                                   , const DomainType * dom
                                   ) const
{
  ASSERT(matlIndex >= -1);
  int idx = m_keyDB->lookup(label, matlIndex, dom);
  if (idx == -1) {
    SCI_THROW(UnknownVariable(label->getName(), -99, dom, matlIndex, "DWDatabase::getDataItem", __FILE__, __LINE__));
  }
  return m_vars[idx];
}

//______________________________________________________________________
//
template<class DomainType>
inline
Variable*
DWDatabase<DomainType>::get( const VarLabel   * label
                           ,       int          matlIndex
                           , const DomainType * dom
                           ) const
{
  const DataItem* dataItem = getDataItem(label, matlIndex, dom);
  ASSERT(dataItem != nullptr);          // should have thrown an exception before
  ASSERT(dataItem->m_next == nullptr);  // should call getlist()
  return dataItem->m_var;
}

//______________________________________________________________________
//
template<class DomainType>
void
DWDatabase<DomainType>::get( const VarLabel   * label
                           ,       int          matlIndex
                           , const DomainType * dom
                           ,       Variable   & var
                           ) const
{
  Variable* tmp = get(label, matlIndex, dom);
  var.copyPointer(*tmp);
}

//______________________________________________________________________
//
template<class DomainType>
void
DWDatabase<DomainType>::getlist( const VarLabel               * label
                               ,       int                      matlIndex
                               , const DomainType             * dom
                               ,       std::vector<Variable*> & varlist
                               ) const
{
  // this function is allowed to return an empty list

  for (DataItem* dataItem = getDataItem(label, matlIndex, dom); dataItem != nullptr; dataItem = dataItem->m_next) {
    varlist.push_back(dataItem->m_var);
  }
}

//______________________________________________________________________
//
template<class DomainType>
void
DWDatabase<DomainType>::print( std::ostream & out, int rank ) const
{
  keyDB_monitor keyDB_lock{ Uintah::CrowdMonitor<keyDB_tag>::READER };

  for (auto keyiter = m_keyDB->m_keys.begin(); keyiter != m_keyDB->m_keys.end(); keyiter++) {
    if (m_vars[keyiter->second]) {
      const VarLabelMatlMemspace<DomainType, MemorySpace>& vlm = keyiter->first;
      const DomainType* dom = vlm.domain_;
      if (dom) {
        out << rank << " Name: " << vlm.label_->getName() << "  domain: " << *dom << "  matl:" << vlm.matlIndex_ << '\n';
      }
      else {
        out << rank << " Name: " << vlm.label_->getName() << "  domain: N/A  matl: " << vlm.matlIndex_ << '\n';
      }
    }
  }
}

//_____________________________________requestAllocationPrivilegerequestAllocationPrivilege_________________________________
//
template<class DomainType>
void
DWDatabase<DomainType>::logMemoryUse(       std::ostream  & out
                                    ,       unsigned long & total
                                    , const std::string   & tag
                                    ,       int             dwid
                                    )
{
  keyDB_monitor keyDB_lock{ Uintah::CrowdMonitor<keyDB_tag>::READER };

  for (auto keyiter = m_keyDB->m_keys.begin(); keyiter != m_keyDB->m_keys.end(); keyiter++) {
    if (m_vars[keyiter->second]) {
      Variable* var = m_vars[keyiter->second]->m_var;
      VarLabelMatlMemspace<DomainType, MemorySpace> vlm = keyiter->first;
      const VarLabel* label = vlm.label_;
      std::string elems;
      unsigned long totsize;
      void* ptr;
      var->getSizeInfo(elems, totsize, ptr);
      const TypeDescription* td = label->typeDescription();

      logMemory(out, total, tag, label->getName(), (td ? td->getName() : "-"), vlm.domain_,
                vlm.matlIndex_, elems, totsize, ptr, dwid);
    }
  }
}

//______________________________________________________________________
//
template<class DomainType>
void
DWDatabase<DomainType>::getVarLabelMatlTriples( std::vector<VarLabelMatlMemspace<DomainType, MemorySpace> > & v) const
{
  keyDB_monitor keyDB_lock{ Uintah::CrowdMonitor<keyDB_tag>::READER };

  for (auto keyiter = m_keyDB->m_keys.begin(); keyiter != m_keyDB->m_keys.end(); keyiter++) {
    const VarLabelMatlMemspace<DomainType, MemorySpace>& vlm = keyiter->first;
    if (m_vars[keyiter->second]) {
      v.push_back(vlm);
    }
  }
}



//The following functions help with tasks that request simulation variables get
//prepared *prior* to task execution, rather than during task execution.  This relies
//heavily on the atomicDataStatus bitset for concurrency.

template<class DomainType>
bool DWDatabase<DomainType>::requestAllocationPrivilege( const VarLabel   * label,
                                                         int                matlIndex,
                                                         const DomainType * dom,
                                                         MemorySpace        memorySpace)
                                                         {

}

// The following are atomicDataStatus bitset functions

//______________________________________________________________________
//
bool setAllocated( atomicDataStatus& status )
{
  bool allocated = false;

  //get the value
  atomicDataStatus oldVarStatus = __sync_or_and_fetch( &status, 0 );
  if ( ( oldVarStatus & BECOMING_ALLOCATED ) == 0 ) {
    //A sanity check
    printf("ERROR:\nGPUDataWarehouse::setAllocated( )  Can't allocate a status if it wasn't previously marked as allocating.\n");
    exit( -1 );
  } else if ( ( oldVarStatus & BECOMING_ALLOCATED ) == BECOMING_ALLOCATED ) {
    //A sanity check
    printf("ERROR:\nGPUDataWarehouse::compareAndSwapAllocate( )  Can't allocate a status if it's already allocated\n");
    exit( -1 );
  }
  else {
    //Attempt to claim we'll allocate it.  Create what we want the status to look like
    //by turning off allocating and turning on allocated.
    //Note: No need to turn off UNALLOCATED, it's defined as all zero bits.
    //But the below is kept in just for readability's sake.
    atomicDataStatus newVarStatus = oldVarStatus & ~UNALLOCATED;
    newVarStatus = newVarStatus & ~BECOMING_ALLOCATED;
    newVarStatus = newVarStatus | ALLOCATED;

    //If we succeeded in our attempt to claim to allocate, this returns true.
    //If we failed, thats a real problem, and we crash the problem below.
    allocated = __sync_bool_compare_and_swap( &status, oldVarStatus, newVarStatus );
  }
  if ( !allocated ) {
    //Another sanity check
    printf("ERROR:\nGPUDataWarehouse::compareAndSwapAllocate( )  Something wrongly modified the atomic status while setting the allocated flag\n");
    exit( -1 );
  }
  return allocated;
}

} // namespace Uintah
#else // #ifdef BRADS_NEW_DWDATABASE
//The original DWDatabase.h

#include <CCA/Components/Schedulers/MemoryLog.h>

#include <Core/Grid/UnknownVariable.h>
#include <Core/Grid/Variables/ReductionVariableBase.h>
#include <Core/Grid/Variables/ScrubItem.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <Core/Grid/Variables/VarLabelMatl.h>
#include <Core/Containers/FastHashTable.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Parallel/MasterLock.h>
#include <Core/Parallel/Parallel.h>
#include <Core/Util/FancyAssert.h>

#include <sci_hash_map.h>

#include <ostream>
#include <sstream>
#include <string>
#include <vector>


/**************************************

  CLASS
    DWDatabase

  GENERAL INFORMATION

    DWDatabase.h

    Steven G. Parker
    Department of Computer Science
    University of Utah

    Center for the Simulation of Accidental Fires and Explosions (C-SAFE)


  KEYWORDS
    DWDatabase

  DESCRIPTION


****************************************/
namespace {

Uintah::MasterLock g_keyDB_mutex{};
Uintah::MasterLock g_mvars_mutex{};

}

namespace Uintah {


template<class DomainType>
class KeyDatabase {

  template<class T> friend class DWDatabase;

public:

  KeyDatabase() {};

  ~KeyDatabase() {};

  void clear();

  void insert( const VarLabel   * label
             ,       int          matlIndex
             , const DomainType * dom
             );

  int lookup( const VarLabel   * label
            ,       int          matlIndex
            , const DomainType * dom
            );

  void merge( const KeyDatabase<DomainType>& newDB );

  void print( std::ostream & out, int rank ) const;

private:

  using keyDBtype = hashmap<VarLabelMatl<DomainType>, int>;
  keyDBtype m_keys;

  int m_key_count { 0 };

};


template<class DomainType>
class DWDatabase {

  public:

    DWDatabase() {};

    ~DWDatabase();

    void clear();

    void doReserve( KeyDatabase<DomainType>* keydb );

    bool exists( const VarLabel   * label
               ,       int          matlIndex
               , const DomainType * dom
               ) const;


    void put( const VarLabel   * label
            ,       int          matlindex
            , const DomainType * dom
            ,       Variable   * var
            ,       bool         init
            ,       bool         replace
            );


    void putReduce( const VarLabel              * label
                  ,       int                     matlindex
                  , const DomainType            * dom
                  ,       ReductionVariableBase * var
                  ,       bool init);


    void putForeign( const VarLabel   * label
                   ,       int          matlindex
                   , const DomainType * dom
                   ,       Variable   * var
                   ,       bool         init
                   );


    void get( const VarLabel   * label
            ,       int          matlindex
            , const DomainType * dom
            ,       Variable   & var
            ) const;


    void getlist( const VarLabel               * label
                ,       int                      matlIndex
                , const DomainType             * dom
                ,       std::vector<Variable*> & varlist
                ) const;


    inline Variable* get( const VarLabel   * label
                        ,       int          matlindex
                        , const DomainType * dom
                        ) const;


    void print( std::ostream & out, int rank ) const;


    void cleanForeign();

    // Scrub counter manipulator functions -- when the scrub count goes to
    // zero, the data is scrubbed.  Return remaining count

    // How Scrubbing works:  If a variable is in the OldDW, at the beginning of the timestep
    // initializeScrubs will be called for each of those variables.  For each variable computed
    // or copied via MPI, setScrubCount will be called on it, based on the scrubCountTable in
    // DetailedTasks.  Then, when the variable is used, decrementScrubCount is called on it
    // and if the count reaches zero, it is scrubbed.
    int decrementScrubCount( const VarLabel   * label
                           ,       int          matlindex
                           , const DomainType * dom
                           );


    void setScrubCount( const VarLabel   * label
                      ,       int          matlindex
                      , const DomainType * dom
                      ,       int          count
                      );


    void scrub( const VarLabel   * label
              ,       int          matlindex
              , const DomainType * dom
              );


    // add means increment the scrub count instead of setting it.  This is for when a DW
    // can act as a CoarseOldDW as well as an OldDW
    void initializeScrubs(       int                        dwid
                         , const FastHashTable<ScrubItem> * scrubcounts
                         ,       bool                       add
                         );


    void logMemoryUse(       std::ostream  & out
                     ,       unsigned long & total
                     , const std::string   & tag
                     ,       int             dwid
                     );

    void getVarLabelMatlTriples(std::vector<VarLabelMatl<DomainType> >& vars) const;


  private:

    struct DataItem {

        DataItem() {}

        ~DataItem()
        {
          if (m_next) {
            delete m_next;
          }
          ASSERT(m_var);
          delete m_var;
        }
        Variable        * m_var  { nullptr };
        struct DataItem * m_next { nullptr };
    };

    DataItem* getDataItem( const VarLabel   * label
                         ,       int          matlindex
                         , const DomainType * dom
                         ) const;

    KeyDatabase<DomainType>* m_keyDB {};

    using varDBtype = std::vector<DataItem*>;
    varDBtype m_vars;

    using scrubDBtype = std::vector<int>;
    scrubDBtype m_scrubs;

    // eliminate copy, assignment and move
    DWDatabase( const DWDatabase & )            = delete;
    DWDatabase& operator=( const DWDatabase & ) = delete;
    DWDatabase( DWDatabase && )                 = delete;
    DWDatabase& operator=( DWDatabase && )      = delete;
};

//______________________________________________________________________
//
template<class DomainType>
DWDatabase<DomainType>::~DWDatabase()
{
  clear();
}

//______________________________________________________________________
//
template<class DomainType>
void DWDatabase<DomainType>::clear()
{
  for (auto iter = m_vars.begin(); iter != m_vars.end(); ++iter) {
    if (*iter) {
      delete *iter;
    }
    *iter = nullptr;
  }
  m_vars.clear();
}

//______________________________________________________________________
//
template<class DomainType>
void
DWDatabase<DomainType>::cleanForeign()
{
  std::lock_guard<Uintah::MasterLock> exists_lock(g_mvars_mutex);

  for (auto iter = m_vars.begin(); iter != m_vars.end(); ++iter) {
    if (*iter && (*iter)->m_var->isForeign()) {
      delete (*iter);
      (*iter) = nullptr;
    }
  }
}

//______________________________________________________________________
//
template<class DomainType>
int
DWDatabase<DomainType>::decrementScrubCount( const VarLabel   * label
                                           ,       int          matlIndex
                                           , const DomainType * dom
                                           )
{
  // Dav's conjectures on how this works:
  //   setScrubCount is called the first time with "count" set to some X.
  //   This X represents the number of tasks that will use the var.  Later,
  //   after a task has used the var, it will call decrementScrubCount
  //   If scrubCount then is equal to 0, the var is scrubbed.

  ASSERT(matlIndex >= -1);
  int idx = m_keyDB->lookup(label, matlIndex, dom);
  if (idx == -1) {
    return 0;
  }
  if (!m_vars[idx]) {
    return 0;
  }
  int rt = __sync_sub_and_fetch(&(m_scrubs[idx]), 1);
  if (rt == 0) {
    delete m_vars[idx];
    m_vars[idx] = nullptr;
  }
  return rt;
}

//______________________________________________________________________
//
template<class DomainType>
void
DWDatabase<DomainType>::setScrubCount( const VarLabel   * label
                                     ,       int          matlIndex
                                     , const DomainType * dom
                                     ,       int          count
                                     )
{
  int idx = m_keyDB->lookup(label, matlIndex, dom);
  if (idx == -1) {
    SCI_THROW(UnknownVariable(label->getName(), -99, dom, matlIndex, "DWDatabase::setScrubCount", __FILE__, __LINE__));
  }
  m_scrubs[idx] = count;

  // TODO do we need this - APH 03/01/17
//  if (!__sync_bool_compare_and_swap(&(m_scrubs[idx]), 0, count)) {
//      SCI_THROW(InternalError("overwriting non-zero scrub counter", __FILE__, __LINE__));
//  }
}

//______________________________________________________________________
//
template<class DomainType>
void
DWDatabase<DomainType>::scrub( const VarLabel   * label
                             ,       int          matlIndex
                             , const DomainType * dom
                             )
{
  ASSERT(matlIndex >= -1);
  int idx = m_keyDB->lookup(label, matlIndex, dom);

#if 0
  if (m_vars[idx] == nullptr) {  // scrub not found
    std::ostringstream msgstr;
    msgstr << label->getName() << ", matl " << matlIndex << ", patch/level " << dom->getID() << " not found for scrubbing.";
    SCI_THROW(InternalError(msgstr.str(), __FILE__, __LINE__));
  }
#endif

  if (idx != -1 && m_vars[idx]) {
    delete m_vars[idx];
    m_vars[idx] = nullptr;
  }
}

//______________________________________________________________________
//
template<class DomainType>
void
DWDatabase<DomainType>::initializeScrubs(       int                        dwid
                                        , const FastHashTable<ScrubItem> * scrubcounts
                                        ,       bool                       add
                                        )
{
  // loop over each variable, probing the scrubcount map. Set the scrubcount appropriately.
  // If the variable has no entry in the scrubcount map, delete it
  for (auto keyiter = m_keyDB->m_keys.begin(); keyiter != m_keyDB->m_keys.end();) {
    if (m_vars[keyiter->second]) {
      VarLabelMatl<DomainType> vlm = keyiter->first;
      // See if it is in the scrubcounts map.
      ScrubItem key(vlm.label_, vlm.matlIndex_, vlm.domain_, dwid);
      ScrubItem* result = scrubcounts->lookup(&key);
      if (!result && !add) {
        delete m_vars[keyiter->second];
        m_vars[keyiter->second] = nullptr;
      }
      else {
        if (result) {
          if (add) {
            __sync_add_and_fetch(&(m_scrubs[keyiter->second]), result->m_count);
          }
          else {
            if (!__sync_bool_compare_and_swap(&(m_scrubs[keyiter->second]), 0, result->m_count)) {
              SCI_THROW(InternalError("initializing non-zero scrub counter", __FILE__, __LINE__));
            }
          }
        }
        keyiter++;
      }
    }
    else {
      keyiter++;
    }
  }
}

//______________________________________________________________________
//
template<class DomainType>
int
KeyDatabase<DomainType>::lookup( const VarLabel   * label
                               ,       int          matlIndex
                               , const DomainType * dom
                               )
{
  std::lock_guard<Uintah::MasterLock> lookup_lock(g_keyDB_mutex);

  VarLabelMatl<DomainType> v(label, matlIndex, getRealDomain(dom));
  typename keyDBtype::const_iterator const_iter = m_keys.find(v);
  if (const_iter == m_keys.end()) {
    return -1;
  }
  else {
    return const_iter->second;
  }
}

//______________________________________________________________________
//
template<class DomainType>
void
KeyDatabase<DomainType>::merge( const KeyDatabase<DomainType> & newDB )
{
  for (typename keyDBtype::const_iterator const_keyiter = newDB.m_keys.begin(); const_keyiter != newDB.m_keys.end();
      const_keyiter++) {
    typename keyDBtype::const_iterator const_db_iter = m_keys.find(const_keyiter->first);
    if (const_db_iter == m_keys.end()) {
      m_keys.insert(std::pair<VarLabelMatl<DomainType>, int>(const_keyiter->first, m_key_count++));
    }
  }
}

//______________________________________________________________________
//
template<class DomainType>
void
KeyDatabase<DomainType>::insert( const VarLabel   * label
                               ,       int          matlIndex
                               , const DomainType * dom
                               )
{
  VarLabelMatl<DomainType> v(label, matlIndex, getRealDomain(dom));
  typename keyDBtype::const_iterator const_iter = m_keys.find(v);
  if (const_iter == m_keys.end()) {
    m_keys.insert(std::pair<VarLabelMatl<DomainType>, int>(v, m_key_count++));
  }
}

//______________________________________________________________________
//
template<class DomainType>
void
KeyDatabase<DomainType>::clear()
{
  m_keys.clear();
  m_key_count = 0;
}

//______________________________________________________________________
//
template<class DomainType>
void
KeyDatabase<DomainType>::print( std::ostream & out, int rank ) const
{
  for (auto keyiter = m_keys.begin(); keyiter != m_keys.end(); keyiter++) {
    const VarLabelMatl<DomainType>& vlm = keyiter->first;
    const DomainType* dom = vlm.domain_;
    if (dom) {
      out << rank << " Name: " << vlm.label_->getName() << "  domain: " << *dom << "  matl:" << vlm.matlIndex_ << '\n';
    }
    else {
      out << rank << " Name: " << vlm.label_->getName() << "  domain: N/A  matl: " << vlm.matlIndex_ << '\n';
    }
  }
}

//______________________________________________________________________
//
template<class DomainType>
void
DWDatabase<DomainType>::doReserve( KeyDatabase<DomainType> * keydb )
{
  m_keyDB = keydb;
  m_vars.resize(m_keyDB->m_key_count + 1, (DataItem*)nullptr);
  m_scrubs.resize(m_keyDB->m_key_count + 1, 0);
}

//______________________________________________________________________
//
template<class DomainType>
bool
DWDatabase<DomainType>::exists( const VarLabel   * label
                              ,       int          matlIndex
                              , const DomainType * dom
                              ) const
{
  // lookup is lock_guard protected
  int idx = m_keyDB->lookup(label, matlIndex, dom);

  {
    std::lock_guard<Uintah::MasterLock> exists_lock(g_mvars_mutex);
    if (idx == -1) {
      return false;
    }
    if (m_vars[idx] == nullptr) {
      return false;
    }
    return true;
  }

}

//______________________________________________________________________
//
template<class DomainType>
void
DWDatabase<DomainType>::put( const VarLabel   * label
                           ,       int          matlIndex
                           , const DomainType * dom
                           ,       Variable   * var
                           ,       bool         init
                           ,       bool         replace
                           )
{

  ASSERT(matlIndex >= -1);

  {
    std::lock_guard<Uintah::MasterLock> put_lock(g_keyDB_mutex);
    if (init) {
      m_keyDB->insert(label, matlIndex, dom);
      this->doReserve(m_keyDB);
    }
  }

  // lookup is lock_guard protected
  int idx = m_keyDB->lookup(label, matlIndex, dom);
  if (idx == -1) {
    SCI_THROW(UnknownVariable(label->getName(), -1, dom, matlIndex, "check task computes", __FILE__, __LINE__));
  }

  if (m_vars[idx]) {
    if (m_vars[idx]->m_next) {
      SCI_THROW(InternalError("More than one vars on this label", __FILE__, __LINE__));
    }
    if (!replace) {
      SCI_THROW(InternalError("Put replacing old vars", __FILE__, __LINE__));
    }
    ASSERT(m_vars[idx]->m_var != var);
    delete m_vars[idx];
  }
  DataItem* newdi = new DataItem();
  newdi->m_var = var;
  m_vars[idx] = newdi;
}

//______________________________________________________________________
//
template<class DomainType>
void
DWDatabase<DomainType>::putReduce( const VarLabel              * label
                                 ,       int                     matlIndex
                                 , const DomainType            * dom
                                 ,       ReductionVariableBase * var
                                 ,       bool                    init
                                 )
{
  ASSERT(matlIndex >= -1);

  {
    std::lock_guard<Uintah::MasterLock> put_reduce_lock(g_keyDB_mutex);
    if (init) {
      m_keyDB->insert(label, matlIndex, dom);
      this->doReserve(m_keyDB);
    }
  }

  // lookup is lock_guard protected
  int idx = m_keyDB->lookup(label, matlIndex, dom);
  if (idx == -1) {
    SCI_THROW(UnknownVariable(label->getName(), -1, dom, matlIndex, "check task computes", __FILE__, __LINE__));
  }

  DataItem* newdi = new DataItem();
  newdi->m_var = var;
  do {
    DataItem* olddi = __sync_lock_test_and_set(&m_vars[idx], 0);
    if (olddi == nullptr) {
      olddi = newdi;
    }
    else {
      ReductionVariableBase* oldvar = dynamic_cast<ReductionVariableBase*>(olddi->m_var);
      ReductionVariableBase* newvar = dynamic_cast<ReductionVariableBase*>(newdi->m_var);
      oldvar->reduce(*newvar);
      delete newdi;
    }
    newdi = __sync_lock_test_and_set(&m_vars[idx], olddi);
  }
  while (newdi != nullptr);
}

//______________________________________________________________________
//
template<class DomainType>
void
DWDatabase<DomainType>::putForeign( const VarLabel   * label
                                  ,       int          matlIndex
                                  , const DomainType * dom
                                  ,       Variable   * var
                                  ,       bool         init
                                  )
{
  ASSERT(matlIndex >= -1);

  {
    std::lock_guard<Uintah::MasterLock> put_foreign_lock(g_keyDB_mutex);
    if (init) {
      m_keyDB->insert(label, matlIndex, dom);
      this->doReserve(m_keyDB);
    }
  }

  // lookup is lock_guard protected
  int idx = m_keyDB->lookup(label, matlIndex, dom);
  if (idx == -1) {
    SCI_THROW(UnknownVariable(label->getName(), -1, dom, matlIndex, "check task computes", __FILE__, __LINE__));
  }

  DataItem* newdi = new DataItem();
  newdi->m_var = var;
  do {
    newdi->m_next = m_vars[idx];
  }
  while (!__sync_bool_compare_and_swap(&m_vars[idx], newdi->m_next, newdi));  // vars[iter->second] = newdi;
}

//______________________________________________________________________
//
template<class DomainType>
typename DWDatabase<DomainType>::DataItem*
DWDatabase<DomainType>::getDataItem( const VarLabel   * label
                                   ,       int          matlIndex
                                   , const DomainType * dom
                                   ) const
{
  ASSERT(matlIndex >= -1);
  int idx = m_keyDB->lookup(label, matlIndex, dom);
  if (idx == -1) {
    SCI_THROW(UnknownVariable(label->getName(), -99, dom, matlIndex, "DWDatabase::getDataItem", __FILE__, __LINE__));
  }
  return m_vars[idx];
}

//______________________________________________________________________
//
template<class DomainType>
inline
Variable*
DWDatabase<DomainType>::get( const VarLabel   * label
                           ,       int          matlIndex
                           , const DomainType * dom
                           ) const
{
  const DataItem* dataItem = getDataItem(label, matlIndex, dom);
  ASSERT(dataItem != nullptr);          // should have thrown an exception before
  ASSERT(dataItem->m_next == nullptr);  // should call getlist()
  return dataItem->m_var;
}

//______________________________________________________________________
//
template<class DomainType>
void
DWDatabase<DomainType>::get( const VarLabel   * label
                           ,       int          matlIndex
                           , const DomainType * dom
                           ,       Variable   & var
                           ) const
{
  Variable* tmp = get(label, matlIndex, dom);
  var.copyPointer(*tmp);
}

//______________________________________________________________________
//
template<class DomainType>
void
DWDatabase<DomainType>::getlist( const VarLabel               * label
                               ,       int                      matlIndex
                               , const DomainType             * dom
                               ,       std::vector<Variable*> & varlist
                               ) const
{
  // this function is allowed to return an empty list

  for (DataItem* dataItem = getDataItem(label, matlIndex, dom); dataItem != nullptr; dataItem = dataItem->m_next) {
    varlist.push_back(dataItem->m_var);
  }
}

//______________________________________________________________________
//
template<class DomainType>
void
DWDatabase<DomainType>::print( std::ostream & out, int rank ) const
{
  for (auto keyiter = m_keyDB->m_keys.begin(); keyiter != m_keyDB->m_keys.end(); keyiter++) {
    if (m_vars[keyiter->second]) {
      const VarLabelMatl<DomainType>& vlm = keyiter->first;
      const DomainType* dom = vlm.domain_;
      if (dom) {
        out << rank << " Name: " << vlm.label_->getName() << "  domain: " << *dom << "  matl:" << vlm.matlIndex_ << '\n';
      }
      else {
        out << rank << " Name: " << vlm.label_->getName() << "  domain: N/A  matl: " << vlm.matlIndex_ << '\n';
      }
    }
  }
}

//______________________________________________________________________
//
template<class DomainType>
void
DWDatabase<DomainType>::logMemoryUse(       std::ostream  & out
                                    ,       unsigned long & total
                                    , const std::string   & tag
                                    ,       int             dwid
                                    )
{
  for (auto keyiter = m_keyDB->m_keys.begin(); keyiter != m_keyDB->m_keys.end(); keyiter++) {
    if (m_vars[keyiter->second]) {
      Variable* var = m_vars[keyiter->second]->m_var;
      VarLabelMatl<DomainType> vlm = keyiter->first;
      const VarLabel* label = vlm.label_;
      std::string elems;
      unsigned long totsize;
      void* ptr;
      var->getSizeInfo(elems, totsize, ptr);
      const TypeDescription* td = label->typeDescription();

      logMemory(out, total, tag, label->getName(), (td ? td->getName() : "-"), vlm.domain_,
                vlm.matlIndex_, elems, totsize, ptr, dwid);
    }
  }
}

//______________________________________________________________________
//
template<class DomainType>
void
DWDatabase<DomainType>::getVarLabelMatlTriples( std::vector<VarLabelMatl<DomainType> > & v) const
{
  for (auto keyiter = m_keyDB->m_keys.begin(); keyiter != m_keyDB->m_keys.end(); keyiter++) {
    const VarLabelMatl<DomainType>& vlm = keyiter->first;
    if (m_vars[keyiter->second]) {
      v.push_back(vlm);
    }
  }
}

} // namespace Uintah


//
// Hash function for VarLabelMatl
//
#ifdef HAVE_GNU_HASHMAP

  namespace __gnu_cxx
  {
    using Uintah::DWDatabase;
    using Uintah::VarLabelMatl;
    template <class DomainType>
    struct hash<VarLabelMatl<DomainType> > : public std::unary_function<VarLabelMatl<DomainType>, size_t>
    {
      size_t operator()(const VarLabelMatl<DomainType>& v) const
      {
        size_t h=0;
        char *str =const_cast<char*> (v.label_->getName().data());
        while (int c = *str++) h = h*7+c;
        return ( ( ((size_t)v.label_) << (sizeof(size_t)/2) ^ ((size_t)v.label_) >> (sizeof(size_t)/2) )
                 ^ (size_t)v.domain_ ^ (size_t)v.matlIndex_ );
      }
    };
  }

#elif HAVE_TR1_HASHMAP || HAVE_C11_HASHMAP

  namespace std {
#if HAVE_TR1_HASHMAP
    namespace tr1 {
#endif
      using Uintah::DWDatabase;
      using Uintah::VarLabelMatl;
      template <class DomainType>
      struct hash<VarLabelMatl<DomainType> > : public unary_function<VarLabelMatl<DomainType>, size_t>
      {
        size_t operator()(const VarLabelMatl<DomainType>& v) const
        {
          size_t h=0;
          char *str =const_cast<char*> (v.label_->getName().data());
          while (int c = *str++) h = h*7+c;
          return ( ( ((size_t)v.label_) << (sizeof(size_t)/2) ^ ((size_t)v.label_) >> (sizeof(size_t)/2) )
                   ^ (size_t)v.domain_ ^ (size_t)v.matlIndex_ );
        }
      };
#if HAVE_TR1_HASHMAP
    } // end namespace tr1
#endif
  } // end namespace std

#endif // HAVE_GNU_HASHMAP



#endif // #ifdef BRADS_NEW_DWDATABASE

#endif // CCA_COMPONENTS_SCHEDULERS_DWDATABASE_H
