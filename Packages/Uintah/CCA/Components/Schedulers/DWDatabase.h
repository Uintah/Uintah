#ifndef UINTAH_HOMEBREW_DWDatabase_H
#define UINTAH_HOMEBREW_DWDatabase_H

#include <Packages/Uintah/Core/Grid/UnknownVariable.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/CCA/Components/Schedulers/MemoryLog.h>

#include <Core/Exceptions/InternalError.h>
#include <Core/Malloc/Allocator.h>

#include <map>
#include <vector>
#include <iosfwd>
#include <list>
#include <sstream>

namespace Uintah {

using std::vector;
using std::iostream;
using std::ostringstream;

using namespace SCIRun;
   
   /**************************************
     
     CLASS
       DWDatabase
      
       Short Description...
      
     GENERAL INFORMATION
      
       DWDatabase.h
      
       Steven G. Parker
       Department of Computer Science
       University of Utah
      
       Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
      
       Copyright (C) 2000 SCI Group
      
     KEYWORDS
       DWDatabase
      
     DESCRIPTION
       Long description...
      
     WARNING
      
     ****************************************/
template<class VarType>
class DWDatabase {
public:
   DWDatabase();
   ~DWDatabase();

   // Note: for global variables, use matlIndex = -1, patch = NULL
  
   bool exists(const VarLabel* label, int matlIndex, const Patch* patch) const;
   bool exists(const VarLabel* label, const Patch* patch) const;
   void put(const VarLabel* label, int matlindex, const Patch* patch,
	    VarType* var, bool replace);
   void get(const VarLabel* label, int matlindex, const Patch* patch,
	    VarType& var) const;
   inline VarType* get(const VarLabel* label, int matlindex,
		const Patch* patch) const;
   void copyAll(const DWDatabase& from, const VarLabel*, const Patch* patch);
   void print(ostream&) const;
   void cleanForeign();

   // Scrub counter manipulator functions -- when the scrub count goes to
   // zero, the data is scrubbed.
   // Note: count can be negative to decrement.
   void addScrubCount(const VarLabel* label, int matlindex,
		      const Patch* patch, int count, unsigned int addIfZero=0);

   // scrub everything with a scrubcount of zero
   void scrubExtraneous();
  
   void logMemoryUse(ostream& out, unsigned long& total,
		     const std::string& tag, int dwid);
private:
   struct DataItem {
     DataItem()
       : var(0), scrubCount(0) {}
     VarType* var;
     int scrubCount;
   };
  
   const DataItem& getDataItem(const VarLabel* label, int matlindex,
			       const Patch* patch) const;
   void scrub(const VarLabel* label, int matlindex, const Patch* patch);
    
   typedef vector<DataItem> dataDBtype;

   class PatchRecord {
   public:
      PatchRecord(const Patch*);
      ~PatchRecord();

      const Patch* getPatch()
      { return patch; }

      void putVar(int matlIndex, VarType* var, bool replace);

      inline DataItem* getDataItem(int matlIndex);
      inline VarType* getVar(int matlIndex) const;

      inline void removeVar(int matlIndex);
     
      const dataDBtype& getVars() const
      { return vars; }
     
      bool empty() const
      { return count == 0; }
   private:
      const Patch* patch;
      dataDBtype vars;
      int count;
   };

   typedef map<const Patch*, PatchRecord*> patchDBtype;
   struct NameRecord {
      const VarLabel* label;
      patchDBtype patches;

      NameRecord(const VarLabel* label);
      ~NameRecord();
   };

   typedef map<const VarLabel*, NameRecord*, VarLabel::Compare> nameDBtype;
   nameDBtype names;

   typedef map<const VarLabel*, DataItem, VarLabel::Compare> globalDBtype;
   // globals are defined for all patches and materials
   // (patch == NULL, matlIndex = -1)
   globalDBtype globals; 

   DWDatabase(const DWDatabase&);
   DWDatabase& operator=(const DWDatabase&);      
};

template<class VarType>
DWDatabase<VarType>::DWDatabase()
{
}

template<class VarType>
DWDatabase<VarType>::~DWDatabase()
{
   for(typename nameDBtype::iterator iter = names.begin();
       iter != names.end(); iter++){
     
     if (iter->first->typeDescription() != 0) {
       ostringstream msgstr;
       msgstr << "Failed to scrub: " << iter->first->getName()
	      << " completely";
       throw InternalError(msgstr.str());       
     } 
     delete iter->second;
   }
   for(typename globalDBtype::iterator iter = globals.begin();
       iter != globals.end(); iter++){
      delete iter->second.var;
   }
}

template<class VarType>
void DWDatabase<VarType>::scrubExtraneous()
{
  // scrub all vars with scrubcounts of zero
  list<const VarLabel*> labelsToRemove;
  
  for(typename nameDBtype::iterator iter = names.begin();
      iter != names.end(); iter++){
    list<const Patch*> patchesToRemove;
    patchDBtype& patchDB = iter->second->patches;
    typename patchDBtype::iterator patchRecordIter;
    for (patchRecordIter = patchDB.begin(); patchRecordIter != patchDB.end();
	 ++patchRecordIter) {
      PatchRecord* patchRecord = patchRecordIter->second;
      const dataDBtype& vars = patchRecord->getVars();
      for (unsigned int m = 0; m < vars.size(); m++) {
	DataItem* dataItem = patchRecord->getDataItem(m);
	if (dataItem && dataItem->scrubCount == 0)
	  patchRecord->removeVar(m);
      }
      if (patchRecord->empty()) {
	delete patchRecord;
	patchesToRemove.push_back(patchRecordIter->first);
      }
    }
    for (list<const Patch*>::iterator patchIter = patchesToRemove.begin();
	 patchIter != patchesToRemove.end(); ++patchIter) {
      patchDB.erase(*patchIter);
    }
    if (patchDB.size() == 0) {
      delete iter->second;
      labelsToRemove.push_back(iter->first);
    }
  }
  for (list<const VarLabel*>::iterator labelIter = labelsToRemove.begin();
       labelIter != labelsToRemove.end(); ++labelIter) {
    names.erase(*labelIter);
  }

  labelsToRemove.clear();
  for(typename globalDBtype::iterator iter = globals.begin();
      iter != globals.end(); iter++){
    if (iter->second.scrubCount == 0) {
      delete iter->second.var;
      labelsToRemove.push_back(iter->first);
    }
  }
  for (list<const VarLabel*>::iterator labelIter = labelsToRemove.begin();
       labelIter != labelsToRemove.end(); ++labelIter) {
    globals.erase(*labelIter);
  }
}

template<class VarType>
void
DWDatabase<VarType>::cleanForeign()
{
   for(typename nameDBtype::iterator iter = names.begin();
       iter != names.end(); iter++){
      NameRecord* nr = iter->second;
      for(typename patchDBtype::iterator iter = nr->patches.begin();
	  iter != nr->patches.end(); iter++){
	 PatchRecord* pr = iter->second;
	 for (int m = 0; m < (int)pr->getVars().size(); m++) {
	    const VarType* var = pr->getVars()[m].var;
	    if(var && var->isForeign()){
	      pr->removeVar(m);
	    }
	 }
      }
   }

   list<const VarLabel*> toBeRemoved;
   for(typename globalDBtype::iterator iter = globals.begin();
       iter != globals.end(); iter++){
      VarType* var = iter->second.var;
      if(var && var->isForeign()){
	 toBeRemoved.push_back(iter->first);
	 delete var;
      }
   }
   
   for (list<const VarLabel*>::iterator iter = toBeRemoved.begin();
	iter != toBeRemoved.end(); iter++)
      globals.erase(*iter);
}

template<class VarType>
void DWDatabase<VarType>::
addScrubCount(const VarLabel* label, int matlIndex,
	      const Patch* patch, int count, unsigned int addIfZero)
{
  // Dav's conjectures on how this works:
  //   addScrubCount is called the first time with "addIfZero" set to some X.  
  //   This X represents the number of tasks that will use the var.  Later,
  //   after a task has used the var, it will call addScrubCount with 
  //   "count" set to -1 (in order to decrement the value of scrubCount.
  //   If scrubCount then is equal to 0, the var is scrubbed.

  DataItem& data = const_cast<DataItem&>(getDataItem(label, matlIndex, patch));

  // If it was zero to begin with, then add addIfZero first.
  if (data.scrubCount == 0)
    data.scrubCount += addIfZero;
  data.scrubCount += count;
  ASSERT(data.scrubCount >= 0);
  if (data.scrubCount == 0) {
    // some things can get scrubbed right after the task creates it.
    scrub(label, matlIndex, patch);
  }
}

template<class VarType>
void
DWDatabase<VarType>::scrub(const VarLabel* var, int matlIndex,
			   const Patch* patch)
{
  if (matlIndex < 0) {
    if (patch == NULL) {
      // get from globals
      typename globalDBtype::const_iterator globaliter = globals.find(var);
      if (globaliter != globals.end() && globaliter->second.var != 0) {
	delete globaliter->second.var;
	globals.erase(var);
	return; // found and scrubbed
      }
      else
	throw UnknownVariable(var->getName(),
			      "no global variable with this name");
    }
    else
      throw InternalError("matlIndex must be >= 0");
  }
  else {
    typename nameDBtype::iterator iter = names.find(var);
    if(iter != names.end()){
      patchDBtype& patchDB = iter->second->patches;
      typename patchDBtype::iterator patchRecordIter = patchDB.find(patch);
      PatchRecord* patchRecord;
      if (patchRecordIter != patchDB.end() &&
	  ((patchRecord = patchRecordIter->second) != 0)) {
	ASSERT(patchRecord->getDataItem(matlIndex)->scrubCount == 0);
	patchRecord->removeVar(matlIndex);
	if (patchRecord->empty()) {
	  delete patchRecord;
	  patchDB.erase(patchRecordIter);
	  if (patchDB.size() == 0) {
	    delete iter->second;
	    names.erase(iter);
	  }
	}
	return; // found and scrubbed
      }
    }
  }

  // scrub not found
  ostringstream msgstr;
  msgstr << var->getName() << ", matl " << matlIndex
	 << ", patch " << (patch ? patch->getID() : -1)
	 << " not found for scrubbing.";

  throw InternalError(msgstr.str());
}

template<class VarType>
DWDatabase<VarType>::NameRecord::NameRecord(const VarLabel* label)
   : label(label)
{
}

template<class VarType>
DWDatabase<VarType>::NameRecord::~NameRecord()
{
   for(typename patchDBtype::iterator iter = patches.begin();
       iter != patches.end(); iter++){
      delete iter->second;
   }   
}

template<class VarType>
DWDatabase<VarType>::PatchRecord::PatchRecord(const Patch* patch)
  : patch(patch), count(0)
{
}

template <class VarType>
void DWDatabase<VarType>::PatchRecord::putVar(int matlIndex, VarType* var,
					      bool replace)
{   
  if(matlIndex >= (int)vars.size()){
    vars.resize(matlIndex+1);
  }
  
  VarType* oldVar = vars[matlIndex].var;
  
  if (oldVar != 0) {
    if (!replace) {
      throw InternalError("Put replacing old variable");
    }

    // replace
    ASSERT(oldVar != var);
    delete oldVar;
    if (var == 0) count--;
  }
  else {
    if (var != 0) count++;
  }
  vars[matlIndex].var = var;      
}

template <class VarType>
inline typename DWDatabase<VarType>::DataItem*
DWDatabase<VarType>::PatchRecord::getDataItem(int matlIndex)
{
  if (matlIndex < (int)vars.size())
    return &vars[matlIndex];
  else
    return 0;
}

template <class VarType>
inline VarType* DWDatabase<VarType>::PatchRecord::getVar(int matlIndex) const
{
  if (matlIndex < (int)vars.size())
    return vars[matlIndex].var;
  else
    return 0;
}

template <class VarType>
inline void DWDatabase<VarType>::PatchRecord::removeVar(int matlIndex)
{
  if (matlIndex < (int)vars.size()) {
    if (vars[matlIndex].var != 0) {
      //ASSERT(vars[matlIndex].scrubCount == 0);
      delete vars[matlIndex].var;
      vars[matlIndex].var = 0;
      count--;
    }
  }
}

template<class VarType>
DWDatabase<VarType>::PatchRecord::~PatchRecord()
{
   for(typename dataDBtype::iterator iter = vars.begin();
       iter != vars.end(); iter++){
      if(iter->var){
	 delete iter->var;
      }
   }   
}

template<class VarType>
bool DWDatabase<VarType>::exists(const VarLabel* label, int matlIndex,
				 const Patch* patch) const
{
   if (patch && patch->isVirtual())
     patch = patch->getRealPatch();
   if (matlIndex < 0)
      return (patch == NULL) && (globals.find(label) != globals.end());
  
   typename nameDBtype::const_iterator nameiter = names.find(label);
   if(nameiter != names.end()) {
      NameRecord* nr = nameiter->second;
      typename patchDBtype::const_iterator patchiter = nr->patches.find(patch);
      if(patchiter != nr->patches.end()) {
	 PatchRecord* rr = patchiter->second;
	 if (rr->getVar(matlIndex) != 0) return true;
      }
   }
   return false;
}

template<class VarType>
bool DWDatabase<VarType>::exists(const VarLabel* label, const Patch* patch) const
{
   if (patch && patch->isVirtual())
     patch = patch->getRealPatch();
   typename nameDBtype::const_iterator nameiter = names.find(label);
   if(nameiter != names.end()) {
      NameRecord* nr = nameiter->second;
      typename patchDBtype::const_iterator patchiter = nr->patches.find(patch);
      if(patchiter != nr->patches.end()) {
	 PatchRecord* rr = patchiter->second;
	 for(int i=0; i<(int)rr->getVars().size(); i++){
	    if(rr->getVars()[i].var != 0){
	       return true;
	    }
	 }
      }
   }
   if (patch == NULL) {
      // try globals as last resort
      return globals.find(label) != globals.end();
   }
   return false;
}

template<class VarType>
void DWDatabase<VarType>::put(const VarLabel* label, int matlIndex,
			      const Patch* patch,
			      VarType* var,
			      bool replace)
{
  ASSERT(patch == 0 || !patch->isVirtual());
  if(matlIndex < 0) {
    if (patch == NULL) {
      // add to globals
      typename globalDBtype::iterator globaliter = globals.find(label);
      if (globaliter == globals.end()) {
	globals[label].var = var;
	return;
      }
      else if (replace) {
	DataItem& globalData = globals[label];
	if (globalData.var != NULL) {
	  ASSERT(globalData.var != var);
	  delete globalData.var;
	}
	globalData.var = var;
	return;
      }
      else
	 throw InternalError("Put replacing old global variable");	
    }
    else
      throw InternalError("matlIndex must be >= 0");
  }
  
   typename nameDBtype::iterator nameiter = names.find(label);
   if(nameiter == names.end()){
      names[label] = scinew NameRecord(label);
      nameiter = names.find(label);
   }

   NameRecord* nr = nameiter->second;
   typename patchDBtype::const_iterator patchiter = nr->patches.find(patch);
   if(patchiter == nr->patches.end()) {
      nr->patches[patch] = scinew PatchRecord(patch);
      patchiter = nr->patches.find(patch);
   }

   PatchRecord* rr = patchiter->second;
   rr->putVar(matlIndex, var, replace);
}

template<class VarType>
const typename DWDatabase<VarType>::DataItem&
DWDatabase<VarType>::getDataItem(const VarLabel* label, int matlIndex,
				 const Patch* patch) const
{
   ASSERT(patch == 0 || !patch->isVirtual());
   if(matlIndex < 0) {
      if (patch == NULL) {
         // get from globals
         typename globalDBtype::const_iterator globaliter =globals.find(label);
         if (globaliter != globals.end() && globaliter->second.var != 0)
	    return globaliter->second;
         else
	    throw UnknownVariable(label->getName(),
                                  "no global variable with this name");
      }
      else
         throw InternalError("matlIndex must be >= 0");
   }
  
   typename nameDBtype::const_iterator nameiter = names.find(label);
   if(nameiter == names.end())
      throw UnknownVariable(label->getName(), patch, matlIndex,
			    "no variable name");

   NameRecord* nr = nameiter->second;

   typename patchDBtype::const_iterator patchiter = nr->patches.find(patch);
   if(patchiter == nr->patches.end())
      throw UnknownVariable(label->getName(), patch, matlIndex,
			    "no patch with this variable name");

   DataItem* dataItem = patchiter->second->getDataItem(matlIndex);
   if (dataItem == 0 || (dataItem->var == 0)) {
      throw UnknownVariable(label->getName(), patch, matlIndex,
			    "no material with this patch and variable name");
   }
   return *dataItem;
}

template<class VarType>
inline VarType* DWDatabase<VarType>::get(const VarLabel* label, int matlIndex,
					 const Patch* patch) const
{
  const DataItem& dataItem = getDataItem(label, matlIndex, patch);
  ASSERT(dataItem.var != 0); // should have thrown an exception before
  return dataItem.var;
}

template<class VarType>
void DWDatabase<VarType>::get(const VarLabel* label, int matlIndex,
			      const Patch* patch,
			      VarType& var) const
{
   if (patch && patch->isVirtual()) {
     VarType* tmp = get(label, matlIndex, patch->getRealPatch());
     var.copyPointer(*tmp);
     var.offsetGrid(patch->getVirtualOffset());
   }
   else {
     VarType* tmp = get(label, matlIndex, patch);
     var.copyPointer(*tmp);
   }
}

template<class VarType>
void DWDatabase<VarType>::copyAll(const DWDatabase& from,
				  const VarLabel* label,
				  const Patch* patch)
{
   ASSERT(patch == 0 || !patch->isVirtual());
  
   typename nameDBtype::const_iterator nameiter = from.names.find(label);
   if(nameiter == from.names.end())
      return;

   NameRecord* nr = nameiter->second;
   typename patchDBtype::const_iterator patchiter = nr->patches.find(patch);
   if(patchiter == nr->patches.end())
      return;

   PatchRecord* rr = patchiter->second;

   for(int i=0;i<rr->getVars().size();i++){
      if(rr->getVars()[i].var)
	 put(label, i, patch, rr->getVars()[i].var->clone(), false);
   }

   globals = from.globals;
}

template<class VarType>
void DWDatabase<VarType>::print(std::ostream& out) const
{
   for(typename nameDBtype::const_iterator nameiter = names.begin();
       nameiter != names.end(); nameiter++){
      NameRecord* nr = nameiter->second;
      out << nr->label->getName() << '\n';
      for(typename patchDBtype::const_iterator patchiter = nr->patches.begin();
	  patchiter != nr->patches.end();patchiter++){
	 PatchRecord* rr = patchiter->second;
	 out <<  "  " << *(rr->patch) << '\n';
	 for(int i=0;i<(int)rr->getVars().size();i++){
	    if(rr->getVars()[i].var){
	       out << "    Material " << i << '\n';
	    }
	 }
      }
   }

   for( typename globalDBtype::const_iterator globaliter = globals.begin();
	globaliter != globals.end(); globaliter++ )
     out << (*globaliter).first->getName() << '\n';
}

template<class VarType>
void
DWDatabase<VarType>::logMemoryUse(ostream& out, unsigned long& total,
				  const std::string& tag, int dwid)
{
  for(typename nameDBtype::iterator iter = names.begin();
      iter != names.end(); iter++){
    NameRecord* nr = iter->second;
    for(typename patchDBtype::iterator iter = nr->patches.begin();
	iter != nr->patches.end(); iter++){
      PatchRecord* pr = iter->second;
      for(int i=0;i<(int)pr->getVars().size();i++){
	VarType* var = pr->getVars()[i].var;
	if(var){
	  const VarLabel* label = nr->label;
	  string elems;
	  unsigned long totsize;
	  void* ptr;
	  var->getSizeInfo(elems, totsize, ptr);
	  const TypeDescription* td = label->typeDescription();
	  logMemory(out, total, tag, label->getName(), (td?td->getName():"-"),
		    pr->getPatch(), i, elems, totsize, ptr, dwid);
	}
      }
    }
  }
  
  for(typename globalDBtype::iterator iter = globals.begin();
      iter != globals.end(); iter++){
    const VarLabel* label = iter->first;
    VarType* var = iter->second.var;
    string elems;
    unsigned long totsize;
    void* ptr;
    var->getSizeInfo(elems, totsize, ptr);
    const TypeDescription* td = label->typeDescription();
    logMemory(out, total, tag, label->getName(), (td?td->getName():"-"),
	      0, -1, elems, totsize, ptr, dwid);
  }
}

} // End namespace Uintah

#endif
