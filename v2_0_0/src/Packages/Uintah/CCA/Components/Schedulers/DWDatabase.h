#ifndef UINTAH_HOMEBREW_DWDatabase_H
#define UINTAH_HOMEBREW_DWDatabase_H

#include <Packages/Uintah/Core/Grid/UnknownVariable.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>
#include <Packages/Uintah/CCA/Components/Schedulers/MemoryLog.h>
#include <Packages/Uintah/Core/Grid/VarLabelMatlPatchDW.h>

#include <Packages/Uintah/Core/Parallel/Parallel.h>

#include <Core/Exceptions/InternalError.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/FancyAssert.h>

#include <sgi_stl_warnings_off.h>
#include <map>
#include <vector>
#include <iosfwd>
#include <list>
#include <sstream>
#include <sgi_stl_warnings_on.h>

namespace Uintah {

using std::vector;
using std::iostream;
using std::ostringstream;

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
 template<class VarType, class DomainType>
   class DWDatabase {
   public:
   DWDatabase();
   ~DWDatabase();

   bool exists(const VarLabel* label, int matlIndex, const DomainType* dom) const;
   bool exists(const VarLabel* label, const DomainType* dom) const;
   void put(const VarLabel* label, int matlindex, const DomainType* dom,
	    VarType* var, bool replace);
   void get(const VarLabel* label, int matlindex, const DomainType* dom,
	    VarType& var) const;
   inline VarType* get(const VarLabel* label, int matlindex,
		const DomainType* dom) const;
   void print(ostream&) const;
   void cleanForeign();

   // Scrub counter manipulator functions -- when the scrub count goes to
   // zero, the data is scrubbed.
   void decrementScrubCount(const VarLabel* label, int matlindex,
			    const DomainType* dom);
   void setScrubCount(const VarLabel* label, int matlindex,
		      const DomainType* dom, int count);
   void scrub(const VarLabel* label, int matlindex, const DomainType* dom);
   void initializeScrubs(int dwid, const map<VarLabelMatlPatchDW, int>& scrubcounts);

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
			       const DomainType* dom) const;
    
   typedef vector<DataItem> dataDBtype;

   class DomainRecord {
   public:
      DomainRecord(const DomainType*);
      ~DomainRecord();

      const DomainType* getDomain()
      { return dom; }

      void putVar(int matlIndex, VarType* var, bool replace);

      inline DataItem* getDataItem(int matlIndex);
      inline VarType* getVar(int matlIndex) const;

      inline void removeVar(int matlIndex);
     
      const dataDBtype& getVars() const
      { return vars; }
     
      bool empty() const
      { return count == 0; }
   private:
      const DomainType* dom;
      dataDBtype vars;
      int count;
   };

   typedef map<int, DomainRecord*> domainDBtype;
   struct NameRecord {
      const VarLabel* label;
      domainDBtype domains;

      NameRecord(const VarLabel* label);
      ~NameRecord();
   };

   typedef map<const VarLabel*, NameRecord*, VarLabel::Compare> nameDBtype;
   nameDBtype names;

   DWDatabase(const DWDatabase&);
   DWDatabase& operator=(const DWDatabase&);      
};

template<class VarType, class DomainType>
DWDatabase<VarType, DomainType>::DWDatabase()
{
}

template<class VarType, class DomainType>
DWDatabase<VarType, DomainType>::~DWDatabase()
{
  for(typename nameDBtype::iterator iter = names.begin();
      iter != names.end(); iter++){

#ifdef DEBUG
    // This can happen in some normal cases (especially at program
    // shutdown), but catching it is useful for debugging the scrubbing
    // stuff...
    if (iter->first->typeDescription() != 0
	&& iter->first->typeDescription()->getType() != TypeDescription::ReductionVariable) {
      ostringstream msgstr;
      msgstr << "Failed to scrub: " << iter->first->getName()
	     << " completely";
      SCI_THROW(InternalError(msgstr.str()));
    }
#endif
    delete iter->second;
  }
}

template<class VarType, class DomainType>
void
DWDatabase<VarType, DomainType>::cleanForeign()
{
  for(typename nameDBtype::iterator iter = names.begin();
      iter != names.end(); iter++){
    NameRecord* nr = iter->second;
    for(typename domainDBtype::iterator iter = nr->domains.begin();
	iter != nr->domains.end(); iter++){
      DomainRecord* pr = iter->second;
      for (int m = 0; m < (int)pr->getVars().size(); m++) {
	const VarType* var = pr->getVars()[m].var;
	if(var && var->isForeign()){
	  pr->removeVar(m-1);
	}
      }
    }
  }
}

template<class VarType, class DomainType>
void DWDatabase<VarType, DomainType>::
decrementScrubCount(const VarLabel* label, int matlIndex,
		    const DomainType* dom)
{
  DataItem& data = const_cast<DataItem&>(getDataItem(label, matlIndex, dom));
  // Dav's conjectures on how this works:
  //   setScrubCount is called the first time with "count" set to some X.  
  //   This X represents the number of tasks that will use the var.  Later,
  //   after a task has used the var, it will call decrementScrubCount
  //   If scrubCount then is equal to 0, the var is scrubbed.

  ASSERT(data.scrubCount > 0);
  if(!--data.scrubCount)
    scrub(label, matlIndex, dom);
}

template<class VarType, class DomainType>
void DWDatabase<VarType, DomainType>::
setScrubCount(const VarLabel* label, int matlIndex,
	      const DomainType* dom, int count)
{
  DataItem& data = const_cast<DataItem&>(getDataItem(label, matlIndex, dom));
  ASSERT(data.var != 0); // should have thrown an exception before
  if(data.scrubCount == 0)
    data.scrubCount = count;
}

template<class VarType, class DomainType>
void
DWDatabase<VarType, DomainType>::scrub(const VarLabel* var, int matlIndex,
				       const DomainType* dom)
{
  ASSERT(matlIndex >= -1);
  int domainid = getDB_ID(dom);
  typename nameDBtype::iterator iter = names.find(var);
  if(iter != names.end()){
    domainDBtype& domainDB = iter->second->domains;
    typename domainDBtype::iterator domainRecordIter = domainDB.find(domainid);
    DomainRecord* domainRecord = 0;
    if (domainRecordIter != domainDB.end() &&
	((domainRecord = domainRecordIter->second) != 0)) {
      ASSERTEQ(domainRecord->getDataItem(matlIndex)->scrubCount, 0);
      domainRecord->removeVar(matlIndex);
      if (domainRecord->empty()) {
	delete domainRecord;
	domainDB.erase(domainRecordIter);
	if (domainDB.size() == 0) {
	  delete iter->second;
	  names.erase(iter);
	}
      }
      return; // found and scrubbed
    }
  }

  // scrub not found
  ostringstream msgstr;
  msgstr << var->getName() << ", matl " << matlIndex
	 << ", patch/level " << domainid
	 << " not found for scrubbing.";

  SCI_THROW(InternalError(msgstr.str()));
}

template<class VarType, class DomainType>
void
DWDatabase<VarType, DomainType>::initializeScrubs(int dwid,
						  const map<VarLabelMatlPatchDW, int>& scrubcounts)
{
  // loop over each variable, probing the scrubcount map. Set the
  // scrubcount appropriately.  if the variable has no entry in
  // the scrubcount map, delete it
  for(typename nameDBtype::iterator nameiter = names.begin();
      nameiter != names.end();){
    NameRecord* nr = nameiter->second;
    for(typename domainDBtype::iterator domainiter = nr->domains.begin();
	domainiter != nr->domains.end();){
      DomainRecord* rr = domainiter->second;
      for(int i=0;i<(int)rr->getVars().size();i++){
	if(rr->getVars()[i].var){
	  // See if it is in the scrubcounts map.  matls are offset by 1
	  VarLabelMatlPatchDW key(nr->label, i-1, rr->getDomain(), dwid);
	  map<VarLabelMatlPatchDW, int>::const_iterator iter = scrubcounts.find(key);
	  if(iter == scrubcounts.end()){
	    // Delete this...
	    rr->removeVar(i-1);
	  } else {
	    ASSERTEQ(rr->getDataItem(i-1)->scrubCount, 0);
	    rr->getDataItem(i-1)->scrubCount = iter->second;
	  }
	}
      }
      if(rr->empty()) {
	const DomainType* dom = rr->getDomain();
	delete rr;
	nr->domains.erase(domainiter);
	domainiter = nr->domains.lower_bound(getDB_ID(dom));
      } else {
	++domainiter;
      }
    }
    if(nr->domains.size() == 0){
      const VarLabel* var = nr->label;
      delete nr;
      names.erase(nameiter);
      nameiter = names.lower_bound(var);
    } else {
      ++nameiter;
    }
  }
}

template<class VarType, class DomainType>
DWDatabase<VarType, DomainType>::NameRecord::NameRecord(const VarLabel* label)
   : label(label)
{
}

template<class VarType, class DomainType>
DWDatabase<VarType, DomainType>::NameRecord::~NameRecord()
{
   for(typename domainDBtype::iterator iter = domains.begin();
       iter != domains.end(); iter++){
      delete iter->second;
   }   
}

template<class VarType, class DomainType>
DWDatabase<VarType, DomainType>::DomainRecord::DomainRecord(const DomainType* dom)
  : dom(dom), count(0)
{
}

template <class VarType, class DomainType>
void
DWDatabase<VarType, DomainType>::DomainRecord::putVar( int matlIndex,
						       VarType* var,
						       bool replace )
{
  if(matlIndex+1 >= (int)vars.size()){
    vars.resize(matlIndex+2);
  }
  
  VarType* oldVar = vars[matlIndex+1].var;

  //dataDBtype::iterator iter = find( vars.begin(), vars.end(), matlIndex+1 );
  //if( iter == vars.end() )
  
  if (oldVar != 0) {
    if (!replace) {
      SCI_THROW(InternalError("Put replacing old variable"));
    }

    // replace
    ASSERT(oldVar != var);
    delete oldVar;
    if (var == 0) count--;
  }
  else {
    if (var != 0) count++;
  }
  vars[matlIndex+1].var = var;      
}

template <class VarType, class DomainType>
inline
typename DWDatabase<VarType, DomainType>::DataItem*
DWDatabase<VarType, DomainType>::DomainRecord::getDataItem(int matlIndex)
{
  if (matlIndex+1 < (int)vars.size())
    return &vars[matlIndex+1];
  else
    return 0;
}

template <class VarType, class DomainType>
inline
VarType*
DWDatabase<VarType, DomainType>::DomainRecord::getVar(int matlIndex) const
{
  if (matlIndex+1 < (int)vars.size())
    return vars[matlIndex+1].var;
  else
    return 0;
}

template <class VarType, class DomainType>
inline
void
DWDatabase<VarType, DomainType>::DomainRecord::removeVar(int matlIndex)
{
  ASSERT(matlIndex+1 < (int)vars.size());
  ASSERT(vars[matlIndex+1].var != 0);
  // This assertion is invalid when timestep restarts occur
  //ASSERT(vars[matlIndex+1].scrubCount == 0);
  delete vars[matlIndex+1].var;
  vars[matlIndex+1].var = 0;
  count--;
}

template<class VarType, class DomainType>
DWDatabase<VarType, DomainType>::DomainRecord::~DomainRecord()
{
  for(typename dataDBtype::iterator iter = vars.begin();
      iter != vars.end(); iter++){
    if(iter->var){
      delete iter->var;
    }
  }   
}

template<class VarType, class DomainType>
bool DWDatabase<VarType, DomainType>::exists(const VarLabel* label, int matlIndex,
				 const DomainType* dom) const
{
  int domainid = getDB_ID(dom);
 
  typename nameDBtype::const_iterator nameiter = names.find(label);
  if(nameiter != names.end()) {
    NameRecord* nr = nameiter->second;
    typename domainDBtype::const_iterator domainiter = nr->domains.find(domainid);
    if(domainiter != nr->domains.end()) {
      DomainRecord* rr = domainiter->second;
      if (rr->getVar(matlIndex) != 0) return true;
    }
  }
  return false;
}

template<class VarType, class DomainType>
bool DWDatabase<VarType, DomainType>::exists(const VarLabel* label, const DomainType* dom) const
{
  int domainid = getDB_ID(dom);
  typename nameDBtype::const_iterator nameiter = names.find(label);
  if(nameiter != names.end()) {
    NameRecord* nr = nameiter->second;
    typename domainDBtype::const_iterator domainiter = nr->domains.find(domainid);
    if(domainiter != nr->domains.end()) {
      DomainRecord* rr = domainiter->second;
      for(int i=0; i<(int)rr->getVars().size(); i++){
	if(rr->getVars()[i].var != 0){
	  return true;
	}
      }
    }
  }
  return false;
}

template<class VarType, class DomainType>
void
DWDatabase<VarType, DomainType>::put( const VarLabel* label, 
				      int matlIndex,
				      const DomainType* dom,
				      VarType* var,
				      bool replace )
{
  ASSERT(matlIndex+1 >= 0);
  
  int domainid = getDB_ID(dom);
  typename nameDBtype::iterator nameiter = names.find(label);
  if(nameiter == names.end()){
    names[label] = scinew NameRecord(label);
    nameiter = names.find(label);
  }

  NameRecord* nr = nameiter->second;
  typename domainDBtype::const_iterator domainiter = nr->domains.find(domainid);
  if(domainiter == nr->domains.end()) {
    nr->domains[domainid] = scinew DomainRecord(dom);
    domainiter = nr->domains.find(domainid);
  }
  
  DomainRecord* rr = domainiter->second;

  rr->putVar(matlIndex, var, replace);
}

template<class VarType, class DomainType>
const typename DWDatabase<VarType, DomainType>::DataItem&
DWDatabase<VarType, DomainType>::getDataItem( const VarLabel* label,
					      int matlIndex,
					      const DomainType* dom ) const
{
  ASSERT(matlIndex+1 >= 0);
  
  typename nameDBtype::const_iterator nameiter = names.find(label);
  if(nameiter == names.end())
    SCI_THROW(UnknownVariable(label->getName(), -99, dom, matlIndex,
			      "no variable name"));

  NameRecord* nr = nameiter->second;

  int domainid = getDB_ID(dom);
  typename domainDBtype::const_iterator domainiter = nr->domains.find(domainid);
  if(domainiter == nr->domains.end())
    SCI_THROW(UnknownVariable(label->getName(), -98, dom, matlIndex,
			      "no domain with this variable name"));

  DataItem* dataItem = domainiter->second->getDataItem(matlIndex);
  if (dataItem == 0 || (dataItem->var == 0)) {
    SCI_THROW(UnknownVariable(label->getName(), -97, dom, matlIndex,
			  "no material with this domain and variable name"));
  }
  return *dataItem;
}

template<class VarType, class DomainType>
inline
VarType*
DWDatabase<VarType, DomainType>::get( const VarLabel* label,
				      int matlIndex,
				      const DomainType* dom ) const
{
  const DataItem& dataItem = getDataItem(label, matlIndex, dom);
  ASSERT(dataItem.var != 0); // should have thrown an exception before
  return dataItem.var;
}

template<class VarType, class DomainType>
void
DWDatabase<VarType, DomainType>::get( const VarLabel* label,
				      int matlIndex,
				      const DomainType* dom,
				      VarType& var ) const
{
  VarType* tmp = get(label, matlIndex, dom);
  var.copyPointer(*tmp);
}

template<class VarType, class DomainType>
void DWDatabase<VarType, DomainType>::print(std::ostream& out) const
{
  for(typename nameDBtype::const_iterator nameiter = names.begin();
      nameiter != names.end(); nameiter++){
    NameRecord* nr = nameiter->second;
    out << nr->label->getName() << '\n';
    for(typename domainDBtype::const_iterator domainiter = nr->domains.begin();
	domainiter != nr->domains.end();domainiter++){
      DomainRecord* rr = domainiter->second;
      out <<  "  " << getDB_ID(rr->getDomain()) << '\n';
      for(int i=0;i<(int)rr->getVars().size();i++){
	if(rr->getVars()[i].var){
	  out << "    Material " << i-1 << '\n';
	}
      }
    }
  }
}

template<class VarType, class DomainType>
void
DWDatabase<VarType, DomainType>::logMemoryUse(ostream& out, unsigned long& total,
				  const std::string& tag, int dwid)
{
  for(typename nameDBtype::iterator iter = names.begin();
      iter != names.end(); iter++){
    NameRecord* nr = iter->second;
    for(typename domainDBtype::iterator iter = nr->domains.begin();
	iter != nr->domains.end(); iter++){
      DomainRecord* pr = iter->second;
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
		    pr->getDomain(), i, elems, totsize, ptr, dwid);
	}
      }
    }
  }
}

} // End namespace Uintah

#endif
