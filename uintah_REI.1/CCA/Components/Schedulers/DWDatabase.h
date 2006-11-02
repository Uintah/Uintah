#ifndef UINTAH_HOMEBREW_DWDatabase_H
#define UINTAH_HOMEBREW_DWDatabase_H

#include <Packages/Uintah/Core/Grid/UnknownVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/VarLabel.h>
#include <Packages/Uintah/CCA/Components/Schedulers/MemoryLog.h>
#include <Packages/Uintah/Core/Grid/Variables/VarLabelMatl.h>
#include <Packages/Uintah/Core/Grid/Variables/ScrubItem.h>

#include <Packages/Uintah/Core/Parallel/Parallel.h>

#include <Core/Containers/FastHashTable.h>
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
using SCIRun::FastHashTable;

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
 template<class DomainType>
   class DWDatabase {
   public:
   DWDatabase();
   ~DWDatabase();

   bool exists(const VarLabel* label, int matlIndex, const DomainType* dom) const;
   void put(const VarLabel* label, int matlindex, const DomainType* dom,
	    Variable* var, bool replace);
   void get(const VarLabel* label, int matlindex, const DomainType* dom,
	    Variable& var) const;
   inline Variable* get(const VarLabel* label, int matlindex,
		const DomainType* dom) const;
   void print(ostream&, int rank) const;
   void cleanForeign();

   // Scrub counter manipulator functions -- when the scrub count goes to
   // zero, the data is scrubbed.  Return remaining count

   // How Scrubbing works:  If a variable is in the OldDW, at the beginning of the timestep
   // initializeScrubs will be called for each of those variables.  For each variable computed 
   // or copied via MPI, setScrubCount will be called on it, based on the scrubCountTable in
   // DetailedTasks.  Then, when the variable is used, decrementScrubCount is called on it
   // and if the count reaches zero, it is scrubbed.
   int decrementScrubCount(const VarLabel* label, int matlindex,
			    const DomainType* dom);
   void setScrubCount(const VarLabel* label, int matlindex,
		      const DomainType* dom, int count);
   void scrub(const VarLabel* label, int matlindex, const DomainType* dom);
   
   // add means increment the scrub count instead of setting it.  This is for when a DW
   // can act as a CoarseOldDW as well as an OldDW
   void initializeScrubs(int dwid, const FastHashTable<ScrubItem>* scrubcounts, bool add);

   void logMemoryUse(ostream& out, unsigned long& total,
		     const std::string& tag, int dwid);

   void getVarLabelMatlTriples(vector<VarLabelMatl<DomainType> >& vars) const;


private:
   struct DataItem {
     DataItem()
       : var(0), scrubCount(0) {}
     Variable* var;
     int scrubCount;
   };
  
   const DataItem& getDataItem(const VarLabel* label, int matlindex,
			       const DomainType* dom) const;
    
   typedef map<VarLabelMatl<DomainType>, DataItem> varDBtype;
   varDBtype vars;

   DWDatabase(const DWDatabase&);
   DWDatabase& operator=(const DWDatabase&);      
};

template<class DomainType>
DWDatabase<DomainType>::DWDatabase()
{
}

template<class DomainType>
DWDatabase<DomainType>::~DWDatabase()
{
  for(typename varDBtype::iterator iter = vars.begin();
      iter != vars.end(); iter++){

#ifdef DEBUG
    // This can happen in some normal cases (especially at program
    // shutdown), but catching it is useful for debugging the scrubbing
    // stuff...
    if (iter->first->typeDescription() != 0
	&& iter->first->typeDescription()->getType() != TypeDescription::ReductionVariable) {
      ostringstream msgstr;
      msgstr << "Failed to scrub: " << iter->first->getName()
	     << " completely";
      SCI_THROW(InternalError(msgstr.str(), __FILE__, __LINE__));
    }
#endif
    delete iter->second.var;
  }
}

template<class DomainType>
void
DWDatabase<DomainType>::cleanForeign()
{
  for(typename varDBtype::iterator iter = vars.begin();
      iter != vars.end();){
    const Variable* var = iter->second.var;
    if(var && var->isForeign()){
      delete var;
      typename varDBtype::iterator deliter = iter;
      iter++;
      vars.erase(deliter);
    }
    else
      iter++;
  }
}

template<class DomainType>
int DWDatabase<DomainType>::
decrementScrubCount(const VarLabel* label, int matlIndex, const DomainType* dom)
{
  DataItem& data = const_cast<DataItem&>(getDataItem(label, matlIndex, dom));
  // Dav's conjectures on how this works:
  //   setScrubCount is called the first time with "count" set to some X.  
  //   This X represents the number of tasks that will use the var.  Later,
  //   after a task has used the var, it will call decrementScrubCount
  //   If scrubCount then is equal to 0, the var is scrubbed.

  USE_IF_ASSERTS_ON(if (data.scrubCount <= 0) { cerr << "Var: " << *label << " matl " << matlIndex << " patch " << dom->getID() << endl; })
  ASSERT(data.scrubCount > 0);
  int count = data.scrubCount-1;
  if(!--data.scrubCount)
    scrub(label, matlIndex, dom);
  return count;
}

template<class DomainType>
void DWDatabase<DomainType>::
setScrubCount(const VarLabel* label, int matlIndex, const DomainType* dom, int count)
{
  DataItem& data = const_cast<DataItem&>(getDataItem(label, matlIndex, dom));
  ASSERT(data.var != 0); // should have thrown an exception before
  if(data.scrubCount == 0)
    data.scrubCount = count;
}

template<class DomainType>
void
DWDatabase<DomainType>::scrub(const VarLabel* var, int matlIndex, const DomainType* dom)
{
  ASSERT(matlIndex >= -1);
  VarLabelMatl<DomainType> vlm(var, matlIndex, getRealDomain(dom));
  typename varDBtype::iterator iter = vars.find(vlm);
  if (iter != vars.end()) {
    ASSERTEQ(iter->second.scrubCount, 0);
    delete iter->second.var;
    vars.erase(iter);
    return; // found and scrubbed
  }

  // scrub not found
  ostringstream msgstr;
  msgstr << var->getName() << ", matl " << matlIndex
	 << ", patch/level " << dom->getID()
	 << " not found for scrubbing.";

  SCI_THROW(InternalError(msgstr.str(), __FILE__, __LINE__));
}

template<class DomainType>
void
DWDatabase<DomainType>::initializeScrubs(int dwid, const FastHashTable<ScrubItem>* scrubcounts, bool add)
{
  // loop over each variable, probing the scrubcount map. Set the
  // scrubcount appropriately.  if the variable has no entry in
  // the scrubcount map, delete it
  for(typename varDBtype::iterator variter = vars.begin();
      variter != vars.end();){
    if(variter->second.var){
      VarLabelMatl<DomainType> vlm = variter->first;
      // See if it is in the scrubcounts map.
      ScrubItem key(vlm.label_, vlm.matlIndex_, vlm.domain_, dwid);
      ScrubItem* result = scrubcounts->lookup(&key);
      if(!result && !add){
        delete variter->second.var;
        typename varDBtype::iterator deliter = variter;
        variter++;
        vars.erase(deliter);
      } else {
        if (result){
          if (add)
            variter->second.scrubCount += result->count;
          else {
            variter->second.scrubCount = result->count;
          }
        }
        variter++;
      }
    }
    else {
      variter++;
    }
  }
}

template<class DomainType>
bool DWDatabase<DomainType>::exists(const VarLabel* label, int matlIndex, const DomainType* dom) const
{
  return vars.find(VarLabelMatl<DomainType>(label, matlIndex, getRealDomain(dom))) != vars.end();
}

template<class DomainType>
void
DWDatabase<DomainType>::put( const VarLabel* label, int matlIndex,const DomainType* dom,
				      Variable* var, bool replace )
{
  ASSERT(matlIndex >= -1);
  
  VarLabelMatl<DomainType> v(label, matlIndex, getRealDomain(dom));
  DataItem& di = vars[v]; 
  if (di.var) {
    if (!replace) {
      SCI_THROW(InternalError("Put replacing old variable", __FILE__, __LINE__));
    }
    ASSERT(di.var != var);
    delete di.var;
  }
  di.var = var;      
}

template<class DomainType>
const typename DWDatabase<DomainType>::DataItem&
DWDatabase<DomainType>::getDataItem( const VarLabel* label, int matlIndex, const DomainType* dom ) const
{
  ASSERT(matlIndex >= -1);
    
  VarLabelMatl<DomainType> v(label, matlIndex, getRealDomain(dom));
  typename varDBtype::const_iterator iter = vars.find(v);
  if(iter == vars.end())
    SCI_THROW(UnknownVariable(label->getName(), -99, dom, matlIndex,
			      "DWDatabase::getDataItem", __FILE__, __LINE__));

  return iter->second;
}

template<class DomainType>
inline
Variable*
DWDatabase<DomainType>::get( const VarLabel* label, int matlIndex, const DomainType* dom ) const
{
  const DataItem& dataItem = getDataItem(label, matlIndex, dom);
  ASSERT(dataItem.var != 0); // should have thrown an exception before
  return dataItem.var;
}

template<class DomainType>
void
DWDatabase<DomainType>::get( const VarLabel* label,
				      int matlIndex,
				      const DomainType* dom,
				      Variable& var ) const
{
  Variable* tmp = get(label, matlIndex, dom);
  var.copyPointer(*tmp);
}

template<class DomainType>
void DWDatabase<DomainType>::print(std::ostream& out, int rank) const
{
  for(typename varDBtype::const_iterator variter = vars.begin();
      variter != vars.end(); variter++){
    const VarLabelMatl<DomainType>& vlm = variter->first;
    out << rank << " " << vlm.label_->getName() << "  " << (vlm.domain_?vlm.domain_->getID():0) << "  " << vlm.matlIndex_ << '\n';
  }
}

template<class DomainType>
void
DWDatabase<DomainType>::logMemoryUse(ostream& out, unsigned long& total, const std::string& tag, int dwid)
{
  for(typename varDBtype::const_iterator variter = vars.begin();
      variter != vars.end(); variter++){
    Variable* var = variter->second.var;
    if(var){
      VarLabelMatl<DomainType> vlm = variter->first;
      const VarLabel* label = vlm.label_;
      string elems;
      unsigned long totsize;
      void* ptr;
      var->getSizeInfo(elems, totsize, ptr);
      const TypeDescription* td = label->typeDescription();
      logMemory(out, total, tag, label->getName(), (td?td->getName():"-"),
		    vlm.domain_, vlm.matlIndex_, elems, totsize, ptr, dwid);
    }
  }
}

template<class DomainType>
void
DWDatabase<DomainType>::getVarLabelMatlTriples( vector<VarLabelMatl<DomainType> >& v ) const
{
  for(typename varDBtype::const_iterator variter = vars.begin();
      variter != vars.end(); variter++){
    const VarLabelMatl<DomainType>& vlm = variter->first;
    if(variter->second.var){
      v.push_back(vlm);
    }
  }
}

} // End namespace Uintah

#endif
