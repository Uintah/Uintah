/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#ifndef UINTAH_HOMEBREW_DWDatabase_H
#define UINTAH_HOMEBREW_DWDatabase_H

#include <Core/Grid/UnknownVariable.h>
#include <Core/Grid/Variables/VarLabel.h>
#include <CCA/Components/Schedulers/MemoryLog.h>
#include <CCA/Components/Schedulers/DetailedTasks.h>
#include <Core/Grid/Variables/VarLabelMatl.h>
#include <Core/Grid/Variables/ScrubItem.h>

#include <Core/Parallel/Parallel.h>

#include <Core/Containers/FastHashTable.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/FancyAssert.h>

#include <map>
#include <vector>
#include <iosfwd>
#include <list>
#include <sstream>

#include <sci_hash_map.h>
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

   void clear();
   
   bool exists(const VarLabel* label, int matlIndex, const DomainType* dom) const;
   void put(const VarLabel* label, int matlindex, const DomainType* dom,
	    Variable* var, bool replace);
   void putForeign(const VarLabel* label, int matlindex, const DomainType* dom,
	    Variable* var);
   void get(const VarLabel* label, int matlindex, const DomainType* dom,
	    Variable& var) const;
   void getlist( const VarLabel* label, int matlIndex, const DomainType* dom,
       vector<Variable*>& varlist ) const;
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
       : var(0), scrubCount(0), version(0) {}
     Variable* var;
     int scrubCount;
     unsigned int version; 
   };
  
   const DataItem& getDataItem(const VarLabel* label, int matlindex,
			       const DomainType* dom) const;
    
   typedef hash_multimap<VarLabelMatl<DomainType>, DataItem>  varDBtype;
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
  clear();
}

template<class DomainType>
void DWDatabase<DomainType>::clear()
{
  for(typename varDBtype::iterator iter = vars.begin();
      iter != vars.end(); iter++){

#if 0
    // This can happen in some normal cases (especially at program
    // shutdown), but catching it is useful for debugging the scrubbing
    // stuff...
    if ( iter->first.label_ != 0 && iter->first.label_->typeDescription() != 0
	&& iter->first.label_->typeDescription()->getType() != TypeDescription::ReductionVariable) {
      cout << "Failed to scrub: " << iter->first.label_->getName()
	     << " completely.  scrub count: " << iter->second.scrubCount << endl;
      //SCI_THROW(InternalError("Scubbing Failed"), __FILE__, __LINE__);
    }
#endif
    if (iter->second.var)
      delete iter->second.var;
    iter->second.var=0;
  }
  vars.clear();
}

template<class DomainType>
void
DWDatabase<DomainType>::cleanForeign()
{
  for(typename varDBtype::iterator iter = vars.begin();
      iter != vars.end();){
    if(iter->second.var && iter->second.var->isForeign()){
      delete iter->second.var;
      iter->second.var=0;
      vars.erase(iter++);
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

  USE_IF_ASSERTS_ON(if (data.scrubCount <= 0) { std::cerr << "Var: " << *label << " matl " << matlIndex << " patch " << dom->getID() << std::endl; })
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
DWDatabase<DomainType>::scrub(const VarLabel* label, int matlIndex, const DomainType* dom)
{
  ASSERT(matlIndex >= -1);
  VarLabelMatl<DomainType> v(label, matlIndex, getRealDomain(dom));
#if 0
  if (vars.count(v)==0){ // scrub not found
  ostringstream msgstr;
  msgstr << label->getName() << ", matl " << matlIndex
	 << ", patch/level " << dom->getID()
	 << " not found for scrubbing.";
  SCI_THROW(InternalError(msgstr.str(), __FILE__, __LINE__));
  }
#endif
  std::pair<typename varDBtype::iterator, typename varDBtype::iterator> ret = vars.equal_range(v);
  for (typename varDBtype::iterator iter=ret.first; iter!=ret.second; ++iter){
    if (iter->second.var!=NULL) delete iter->second.var;
    iter->second.var = NULL; //leave a hole in the map instead of erase, readonly to map
  }
  //vars.erase(v);

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
        //vars.erase(variter++);
        //leave a hole in the map instead of erase, read only operation 
        variter->second.var=NULL;  
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
  VarLabelMatl<DomainType> v(label, matlIndex, getRealDomain(dom));
  typename varDBtype::const_iterator iter = vars.find(v);
  if (iter== vars.end()) return false;
  if (iter->second.var==NULL) return false;
  return true;
}

template<class DomainType>
void
DWDatabase<DomainType>::put( const VarLabel* label, int matlIndex,const DomainType* dom,
				      Variable* var, bool replace )
{
  ASSERT(matlIndex >= -1);

  VarLabelMatl<DomainType> v(label, matlIndex, getRealDomain(dom));
  unsigned int count=vars.count(v);
  if (count > 1 ) 
    SCI_THROW(InternalError("More than one vars on this label", __FILE__, __LINE__));
  if (count == 1) {
    typename varDBtype::iterator iter = vars.find(v);
    if (!replace && iter->second.var) 
      SCI_THROW(InternalError("Put replacing old vars", __FILE__, __LINE__));
    else {
      ASSERT(iter->second.var != var);
      if (iter->second.var)
        delete iter->second.var;
      iter->second.var=var; 
    } 
  }
  if (count == 0) {
    typename varDBtype::iterator iter = vars.insert(std::pair<VarLabelMatl<DomainType>, DataItem>(v, DataItem()));
    iter->second.var=var; 
  }
}


template<class DomainType>
void
DWDatabase<DomainType>::putForeign( const VarLabel* label, int matlIndex,const DomainType* dom,
				      Variable* var)
{
  ASSERT(matlIndex >= -1);
  
  VarLabelMatl<DomainType> v(label, matlIndex, getRealDomain(dom));
  typename varDBtype::iterator iter = vars.insert(std::pair<VarLabelMatl<DomainType>, DataItem>(v, DataItem()));
  iter->second.var=var; 
  iter->second.version=vars.count(v)-1;
}

template<class DomainType>
const typename DWDatabase<DomainType>::DataItem&
DWDatabase<DomainType>::getDataItem( const VarLabel* label, int matlIndex, const DomainType* dom ) const
{
  ASSERT(matlIndex >= -1);
  VarLabelMatl<DomainType> v(label, matlIndex, getRealDomain(dom));
  std::pair<typename varDBtype::const_iterator, typename varDBtype::const_iterator> ret = vars.equal_range(v);
  for (typename varDBtype::const_iterator iter=ret.first; iter!=ret.second; ++iter){
    if (iter->second.version == 0 ) {
      return iter->second;
    }
  }
  SCI_THROW(UnknownVariable(label->getName(), -99, dom, matlIndex,
			      "DWDatabase::getDataItem", __FILE__, __LINE__));
}

template<class DomainType>
inline
Variable*
DWDatabase<DomainType>::get( const VarLabel* label, int matlIndex, const DomainType* dom ) const
{
  VarLabelMatl<DomainType> v(label, matlIndex, getRealDomain(dom));
  ASSERT(vars.count(v) == 1 ) // should call getlist() on possible foregin vars
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
void
DWDatabase<DomainType>::getlist( const VarLabel* label,
				      int matlIndex,
				      const DomainType* dom,
				      vector<Variable*>& varlist ) const
{
  VarLabelMatl<DomainType> v(label, matlIndex, getRealDomain(dom));
  std::pair<typename varDBtype::const_iterator, typename varDBtype::const_iterator> ret = vars.equal_range(v);

  varlist.resize(vars.count(v));
  for (typename varDBtype::const_iterator iter=ret.first; iter!=ret.second; ++iter)
    varlist[iter->second.version] = iter->second.var;

  //this function is allowed to return an empty list
  //if(varlist.size() == 0)
  //  SCI_THROW(UnknownVariable(label->getName(), -99, dom, matlIndex,
	//		      "DWDatabase::getlist", __FILE__, __LINE__));

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

//Hash function for VarLabelMatl
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
#else
namespace std { 
  namespace tr1 {
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
  }
} // End namespace std
#endif

#endif
