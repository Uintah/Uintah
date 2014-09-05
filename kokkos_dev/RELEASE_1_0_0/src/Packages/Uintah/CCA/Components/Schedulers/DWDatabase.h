#ifndef UINTAH_HOMEBREW_DWDatabase_H
#define UINTAH_HOMEBREW_DWDatabase_H

#include <Packages/Uintah/Core/Grid/UnknownVariable.h>
#include <Packages/Uintah/Core/Grid/VarLabel.h>

#include <Core/Exceptions/InternalError.h>
#include <Core/Malloc/Allocator.h>

#include <map>
#include <vector>
#include <iosfwd>
#include <list>

namespace Uintah {

using std::vector;
using std::iostream;

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
   VarType* get(const VarLabel* label, int matlindex,
		const Patch* patch) const;
   void copyAll(const DWDatabase& from, const VarLabel*, const Patch* patch);
   void print(ostream&);
   void cleanForeign();
private:
   typedef vector<VarType*> dataDBtype;

   struct PatchRecord {
      const Patch* patch;
      dataDBtype vars;

      PatchRecord(const Patch*);
      ~PatchRecord();
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

   typedef map<const VarLabel*, VarType*, VarLabel::Compare> globalDBtype;
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
   for(nameDBtype::iterator iter = names.begin();
       iter != names.end(); iter++){
      delete iter->second;
   }
   for(globalDBtype::iterator iter = globals.begin();
       iter != globals.end(); iter++){
      delete iter->second;
   }
}

template<class VarType>
void
DWDatabase<VarType>::cleanForeign()
{
   for(nameDBtype::iterator iter = names.begin();
       iter != names.end(); iter++){
      NameRecord* nr = iter->second;
      for(patchDBtype::iterator iter = nr->patches.begin();
	  iter != nr->patches.end(); iter++){
	 PatchRecord* pr = iter->second;
	 for(dataDBtype::iterator iter = pr->vars.begin();
	     iter != pr->vars.end(); iter++){
	    VarType* var = *iter;
	    if(var && var->isForeign()){
	       delete var;
	       *iter=0;
	    }
	 }
      }
   }

   list<const VarLabel*> toBeRemoved;
   for(globalDBtype::iterator iter = globals.begin();
       iter != globals.end(); iter++){
      VarType* var = iter->second;
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
DWDatabase<VarType>::NameRecord::NameRecord(const VarLabel* label)
   : label(label)
{
}

template<class VarType>
DWDatabase<VarType>::NameRecord::~NameRecord()
{
   for(patchDBtype::iterator iter = patches.begin();
       iter != patches.end(); iter++){
      delete iter->second;
   }   
}

template<class VarType>
DWDatabase<VarType>::PatchRecord::PatchRecord(const Patch* patch)
   : patch(patch)
{
}

template<class VarType>
DWDatabase<VarType>::PatchRecord::~PatchRecord()
{
   for(dataDBtype::iterator iter = vars.begin();
       iter != vars.end(); iter++){
      if(*iter){
	 delete *iter;
      }
   }   
}

template<class VarType>
bool DWDatabase<VarType>::exists(const VarLabel* label, int matlIndex,
				 const Patch* patch) const
{
   if (matlIndex < 0)
      return (patch == NULL) && (globals.find(label) != globals.end());
  
   nameDBtype::const_iterator nameiter = names.find(label);
   if(nameiter != names.end()) {
      NameRecord* nr = nameiter->second;
      patchDBtype::const_iterator patchiter = nr->patches.find(patch);
      if(patchiter != nr->patches.end()) {
	 PatchRecord* rr = patchiter->second;
	 if(matlIndex < (int)rr->vars.size()){
	    if(rr->vars[matlIndex] != 0){
	       return true;
	    }
	 }
      }
   }
   return false;
}

template<class VarType>
bool DWDatabase<VarType>::exists(const VarLabel* label, const Patch* patch) const
{
   nameDBtype::const_iterator nameiter = names.find(label);
   if(nameiter != names.end()) {
      NameRecord* nr = nameiter->second;
      patchDBtype::const_iterator patchiter = nr->patches.find(patch);
      if(patchiter != nr->patches.end()) {
	 PatchRecord* rr = patchiter->second;
	 for(int i=0; i<(int)rr->vars.size(); i++){
	    if(rr->vars[i] != 0){
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
   if(matlIndex < 0) {
      if (patch == NULL) {
         // add to globals
         globalDBtype::const_iterator globaliter = globals.find(label);
         if ((globaliter == globals.end()) || replace) {
	    VarType*& globalVar = globals[label];
	    if (globalVar != NULL)
	       delete globalVar;
	    globalVar = var;
	    return;
	 }
         else
	    throw InternalError("Put replacing old global variable");
      }
      else
         throw InternalError("matlIndex must be >= 0");
   }

  
   nameDBtype::const_iterator nameiter = names.find(label);
   if(nameiter == names.end()){
      names[label] = scinew NameRecord(label);
      nameiter = names.find(label);
   }

   NameRecord* nr = nameiter->second;
   patchDBtype::const_iterator patchiter = nr->patches.find(patch);
   if(patchiter == nr->patches.end()) {
      nr->patches[patch] = scinew PatchRecord(patch);
      patchiter = nr->patches.find(patch);
   }

   PatchRecord* rr = patchiter->second;
      
   if(matlIndex >= (int)rr->vars.size()){
      int oldSize = (int)rr->vars.size();
      rr->vars.resize(matlIndex+1);
      for(unsigned long i=oldSize;i<(unsigned long)matlIndex;i++)
	 rr->vars[i]=0;
   }

   if(rr->vars[matlIndex]){
      if(replace)
	 delete rr->vars[matlIndex];
      else
	 throw InternalError("Put replacing old variable");
   }

   rr->vars[matlIndex]=var;

}

template<class VarType>
VarType* DWDatabase<VarType>::get(const VarLabel* label, int matlIndex,
				  const Patch* patch) const
{
   if(matlIndex < 0) {
      if (patch == NULL) {
         // get from globals
         globalDBtype::const_iterator globaliter = globals.find(label);
         if (globaliter != globals.end())
	    return (*globaliter).second;
         else
	    throw UnknownVariable(label->getName(),
                                  "no global variable with this name");
      }
      else
         throw InternalError("matlIndex must be >= 0");
   }
  
   nameDBtype::const_iterator nameiter = names.find(label);
   if(nameiter == names.end())
      throw UnknownVariable(label->getName(), patch, matlIndex,
			    "no variable name");

   NameRecord* nr = nameiter->second;

   patchDBtype::const_iterator patchiter = nr->patches.find(patch);
   if(patchiter == nr->patches.end())
      throw UnknownVariable(label->getName(), patch, matlIndex,
			    "no patch with this variable name");

   PatchRecord* rr = patchiter->second;

   if(matlIndex >= (int)rr->vars.size())
      throw UnknownVariable(label->getName(), patch, matlIndex,
			    "no material with this patch and variable name");

   if(!rr->vars[matlIndex])
      throw UnknownVariable(label->getName(), patch, matlIndex,
			    "no material with this patch and variable name");

   return rr->vars[matlIndex];
}

template<class VarType>
void DWDatabase<VarType>::get(const VarLabel* label, int matlIndex,
			      const Patch* patch,
			      VarType& var) const
{
   VarType* tmp = get(label, matlIndex, patch);
   var.copyPointer(*tmp);
}

template<class VarType>
void DWDatabase<VarType>::copyAll(const DWDatabase& from,
				  const VarLabel* label,
				  const Patch* patch)
{
   nameDBtype::const_iterator nameiter = from.names.find(label);
   if(nameiter == from.names.end())
      return;

   NameRecord* nr = nameiter->second;
   patchDBtype::const_iterator patchiter = nr->patches.find(patch);
   if(patchiter == nr->patches.end())
      return;

   PatchRecord* rr = patchiter->second;

   for(int i=0;i<rr->vars.size();i++){
      if(rr->vars[i])
	 put(label, i, patch, rr->vars[i]->clone(), false);
   }

   globals = from.globals;
}

template<class VarType>
void DWDatabase<VarType>::print(std::ostream& out)
{
   for(nameDBtype::const_iterator nameiter = names.begin();
       nameiter != names.end(); nameiter++){
      NameRecord* nr = nameiter->second;
      out << nr->label->getName() << '\n';
      for(patchDBtype::const_iterator patchiter = nr->patches.begin();
	  patchiter != nr->patches.end();patchiter++){
	 PatchRecord* rr = patchiter->second;
	 out <<  "  " << *(rr->patch) << '\n';
	 for(int i=0;i<(int)rr->vars.size();i++){
	    if(rr->vars[i]){
	       out << "    Material " << i << '\n';
	    }
	 }
      }
   }

   for (globalDBtype::iterator globaliter = globals.begin();
	globaliter != globals.end(); globaliter++)
     out << (*globaliter).first->getName() << '\n';
}

} // End namespace Uintah

#endif
