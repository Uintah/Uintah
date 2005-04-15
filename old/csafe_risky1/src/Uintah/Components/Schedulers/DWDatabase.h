#ifndef UINTAH_HOMEBREW_DWDatabase_H
#define UINTAH_HOMEBREW_DWDatabase_H

#include <map>
#include <vector>
#include <SCICore/Exceptions/InternalError.h>
#include <Uintah/Exceptions/UnknownVariable.h>
#include <Uintah/Grid/VarLabel.h>
#include <SCICore/Malloc/Allocator.h>
#include <iosfwd>

namespace Uintah {
   using SCICore::Exceptions::InternalError;

   using std::vector;
   using std::iostream;
   
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
template<class VarType> class DWDatabase {
public:
   DWDatabase();
   ~DWDatabase();

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
   nameDBtype::const_iterator nameiter = names.find(label);
   if(nameiter != names.end()) {
      NameRecord* nr = nameiter->second;
      patchDBtype::const_iterator patchiter = nr->patches.find(patch);
      if(patchiter != nr->patches.end()) {
	 PatchRecord* rr = patchiter->second;
	 if(matlIndex >= 0 && matlIndex < (int)rr->vars.size()){
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
   return false;
}

template<class VarType>
void DWDatabase<VarType>::put(const VarLabel* label, int matlIndex,
			      const Patch* patch,
			      VarType* var,
			      bool replace)
{
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
   if(matlIndex < 0)
      throw InternalError("matlIndex must be >= 0");

   if(matlIndex >= (int)rr->vars.size()){
      int oldSize = (int)rr->vars.size();
      rr->vars.resize(matlIndex+1);
      for(int i=oldSize;i<matlIndex;i++)
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
   nameDBtype::const_iterator nameiter = names.find(label);
   if(nameiter == names.end())
      throw UnknownVariable(label->getName(), patch->getID(),
			    patch->toString(), matlIndex,
			    "no variable name");

   NameRecord* nr = nameiter->second;

   patchDBtype::const_iterator patchiter = nr->patches.find(patch);
   if(patchiter == nr->patches.end())
      throw UnknownVariable(label->getName(), patch->getID(),
			    patch->toString(), matlIndex,
			    "no patch with this variable name");

   PatchRecord* rr = patchiter->second;
   if(matlIndex < 0)
      throw InternalError("matlIndex must be >= 0");

   if(matlIndex >= (int)rr->vars.size())
      throw UnknownVariable(label->getName(), patch->getID(),
			    patch->toString(), matlIndex,
			    "no material with this patch and variable name");

   if(!rr->vars[matlIndex])
      throw UnknownVariable(label->getName(), patch->getID(),
			    patch->toString(), matlIndex,
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
	       cerr << "    Material " << i << '\n';
	    }
	 }
      }
   }
}

} // end namespace Uintah

//
// $Log$
// Revision 1.16.2.3  2000/10/19 05:17:55  sparker
// Merge changes from main branch into csafe_risky1
//
// Revision 1.16.2.2  2000/10/10 05:28:03  sparker
// Added support for NullScheduler (used for profiling taskgraph overhead)
//
// Revision 1.16.2.1  2000/09/29 06:09:54  sparker
// g++ warnings
// Support for sending only patch edges
//
// Revision 1.17  2000/10/13 20:46:39  sparker
// Clean out foreign variables at finalize time
//
// Revision 1.16  2000/09/28 23:16:45  jas
// Added (int) for anything returning the size of a STL component.  Added
// <algorithm> and using std::find.  Other minor modifications to get
// rid of warnings for g++.
//
// Revision 1.15  2000/09/27 02:07:18  dav
// cosmetics and printout fix
//
// Revision 1.14  2000/09/25 20:43:44  sparker
// Quiet g++ warnings
//
// Revision 1.13  2000/07/27 22:39:47  sparker
// Implemented MPIScheduler
// Added associated support
//
// Revision 1.12  2000/06/19 22:37:18  sparker
// Improved messages for UnknownVariable
//
// Revision 1.11  2000/06/15 21:57:11  sparker
// Added multi-patch support (bugzilla #107)
// Changed interface to datawarehouse for particle data
// Particles now move from patch to patch
//
// Revision 1.10  2000/05/31 17:55:50  sparker
// Fixed database so that carryForward won't die when there are no
// particles.
//
// Revision 1.9  2000/05/30 20:19:23  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
// Revision 1.8  2000/05/21 20:10:49  sparker
// Fixed memory leak
// Added scinew to help trace down memory leak
// Commented out ghost cell logic to speed up code until the gc stuff
//    actually works
//
// Revision 1.7  2000/05/10 20:02:52  sparker
// Added support for ghost cells on node variables and particle variables
//  (work for 1 patch but not debugged for multiple)
// Do not schedule fracture tasks if fracture not enabled
// Added fracture directory to MPM sub.mk
// Be more uniform about using IntVector
// Made patches have a single uniform index space - still needs work
//
// Revision 1.6  2000/05/07 06:02:07  sparker
// Added beginnings of multiple patch support and real dependencies
//  for the scheduler
//
// Revision 1.5  2000/05/06 03:54:10  sparker
// Fixed multi-material carryForward
//
// Revision 1.4  2000/05/02 06:07:16  sparker
// Implemented more of DataWarehouse and SerialMPM
//
// Revision 1.3  2000/05/01 16:18:16  sparker
// Completed more of datawarehouse
// Initial more of MPM data
// Changed constitutive model for bar
//
// Revision 1.2  2000/04/28 21:12:05  jas
// Added some includes to get it to compile on linux.
//
// Revision 1.1  2000/04/28 07:36:13  sparker
// Utility class for datawarehousen
//
//

#endif

