#ifndef UINTAH_HOMEBREW_DWDatabase_H
#define UINTAH_HOMEBREW_DWDatabase_H

#include <map>
#include <vector>

namespace Uintah {
   
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

   bool exists(const VarLabel* label, int matlIndex, const Region* region) const;
   void put(const VarLabel* label, int matlindex, const Region* region,
	    const VarType& var);
private:
   typedef std::vector<VarType*> dataDBtype;

   struct RegionRecord {
      const Region* region;
      dataDBtype vars;

      RegionRecord(const Region*);
      ~RegionRecord();
   };

   typedef std::map<const Region*, RegionRecord*> regionDBtype;
   struct NameRecord {
      const VarLabel* label;
      regionDBtype regions;

      NameRecord(const VarLabel* label);
      ~NameRecord();
   };

   typedef std::map<const VarLabel*, NameRecord*> nameDBtype;
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
DWDatabase<VarType>::NameRecord::NameRecord(const VarLabel* label)
   : label(label)
{
}

template<class VarType>
DWDatabase<VarType>::NameRecord::~NameRecord()
{
   for(regionDBtype::iterator iter = regions.begin();
       iter != regions.end(); iter++){
      delete iter->second;
   }   
}

template<class VarType>
DWDatabase<VarType>::RegionRecord::RegionRecord(const Region* region)
   : region(region)
{
}

template<class VarType>
DWDatabase<VarType>::RegionRecord::~RegionRecord()
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
				 const Region* region) const
{
   nameDBtype::const_iterator nameiter = names.find(label);
   if(nameiter != names.end()) {
      NameRecord* nr = nameiter->second;
      regionDBtype::const_iterator regioniter = nr->regions.find(region);
      if(regioniter != nr->regions.end()) {
	 RegionRecord* rr = regioniter->second;
	 if(matlIndex > 0 && matlIndex < rr->vars.size())
	    if(rr->vars[matlIndex] != 0)
	       return true;
      }
   }
   return false;
}

template<class VarType>
void DWDatabase<VarType>::put(const VarLabel* label, int matlIndex,
			      const Region* region,
			      const VarType& var)
{
   nameDBtype::const_iterator nameiter = names.find(label);
   if(nameiter == names.end()){
      names[label] = new NameRecord(label);
      nameiter = names.find(label);
   }

   NameRecord* nr = nameiter->second;
   regionDBtype::const_iterator regioniter = nr->regions.find(region);
   if(regioniter == nr->regions.end()) {
      nr->regions[region] = new RegionRecord(region);
      regioniter = nr->regions.find(region);
   }

   RegionRecord* rr = regioniter->second;
   if(matlIndex < 0)
      throw InternalError("matlIndex must be >= 0");

   if(matlIndex >= rr->vars.size()){
      unsigned long oldSize = rr->vars.size();
      rr->vars.resize(matlIndex+1);
      for(unsigned long i=oldSize;i<matlIndex;i++)
	 rr->vars[i]=0;
   }

   if(rr->vars[matlIndex])
      throw InternalError("Put replacing old variable");

   rr->vars[matlIndex]=var.clone();
}

} // end namespace Uintah

//
// $Log$
// Revision 1.1  2000/04/28 07:36:13  sparker
// Utility class for datawarehousen
//
//

#endif

