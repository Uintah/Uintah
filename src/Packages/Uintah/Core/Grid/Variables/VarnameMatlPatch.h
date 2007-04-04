
#ifndef UINTAH_HOMEBREW_VarnameMatlPatch_H
#define UINTAH_HOMEBREW_VarnameMatlPatch_H

#include <string>
#include <Core/Containers/HashTable.h>

namespace Uintah {
class Patch;

    /**************************************
      
      struct
        VarnameMatlPatch
      
        Variable name, Material, and Patch
	
      
      GENERAL INFORMATION
      
        VarnameMatlPatch.h
      
        Wayne Witzel
        Department of Computer Science
        University of Utah
      
        Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
      
        Copyright (C) 2002 SCI Group
      ****************************************/

struct VarnameMatlPatch {
  VarnameMatlPatch(const std::string name, int matlIndex, int patchid)
    : name_(name), matlIndex_(matlIndex), patchid_(patchid)
  {
    hash_ = (unsigned int)(((unsigned int)patchid_<<2)
      ^(string_hash(name_.c_str()))
      ^matlIndex_);
  }

  VarnameMatlPatch(const VarnameMatlPatch& copy)
    : name_(copy.name_), matlIndex_(copy.matlIndex_), patchid_(copy.patchid_), hash_(copy.hash_)
  {}
  VarnameMatlPatch& operator=(const VarnameMatlPatch& copy)
  {
    name_=copy.name_; matlIndex_=copy.matlIndex_; patchid_=copy.patchid_; hash_=copy.hash_;
    return *this;
  }
  
  bool operator<(const VarnameMatlPatch& other) const
  {
    if (name_ == other.name_) {
      if (matlIndex_ == other.matlIndex_)
	return patchid_ < other.patchid_;
      else
	return matlIndex_ < other.matlIndex_;
    }
    else {
      return name_ < name_;
    }
  }
 
  bool operator==(const VarnameMatlPatch& other) const
  {
    return name_ == other.name_ && patchid_ == other.patchid_ && matlIndex_ == other.matlIndex_;
  }

  unsigned int string_hash(const char* p) const {
    unsigned int sum=0;
    while(*p)
      sum = sum*7 + (unsigned char)*p++;
    return sum;
  }

  int hash(int hash_size) const
  {
    return hash_ % hash_size;
  }

  std::string name_;
  int matlIndex_;
  int patchid_;    
  unsigned hash_;
};  

} // End namespace Uintah

#endif
