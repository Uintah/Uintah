#ifndef UINTAH_HOMEBREW_PARTICLEDATA_H
#define UINTAH_HOMEBREW_PARTICLEDATA_H

#include <Uintah/Grid/RefCounted.h>
#include <Uintah/Grid/ParticleSet.h>
#include <vector>

namespace Uintah {
   
   template<class T>
      class ParticleVariable;

/**************************************

CLASS
   ParticleData
   
GENERAL INFORMATION

   ParticleData.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   ParticleData

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   template<class T> class ParticleData : public RefCounted {
   public:
      ParticleData();
      ParticleData(const ParticleData<T>&);
      ParticleData<T>& operator=(const ParticleData<T>&);
      virtual ~ParticleData();
      
      //////////
      // Insert Documentation Here:
      void add(particleIndex idx, const T& value);
      
      //////////
      // Insert Documentation Here:
      void resize(int newSize) {
	 data.resize(newSize);
      }
      
   private:
      friend class ParticleVariable<T>;
      
      //////////
      // Insert Documentation Here:
      std::vector<T> data;
   };
   
   template<class T>
      ParticleData<T>::ParticleData()
      {
      }
   
   template<class T>
      void ParticleData<T>::add(particleIndex idx, const T& value)
      {
	 if(idx != data.size())
	    throw ParticleException("add, not at the end");
	 data.push_back(value);
      }
   
   template<class T>
      ParticleData<T>::~ParticleData()
      {
      }
   
} // end namespace Uintah

//
// $Log$
// Revision 1.5  2000/04/26 06:48:51  sparker
// Streamlined namespaces
//
// Revision 1.4  2000/04/11 07:10:50  sparker
// Completing initialization and problem setup
// Finishing Exception modifications
//
// Revision 1.3  2000/03/16 22:07:59  dav
// Added the beginnings of cocoon docs.  Added namespaces.  Did a few other coding standards updates too
//
//

#endif
