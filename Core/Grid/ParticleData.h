#ifndef UINTAH_HOMEBREW_PARTICLEDATA_H
#define UINTAH_HOMEBREW_PARTICLEDATA_H

#include <Packages/Uintah/Core/ProblemSpec/RefCounted.h>
#include <Packages/Uintah/Core/Grid/ParticleSet.h>

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
      ParticleData(particleIndex size);
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
      ParticleData<T>::ParticleData(particleIndex size)
      : data(size)
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

   template<class T>
   ParticleData<T>& ParticleData<T>::operator=(const ParticleData<T>& copy)
   {
     data = copy.data;
     return *this;
   }

} // End namespace Uintah
   
#endif
