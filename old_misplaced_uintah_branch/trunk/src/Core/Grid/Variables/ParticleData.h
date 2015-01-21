#ifndef UINTAH_HOMEBREW_PARTICLEDATA_H
#define UINTAH_HOMEBREW_PARTICLEDATA_H

#include <Core/Util/RefCounted.h>

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
      virtual ~ParticleData();

      //////////
      // Insert Documentation Here:
      void resize(int newSize) {
	T* newdata = scinew T[newSize];
	if(data){
	  int smaller = ((newSize < size ) ? newSize:size);
	  for(int i = 0; i < smaller; i++)
	    newdata[i] = data[i];
	  delete[] data;
	}
	data = newdata;
	size = newSize;
      }

   private:
      ParticleData(const ParticleData<T>&);
      ParticleData<T>& operator=(const ParticleData<T>&);
      friend class ParticleVariable<T>;
      
      //////////
      // Insert Documentation Here:
      T* data;
      particleIndex size;
   };
   
   template<class T>
      ParticleData<T>::ParticleData()
      {
	data=0;
      }
   
   template<class T>
     ParticleData<T>::ParticleData(particleIndex size)
     : size(size)
      {
	data = scinew T[size];
      }
      
   template<class T>
      ParticleData<T>::~ParticleData()
      {
	if(data)
	  delete[] data;
      }

   template<class T>
     ParticleData<T>& ParticleData<T>::operator=(const ParticleData<T>& copy)
     {
       for(particleIndex i=0;i<size;i++)
	 data[i] = copy.data[i];
       return *this;
     }
} // End namespace Uintah
   
#endif
