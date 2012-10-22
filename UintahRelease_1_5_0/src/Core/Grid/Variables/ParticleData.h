/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

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
