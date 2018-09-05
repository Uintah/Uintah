/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
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

#ifndef UINTAH_HOMEBREW_UNSTRUCTURED_PARTICLEDATA_H
#define UINTAH_HOMEBREW_UNSTRUCTURED_PARTICLEDATA_H

#include <Core/Util/RefCounted.h>

namespace Uintah {

template<class T>
   class UnstructuredParticleVariable;

/**************************************

CLASS
   UnstructuredParticleData
   
GENERAL INFORMATION

   UnstructuredParticleData.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  

KEYWORDS
   UnstructuredParticleData

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   template<class T> class UnstructuredParticleData : public RefCounted {
   public:
      UnstructuredParticleData();
      UnstructuredParticleData(particleIndex size);
      virtual ~UnstructuredParticleData();

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
      UnstructuredParticleData(const UnstructuredParticleData<T>&);
      UnstructuredParticleData<T>& operator=(const UnstructuredParticleData<T>&);
      friend class UnstructuredParticleVariable<T>;
      
      //////////
      // Insert Documentation Here:
      T* data;
      particleIndex size;
   };
   
   template<class T>
      UnstructuredParticleData<T>::UnstructuredParticleData()
      {
        data=0;
      }
   
   template<class T>
     UnstructuredParticleData<T>::UnstructuredParticleData(particleIndex size)
     : size(size)
      {
        data = scinew T[size];
      }
      
   template<class T>
      UnstructuredParticleData<T>::~UnstructuredParticleData()
      {
        if(data)
          delete[] data;
      }

   template<class T>
     UnstructuredParticleData<T>& UnstructuredParticleData<T>::operator=(const UnstructuredParticleData<T>& copy)
     {
       for(particleIndex i=0;i<size;i++)
         data[i] = copy.data[i];
       return *this;
     }
} // End namespace Uintah
   
#endif
