/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
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

#ifndef UINTAH_HOMEBREW_REFCOUNTED_H
#define UINTAH_HOMEBREW_REFCOUNTED_H

namespace Uintah {
/**************************************

CLASS
   RefCounted
   
   Short description...

GENERAL INFORMATION

   Task.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  

KEYWORDS
   Reference_Counted, RefCounted, Handle

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class RefCounted {
   public:
      RefCounted();
      virtual ~RefCounted();
      
      //////////
      // Insert Documentation Here:
      void addReference() const;
      
      //////////
      // Insert Documentation Here:
      bool removeReference() const;

      int getReferenceCount() const {
	 return d_refCount;
      }

   private:
      RefCounted& operator=(const RefCounted&);
      //////////
      // Insert Documentation Here:
      mutable int d_refCount;
      int d_lockIndex;
   };
} // End namespace Uintah

#endif
