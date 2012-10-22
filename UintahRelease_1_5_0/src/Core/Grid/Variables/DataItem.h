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
#ifndef UINTAH_HOMEBREW_DATAITEM_H
#define UINTAH_HOMEBREW_DATAITEM_H

namespace Uintah {

class Patch;

/**************************************

CLASS
   DataItem
   
   Short description...

GENERAL INFORMATION

   DataItem.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  

KEYWORDS
   DataItem

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

   class DataItem {
   public:
      
      virtual ~DataItem();
      virtual void get(DataItem&) const = 0;
      virtual DataItem* clone() const = 0;
      virtual void allocate(const Patch*) = 0;
      
   protected:
      DataItem(const DataItem&);
      DataItem();
      
   private:
      DataItem& operator=(const DataItem&);
   };
} // End namespace Uintah

#endif
