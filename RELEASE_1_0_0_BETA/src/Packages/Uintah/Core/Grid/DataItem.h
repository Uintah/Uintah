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
  
   Copyright (C) 2000 SCI Group

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
