#ifndef UINTAH_HOMEBREW_DATAITEM_H
#define UINTAH_HOMEBREW_DATAITEM_H


namespace Uintah {
namespace Grid {

class Region;

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
    virtual void allocate(const Region*) = 0;

protected:
    DataItem(const DataItem&);
    DataItem();

private:
    DataItem& operator=(const DataItem&);
};

} // end namespace Grid
} // end namespace Uintah

//
// $Log$
// Revision 1.2  2000/03/16 22:07:58  dav
// Added the beginnings of cocoon docs.  Added namespaces.  Did a few other coding standards updates too
//
//

#endif
