#ifndef UINTAH_HOMEBREW_SOLEVARIABLE_H
#define UINTAH_HOMEBREW_SOLEVARIABLE_H

#include <Uintah/Grid/DataItem.h>
#include <Uintah/Exceptions/TypeMismatchException.h>
#include <iostream> // TEMPORARY

namespace Uintah {
namespace Grid {
  class TypeDescription;
    using Uintah::Exceptions::TypeMismatchException;

/**************************************

CLASS
   SoleVariable
   
   Short description...

GENERAL INFORMATION

   SoleVariable.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Sole_Variable

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

template<class T>
class SoleVariable : public DataItem {
public:
    inline SoleVariable() {}
    inline SoleVariable(T value) : value(value) {}
    inline SoleVariable(const SoleVariable<T>& copy) : value(copy.value) {}
    virtual ~SoleVariable();

    static const TypeDescription* getTypeDescription();

    inline operator T () const {
	return value;
    }
    virtual void get(DataItem&) const;
    virtual SoleVariable<T>* clone() const;
    virtual void allocate(const Region*);
private:
    SoleVariable<T>& operator=(const SoleVariable<T>& copy);
    T value;
};

template<class T>
const TypeDescription*
SoleVariable<T>::getTypeDescription()
{
    //cerr << "SoleVariable::getTypeDescription not done\n";
    return 0;
}

template<class T>
void
SoleVariable<T>::get(DataItem& copy) const
{
    SoleVariable<T>* ref = dynamic_cast<SoleVariable<T>*>(&copy);
    if(!ref)
	throw TypeMismatchException("SoleVariable<T>");
    *ref = *this;
}

template<class T>
SoleVariable<T>::~SoleVariable()
{
}

template<class T>
SoleVariable<T>*
SoleVariable<T>::clone() const
{
    return new SoleVariable<T>(*this);
}

template<class T>
SoleVariable<T>&
SoleVariable<T>::operator=(const SoleVariable<T>& copy)
{
    value = copy.value;
    return *this;
}

template<class T>
void
SoleVariable<T>::allocate(const Region*)
{
    throw TypeMismatchException("SoleVariable shouldn't use allocate");
}

} // end namespace Grid
} // end namespace Uintah

//
// $Log$
// Revision 1.5  2000/04/13 06:51:02  sparker
// More implementation to get this to work
//
// Revision 1.4  2000/04/11 07:10:50  sparker
// Completing initialization and problem setup
// Finishing Exception modifications
//
// Revision 1.3  2000/03/16 22:08:01  dav
// Added the beginnings of cocoon docs.  Added namespaces.  Did a few other coding standards updates too
//
//

#endif
