#ifndef UINTAH_HOMEBREW_NCVARIABLE_H
#define UINTAH_HOMEBREW_NCVARIABLE_H

#include <Uintah/Grid/Array3.h>
#include <Uintah/Grid/DataItem.h>
#include <Uintah/Exceptions/DataWarehouseException.h>

namespace Uintah {
namespace Grid {

class TypeDescription;

/**************************************

CLASS
   NCVariable
   
GENERAL INFORMATION

   NCVariable.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   NCVariable

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

template<class T>
class NCVariable : public Array3<T>, public DataItem {
public:
    NCVariable();
    NCVariable(const NCVariable<T>&);
    virtual ~NCVariable();

    //////////
    // Insert Documentation Here:
    static const TypeDescription* getTypeDescription();

    //////////
    // Insert Documentation Here:
    virtual void get(DataItem&) const;

    virtual NCVariable<T>* clone() const;

    //////////
    // Insert Documentation Here:
    virtual void allocate(const Region*);

    NCVariable<T>& operator=(const NCVariable<T>&);

private:
};

template<class T>
const TypeDescription*
NCVariable<T>::getTypeDescription()
{
    //cerr << "NCVariable::getTypeDescription not done\n";
    return 0;
}

template<class T>
NCVariable<T>::~NCVariable()
{
}

template<class T>
void
NCVariable<T>::get(DataItem& copy) const
{
    NCVariable<T>* ref=dynamic_cast<NCVariable<T>*>(&copy);
    if(!ref)
	throw TypeMismatchException("NCVariable<T>");
    *ref = *this;
}

template<class T>
NCVariable<T>*
NCVariable<T>::clone() const
{
    return new NCVariable<T>(*this);
}

template<class T>
NCVariable<T>&
NCVariable<T>::operator=(const NCVariable<T>& copy)
{
    if(this != &copy){
	Array3<T>::operator=(copy);
    }
    return *this;
}

template<class T>
NCVariable<T>::NCVariable()
{
}

template<class T>
NCVariable<T>::NCVariable(const NCVariable<T>& copy)
    : Array3<T>(copy)
{
}

template<class T>
void
NCVariable<T>::allocate(const Region* region)
{
    if(getWindow())
	throw DataWarehouseException("Allocating an NCvariable that is apparently already allocated!");
    resize(region->getNx()+1, region->getNy()+1, region->getNz()+1);
}

} // end namespace Grid
} // end namespace Uintah

//
// $Log$
// Revision 1.3  2000/03/16 22:07:59  dav
// Added the beginnings of cocoon docs.  Added namespaces.  Did a few other coding standards updates too
//
//

#endif
