#ifndef UINTAH_HOMEBREW_CCVARIABLE_H
#define UINTAH_HOMEBREW_CCVARIABLE_H

#include <Uintah/Grid/DataItem.h>
#include <Uintah/Exceptions/TypeMismatchException.h>
#include <iostream> // TEMPORARY

namespace Uintah {
namespace Grid {

class TypeDescription;

/**************************************

CLASS
   CCVariable
   
   Short description...

GENERAL INFORMATION

   CCVariable.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Variable__Cell_Centered

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

template<class T>
class CCVariable : public DataItem {
public:
    CCVariable();
    CCVariable(const CCVariable<T>&);
    virtual ~CCVariable();

    //////////
    // Insert Documentation Here:
    static const TypeDescription* getTypeDescription();

    //////////
    // Insert Documentation Here:
    virtual void get(DataItem&) const;

    //////////
    // Insert Documentation Here:
    virtual CCVariable<T>* clone() const;

    //////////
    // Insert Documentation Here:
    virtual void allocate(const Region*);

    CCVariable<T>& operator=(const CCVariable<T>&);
private:
};

template<class T>
const TypeDescription*
CCVariable<T>::getTypeDescription()
{
    //cerr << "CCVariable::getTypeDescription not done\n";
    return 0;
}

template<class T>
CCVariable<T>::~CCVariable()
{
}

template<class T>
void
CCVariable<T>::get(DataItem& copy) const
{
    CCVariable<T>* ref=dynamic_cast<CCVariable<T>*>(&copy);
    if(!ref)
	throw TypeMismatchException("CCVariable<T>");
    *ref = *this;
}

template<class T>
CCVariable<T>*
CCVariable<T>::clone() const
{
    return new CCVariable<T>(*this);
}

template<class T>
CCVariable<T>&
CCVariable<T>::operator=(const CCVariable<T>& copy)
{
    if(this != &copy){
	std::cerr << "CCVariable<T>::operator= not done!\n";
    }
    return *this;
}

template<class T>
CCVariable<T>::CCVariable()
{
    std::cerr << "CCVariable ctor not done!\n";
}

template<class T>
CCVariable<T>::CCVariable(const CCVariable<T>& copy)
{
    std::cerr << "CCVariable copy ctor not done!\n";
}

template<class T>
void CCVariable<T>::allocate(const Region*)
{
    std::cerr << "CCVariable::allocate not done!\n";
}

} // end namespace Grid
} // end namespace Uintah

//
// $Log$
// Revision 1.3  2000/03/16 22:07:58  dav
// Added the beginnings of cocoon docs.  Added namespaces.  Did a few other coding standards updates too
//
//

#endif

