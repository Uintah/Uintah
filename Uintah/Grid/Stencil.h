#ifndef UINTAH_HOMEBREW_STENCIL_H
#define UINTAH_HOMEBREW_STENCIL_H

#include <Uintah/Grid/DataItem.h>
#include <Uintah/Exceptions/TypeMismatchException.h>
#include <iostream> // TEMPORARY

namespace Uintah {
namespace Grid {

class TypeDescription;

/**************************************

CLASS
   Stencil
   
   Short description...

GENERAL INFORMATION

   Stencil.h

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
class Stencil : public DataItem {
public:
    Stencil();
    Stencil(const Region*);
    Stencil(const Stencil<T>&);
    virtual ~Stencil();

    //////////
    // Insert Documentation Here:
    static const TypeDescription* getTypeDescription();

    //////////
    // Insert Documentation Here:
    virtual void get(DataItem&) const;

    //////////
    // Insert Documentation Here:
    virtual Stencil<T>* clone() const;

    //////////
    // Insert Documentation Here:
    virtual void allocate(const Region*);

    Stencil<T>& operator=(const Stencil<T>&);
private:
};

template<class T>
const TypeDescription*
Stencil<T>::getTypeDescription()
{
    //cerr << "Stencil::getTypeDescription not done\n";
    return 0;
}

template<class T>
Stencil<T>::~Stencil()
{
}

template<class T>
void
Stencil<T>::get(DataItem& copy) const
{
    Stencil<T>* ref=dynamic_cast<Stencil<T>*>(&copy);
    if(!ref)
	throw TypeMismatchException("Stencil<T>");
    *ref = *this;
}

template<class T>
Stencil<T>*
Stencil<T>::clone() const
{
    return new Stencil<T>(*this);
}

template<class T>
Stencil<T>&
Stencil<T>::operator=(const Stencil<T>& copy)
{
    if(this != &copy){
	std::cerr << "Stencil<T>::operator= not done!\n";
    }
    return *this;
}

template<class T>
Stencil<T>::Stencil()
{
    std::cerr << "Stencil ctor not done!\n";
}

template<class T>
Stencil<T>::Stencil(const Stencil<T>& copy)
{
    std::cerr << "Stencil copy ctor not done!\n";
}

template<class T>
void Stencil<T>::allocate(const Region*)
{
    std::cerr << "Stencil::allocate not done!\n";
}

} // end namespace Grid
} // end namespace Uintah

//
// $Log$
// Revision 1.1  2000/03/22 23:41:27  sparker
// Working towards getting arches to compile/run
//
//

#endif

