#ifndef UINTAH_HOMEBREW_ReductionVARIABLE_H
#define UINTAH_HOMEBREW_ReductionVARIABLE_H

#include <Uintah/Grid/ReductionVariableBase.h>
#include <Uintah/Grid/DataItem.h>
#include <Uintah/Exceptions/TypeMismatchException.h>
#include <iostream> // TEMPORARY

namespace Uintah {
namespace Grid {
  class TypeDescription;
    using Uintah::Exceptions::TypeMismatchException;

/**************************************

CLASS
   ReductionVariable
   
   Short description...

GENERAL INFORMATION

   ReductionVariable.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Reduction_Variable

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

template<class T>
class ReductionVariable : public ReductionVariableBase {
public:
    inline ReductionVariable() {}
    inline ReductionVariable(T value) : value(value) {}
    inline ReductionVariable(const ReductionVariable<T>& copy) : value(copy.value) {}
    virtual ~ReductionVariable();

    static const TypeDescription* getTypeDescription();

    inline operator T () const {
	return value;
    }
    virtual void get(DataItem&) const;
    virtual ReductionVariable<T>* clone() const;
    virtual void allocate(const Region*);
private:
    ReductionVariable<T>& operator=(const ReductionVariable<T>& copy);
    T value;
};

template<class T>
const TypeDescription*
ReductionVariable<T>::getTypeDescription()
{
    //cerr << "ReductionVariable::getTypeDescription not done\n";
    return 0;
}

template<class T>
void
ReductionVariable<T>::get(DataItem& copy) const
{
    ReductionVariable<T>* ref = dynamic_cast<ReductionVariable<T>*>(&copy);
    if(!ref)
	throw TypeMismatchException("ReductionVariable<T>");
    *ref = *this;
}

template<class T>
ReductionVariable<T>::~ReductionVariable()
{
}

template<class T>
ReductionVariable<T>*
ReductionVariable<T>::clone() const
{
    return new ReductionVariable<T>(*this);
}

template<class T>
ReductionVariable<T>&
ReductionVariable<T>::operator=(const ReductionVariable<T>& copy)
{
    value = copy.value;
    return *this;
}

template<class T>
void
ReductionVariable<T>::allocate(const Region*)
{
    throw TypeMismatchException("ReductionVariable shouldn't use allocate");
}

} // end namespace Grid
} // end namespace Uintah

//
// $Log$
// Revision 1.1  2000/04/19 05:26:14  sparker
// Implemented new problemSetup/initialization phases
// Simplified DataWarehouse interface (not finished yet)
// Made MPM get through problemSetup, but still not finished
//
//

#endif
