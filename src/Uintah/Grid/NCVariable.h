#ifndef UINTAH_HOMEBREW_NCVARIABLE_H
#define UINTAH_HOMEBREW_NCVARIABLE_H

#include <Uintah/Grid/Array3.h>
#include <Uintah/Grid/NCVariableBase.h>
#include <SCICore/Exceptions/InternalError.h>

namespace Uintah {
namespace Grid {

    using SCICore::Exceptions::InternalError;

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
class NCVariable : public Array3<T>, public NCVariableBase{
public:
    NCVariable();
    NCVariable(const NCVariable<T>&);
    virtual ~NCVariable();

    //////////
    // Insert Documentation Here:
    static const TypeDescription* getTypeDescription();

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
	throw InternalError("Allocating an NCvariable that is apparently already allocated!");
#if 0
    resize(region->getNx()+1, region->getNy()+1, region->getNz()+1);
#endif
}

} // end namespace Grid
} // end namespace Uintah

//
// $Log$
// Revision 1.7  2000/04/20 18:56:30  sparker
// Updates to MPM
//
// Revision 1.6  2000/04/12 23:00:48  sparker
// Starting problem setup code
// Other compilation fixes
//
// Revision 1.5  2000/04/11 07:10:50  sparker
// Completing initialization and problem setup
// Finishing Exception modifications
//
// Revision 1.4  2000/03/21 02:22:57  dav
// few more updates to make it compile including moving Array3 stuff out of namespace as I do not know where it should be
//
// Revision 1.3  2000/03/16 22:07:59  dav
// Added the beginnings of cocoon docs.  Added namespaces.  Did a few other coding standards updates too
//
//

#endif
