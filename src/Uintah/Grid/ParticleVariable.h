#ifndef UINTAH_HOMEBREW_PARTICLEVARIABLE_H
#define UINTAH_HOMEBREW_PARTICLEVARIABLE_H

#include <Uintah/Grid/ParticleVariableBase.h>
#include <Uintah/Exceptions/TypeMismatchException.h>
#include <Uintah/Grid/ParticleData.h>
#include <Uintah/Grid/ParticleSubset.h>
#include <iostream> //TEMPORARY

namespace Uintah {
namespace Grid {

class TypeDescription;

/**************************************

CLASS
   ParticleVariable
   
   Short description...

GENERAL INFORMATION

   ParticleVariable.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Particle_Variable

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

template<class T>
class ParticleVariable : public ParticleVariableBase {
public:
    ParticleVariable();
    virtual ~ParticleVariable();
    ParticleVariable(ParticleSubset* pset);

    //////////
    // Insert Documentation Here:
    static const TypeDescription* getTypeDescription();

    //////////
    // Insert Documentation Here:
    void add(particleIndex idx, const T& value);

    //////////
    // Insert Documentation Here:
    ParticleSet* getParticleSet() const {
	return d_pset->getParticleSet();
    }

    //////////
    // Insert Documentation Here:
    ParticleSubset* getParticleSubset() const {
	return d_pset;
    }

    //////////
    // Insert Documentation Here:
    virtual ParticleVariable<T>* clone() const;

    //////////
    // Insert Documentation Here:
    virtual void allocate(const Region*);

    //////////
    // Insert Documentation Here:
    T& operator[](particleIndex idx) {
	//ASSERTRANGE(idx, 0, pdata->data.size());
	return d_pdata->data[idx];
    }

    //////////
    // Insert Documentation Here:
    void resize(int newSize) {
	d_pdata->resize(newSize);
    }
private:

    //////////
    // Insert Documentation Here:
    ParticleData<T>* d_pdata;
    ParticleSubset*  d_pset;

    ParticleVariable(const ParticleVariable<T>&);
    ParticleVariable<T>& operator=(const ParticleVariable<T>&);
};

template<class T>
const TypeDescription*
ParticleVariable<T>::getTypeDescription()
{
    //cerr << "ParticleVariable::getTypeDescription not done\n";
    return 0;
}

template<class T>
ParticleVariable<T>::ParticleVariable()
    : d_pdata(0), d_pset(0)
{
}

template<class T>
ParticleVariable<T>::~ParticleVariable()
{
    if(d_pdata && d_pdata->removeReference())
	delete d_pdata;
    if(d_pset && d_pset->removeReference())
	delete d_pset;
}

template<class T>
ParticleVariable<T>::ParticleVariable(ParticleSubset* pset)
    : d_pset(pset)
{
    d_pset->addReference();
    d_pdata=new ParticleData<T>();
    d_pdata->addReference();
}

template<class T>
ParticleVariable<T>*
ParticleVariable<T>::clone() const
{
    return new ParticleVariable<T>(*this);
}

template<class T>
ParticleVariable<T>::ParticleVariable(const ParticleVariable<T>& copy)
    : d_pdata(copy.d_pdata), d_pset(copy.d_pset)
{
    if(d_pdata)
	d_pdata->addReference();
    if(d_pset)
	d_pset->addReference();
}

template<class T>
ParticleVariable<T>&
ParticleVariable<T>::operator=(const ParticleVariable<T>& copy)
{
    if(this != &copy){
	if(d_pdata && d_pdata->removeReference())
	    delete d_pdata;
	if(d_pset && d_pset->removeReference())
	    delete d_pset;
	d_pset = copy.d_pset;
	d_pdata = copy.d_pdata;
	if(d_pdata)
	    d_pdata->addReference();
	if(d_pset)
	    d_pset->addReference();
    }
    return *this;
}

template<class T>
void
ParticleVariable<T>::add(particleIndex idx, const T& value)
{
    d_pdata->add(idx, value);
}

template<class T>
void
ParticleVariable<T>::allocate(const Region*)
{
    std::cerr << "ParticleVariable::allocate not done!\n";
}

} // end namespace Grid
} // end namespace Uintah

//
// $Log$
// Revision 1.5  2000/04/20 18:56:30  sparker
// Updates to MPM
//
// Revision 1.4  2000/03/21 02:22:57  dav
// few more updates to make it compile including moving Array3 stuff out of namespace as I do not know where it should be
//
// Revision 1.3  2000/03/16 22:08:00  dav
// Added the beginnings of cocoon docs.  Added namespaces.  Did a few other coding standards updates too
//
//

#endif
