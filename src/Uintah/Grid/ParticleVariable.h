
#ifndef UINTAH_HOMEBREW_ParticleVariable_H
#define UINTAH_HOMEBREW_ParticleVariable_H

#include <Uintah/Grid/DataItem.h>
#include <Uintah/Exceptions/TypeMismatchException.h>
#include <Uintah/Grid/ParticleData.h>
#include <Uintah/Grid/ParticleSubset.h>
class TypeDescription;
#include <iostream> //TEMPORARY

template<class T>
class ParticleVariable : public DataItem {
    ParticleData<T>* pdata;
    ParticleSubset* pset;
public:
    ParticleVariable();
    virtual ~ParticleVariable();
    ParticleVariable(ParticleSubset* pset);
    static const TypeDescription* getTypeDescription();

    void add(ParticleSet::index idx, const T& value);
    ParticleSet* getParticleSet() const {
	return pset->getParticleSet();
    }

    ParticleSubset* getParticleSubset() const {
	return pset;
    }

    virtual void get(DataItem&) const;
    virtual ParticleVariable<T>* clone() const;
    virtual void allocate(const Region*);
    T& operator[](ParticleSet::index idx) {
	//ASSERTRANGE(idx, 0, pdata->data.size());
	return pdata->data[idx];
    }
    void resize(int newSize) {
	pdata->resize(newSize);
    }
private:
    ParticleVariable(const ParticleVariable<T>&);
    ParticleVariable<T>& operator=(const ParticleVariable<T>&);
};

template<class T>
const TypeDescription* ParticleVariable<T>::getTypeDescription()
{
    //cerr << "ParticleVariable::getTypeDescription not done\n";
    return 0;
}

template<class T>
ParticleVariable<T>::ParticleVariable()
    : pdata(0), pset(0)
{
}

template<class T>
ParticleVariable<T>::~ParticleVariable()
{
    if(pdata && pdata->removeReference())
	delete pdata;
    if(pset && pset->removeReference())
	delete pset;
}

template<class T>
ParticleVariable<T>::ParticleVariable(ParticleSubset* pset)
    : pset(pset)
{
    pset->addReference();
    pdata=new ParticleData<T>();
    pdata->addReference();
}

template<class T>
void ParticleVariable<T>::get(DataItem& copy) const
{
    ParticleVariable<T>* ref = dynamic_cast<ParticleVariable<T>*>(&copy);
    if(!ref)
	throw TypeMismatchException("ParticleVariable<T>");
    *ref = *this;
}

template<class T>
ParticleVariable<T>* ParticleVariable<T>::clone() const
{
    return new ParticleVariable<T>(*this);
}

template<class T>
ParticleVariable<T>::ParticleVariable(const ParticleVariable<T>& copy)
    : pdata(copy.pdata), pset(copy.pset)
{
    if(pdata)
	pdata->addReference();
    if(pset)
	pset->addReference();
}

template<class T>
ParticleVariable<T>& ParticleVariable<T>::operator=(const ParticleVariable<T>& copy)
{
    if(this != &copy){
	if(pdata && pdata->removeReference())
	    delete pdata;
	if(pset && pset->removeReference())
	    delete pset;
	pset = copy.pset;
	pdata = copy.pdata;
	if(pdata)
	    pdata->addReference();
	if(pset)
	    pset->addReference();
    }
    return *this;
}

template<class T>
void ParticleVariable<T>::add(ParticleSet::index idx, const T& value)
{
    pdata->add(idx, value);
}

template<class T>
void ParticleVariable<T>::allocate(const Region*)
{
    std::cerr << "ParticleVariable::allocate not done!\n";
}

#endif
