
#ifndef UINTAH_HOMEBREW_NCVariable_H
#define UINTAH_HOMEBREW_NCVariable_H

#include <Uintah/Grid/Array3.h>
#include <Uintah/Grid/DataItem.h>
#include <Uintah/Exceptions/DataWarehouseException.h>
class TypeDescription;

template<class T>
class NCVariable : public Array3<T>, public DataItem {
public:
    static const TypeDescription* getTypeDescription();
    virtual ~NCVariable();

    virtual void get(DataItem&) const;
    virtual NCVariable<T>* clone() const;
    virtual void allocate(const Region*);
    NCVariable<T>& operator=(const NCVariable<T>&);
    NCVariable();
    NCVariable(const NCVariable<T>&);

private:
};

template<class T>
const TypeDescription* NCVariable<T>::getTypeDescription()
{
    //cerr << "NCVariable::getTypeDescription not done\n";
    return 0;
}

template<class T>
NCVariable<T>::~NCVariable()
{
}

template<class T>
void NCVariable<T>::get(DataItem& copy) const
{
    NCVariable<T>* ref=dynamic_cast<NCVariable<T>*>(&copy);
    if(!ref)
	throw TypeMismatchException("NCVariable<T>");
    *ref = *this;
}

template<class T>
NCVariable<T>* NCVariable<T>::clone() const
{
    return new NCVariable<T>(*this);
}

template<class T>
NCVariable<T>& NCVariable<T>::operator=(const NCVariable<T>& copy)
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
void NCVariable<T>::allocate(const Region* region)
{
    if(getWindow())
	throw DataWarehouseException("Allocating an NCvariable that is apparently already allocated!");
    resize(region->getNx()+1, region->getNy()+1, region->getNz()+1);
}

#endif
