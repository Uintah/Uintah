
#ifndef UINTAH_HOMEBREW_CCVariable_H
#define UINTAH_HOMEBREW_CCVariable_H

#include "DataItem.h"
#include "TypeMismatchException.h"
#include <iostream> // TEMPORARY
class TypeDescription;

template<class T>
class CCVariable : public DataItem {
public:
    static const TypeDescription* getTypeDescription();
    virtual ~CCVariable();

    virtual void get(DataItem&) const;
    virtual CCVariable<T>* clone() const;
    virtual void allocate(const Region*);
    CCVariable<T>& operator=(const CCVariable<T>&);
    CCVariable();
    CCVariable(const CCVariable<T>&);
private:
};

template<class T>
const TypeDescription* CCVariable<T>::getTypeDescription()
{
    //cerr << "CCVariable::getTypeDescription not done\n";
    return 0;
}

template<class T>
CCVariable<T>::~CCVariable()
{
}

template<class T>
void CCVariable<T>::get(DataItem& copy) const
{
    CCVariable<T>* ref=dynamic_cast<CCVariable<T>*>(&copy);
    if(!ref)
	throw TypeMismatchException("CCVariable<T>");
    *ref = *this;
}

template<class T>
CCVariable<T>* CCVariable<T>::clone() const
{
    return new CCVariable<T>(*this);
}

template<class T>
CCVariable<T>& CCVariable<T>::operator=(const CCVariable<T>& copy)
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

#endif
