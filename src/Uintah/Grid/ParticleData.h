
#ifndef UINTAH_HOMEBREW_ParticleData_H
#define UINTAH_HOMEBREW_ParticleData_H

#include <Uintah/Grid/RefCounted.h>
#include <Uintah/Exceptions/ParticleException.h>
template<class T>
class ParticleVariable;
#include <vector>

template<class T>
class ParticleData : public RefCounted {
public:
    ParticleData();
    ParticleData(const ParticleData<T>&);
    ParticleData<T>& operator=(const ParticleData<T>&);
    virtual ~ParticleData();

    void add(ParticleSet::index idx, const T& value);
    void resize(int newSize) {
	data.resize(newSize);
    }
private:
    friend class ParticleVariable<T>;
    std::vector<T> data;
};

template<class T>
ParticleData<T>::ParticleData()
{
}

template<class T>
void ParticleData<T>::add(ParticleSet::index idx, const T& value)
{
    if(idx != data.size())
	throw ParticleException("add, not at the end");
    data.push_back(value);
}

template<class T>
ParticleData<T>::~ParticleData()
{
}

#endif
