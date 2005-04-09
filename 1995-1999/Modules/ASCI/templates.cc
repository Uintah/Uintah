
/*
 * Manual template instantiations for g++
 */

#include <Classlib/Array1.cc>
#include <Classlib/HashTable.cc>

class Prop;
template class HashTable<int, Prop*>;
class ParticleGroup;
template class HashTable<int, ParticleGroup*>;

class TimeStep;
template class Array1<TimeStep*>;
template class Array1<ParticleGroup*>;
