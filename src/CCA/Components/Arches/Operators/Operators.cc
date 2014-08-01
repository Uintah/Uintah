#include <CCA/Components/Arches/Operators/Operators.h> 

using namespace Uintah; 

Operators& 
Operators::self()
{
  static Operators s; 
  return s; 
}

Operators::Operators()
{}

Operators::~Operators()
{} 
