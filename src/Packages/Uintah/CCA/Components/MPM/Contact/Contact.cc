
#include <Packages/Uintah/CCA/Components/MPM/Contact/Contact.h>
#include <Core/Malloc/Allocator.h>

using namespace Uintah;

Contact::Contact()
{
  lb = scinew MPMLabel();
}

Contact::~Contact()
{
  delete lb;
}



