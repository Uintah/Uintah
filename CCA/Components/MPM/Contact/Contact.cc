#include <Packages/Uintah/CCA/Components/MPM/Contact/Contact.h>
#include <Core/Malloc/Allocator.h>

using namespace Uintah;

Contact::Contact(const ProcessorGroup* myworld, MPMLabel* Mlb, MPMFlags* MFlag, ProblemSpecP ps)
  : UintahParallelComponent(myworld), lb(Mlb), flag(MFlag), d_matls(ps)
{
}

Contact::~Contact()
{
}
