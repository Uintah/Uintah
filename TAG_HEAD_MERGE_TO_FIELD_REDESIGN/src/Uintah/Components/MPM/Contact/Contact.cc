
#include <Uintah/Components/MPM/Contact/Contact.h>
#include <SCICore/Malloc/Allocator.h>

using namespace Uintah::MPM;

Contact::Contact()
{
  lb = scinew MPMLabel();
}

Contact::~Contact()
{
  delete lb;
}

//
// $Log$
// Revision 1.3  2000/08/09 03:18:01  jas
// Changed new to scinew and added deletes to some of the destructors.
//
// Revision 1.2  2000/07/05 23:43:35  jas
// Changed the way MPMLabel is used.  No longer a Singleton class.  Added
// MPMLabel* lb to various classes to retain the original calling
// convention.  Still need to actually fill the d_particleState with
// the various VarLabels that are used.
//
// Revision 1.1  2000/05/30 20:19:08  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
//


