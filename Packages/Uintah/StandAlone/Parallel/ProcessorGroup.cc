
#include <Uintah/Parallel/ProcessorGroup.h>

using namespace Uintah;

ProcessorGroup::ProcessorGroup(const ProcessorGroup* parent,
			       int rank, int size)
   : d_parent(parent), d_rank(rank), d_size(size)
{
}

//
// $Log$
// Revision 1.1  2000/06/17 07:06:49  sparker
// Changed ProcessorContext to ProcessorGroup
//
//

