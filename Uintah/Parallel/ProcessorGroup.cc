
#include <Uintah/Parallel/ProcessorGroup.h>

using namespace Uintah;

ProcessorGroup::ProcessorGroup(const ProcessorGroup* parent,
			       MPI_Comm comm, bool allmpi,
			       int rank, int size)
   : d_parent(parent), d_comm(comm), d_allmpi(allmpi),
     d_rank(rank), d_size(size)
{
}

//
// $Log$
// Revision 1.2  2000/07/27 22:39:54  sparker
// Implemented MPIScheduler
// Added associated support
//
// Revision 1.1  2000/06/17 07:06:49  sparker
// Changed ProcessorContext to ProcessorGroup
//
//

