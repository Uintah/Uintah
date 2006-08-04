
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>

using namespace Uintah;

ProcessorGroup::ProcessorGroup(const ProcessorGroup* parent,
			       MPI_Comm comm, bool allmpi,
			       int rank, int size)
   : d_parent(parent), d_rank(rank), d_size(size),
     d_comm(comm), d_allmpi(allmpi)
{
}

ProcessorGroup::~ProcessorGroup()
{
}

