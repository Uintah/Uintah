#include <Packages/Uintah/CCA/Components/Schedulers/DetailedTasks.h>
#include <sci_config.h> // For MPIPP_H on SGI
#include <mpi.h>

namespace Uintah {

class BatchReceiveHandler {
public:
  BatchReceiveHandler(DependencyBatch* batch)
    : batch_(batch) {}
  BatchReceiveHandler(const BatchReceiveHandler& copy)
    : batch_(copy.batch_) {}
  
  void finishedCommunication(MPI_Comm)
  { batch_->received(); }
private:
  DependencyBatch* batch_;
  
};
}
