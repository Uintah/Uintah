
#ifndef UINTAH_HOMEBREW_PARALLELCONTEXT_H
#define UINTAH_HOMEBREW_PARALLELCONTEXT_H

class ParallelContext {
public:
    bool usingMPI();
    bool usingThreads();
    int totalThreads();
    int numMPIProcessors();
    MPI_Comm getMPICommunicator();
private:
    ParallelContext();
    ParallelContext(const ParallelContext&);
    ~ParallelContext();
    ParallelContext& operator=(const ParallelContext&);
};

#endif
