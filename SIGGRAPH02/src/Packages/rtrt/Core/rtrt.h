
#ifndef RTRT_H
#define RTRT_H 1

#include <Core/Thread/Thread.h>
#include <Core/Thread/Mutex.h>
#include <Packages/rtrt/Core/BallAux.h>
#include <Packages/rtrt/Core/Util.h>

namespace rtrt {

  class RTRT {
  public:
    RTRT():
      cameralock("Frameless Camera Synch lock"),
      do_jitter(0),
      NUMCHUNKS(1<<16),
      updatePercent(0.5),
      clusterSize(1),
      shuffleClusters(1),
      np(4),
      framelessMode(1),
      nworkers(1),
      worker_run_gl_test(false),
      display_run_gl_test(false),
      exit_everybody(false)
    {}
    // semephores
    SCIRun::Mutex cameralock; // to synchronize camera...
    
    // other
    int do_jitter;
    float Galpha;
    int NUMCHUNKS;
    double updatePercent;
    int clusterSize;
    int shuffleClusters;
    int np;
    int framelessMode; // mode for frameless rendering...
    double Gjitter_vals[1000];
    double Gjitter_valsb[1000];
    unsigned int fact_table[NFACT];
    unsigned int C[NCOMB][NCOMB];
    int comb_table_inited;

    int nworkers;
    int rtrt_nworkers() { return nworkers; }

    // these are for performance testing purposes
    bool worker_run_gl_test;
    bool display_run_gl_test;

    // exit stuff
    bool exit_everybody;
    int exit_status;
    void exit_clean(int status = 0) {
      exit_everybody = true;
      exit_status = status;
    }
    bool stop_execution() { return exit_everybody; }
       
  };
  
} // end namespace rtrt

#endif
