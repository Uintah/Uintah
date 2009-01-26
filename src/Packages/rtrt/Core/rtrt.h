/*

The MIT License

Copyright (c) 1997-2009 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/



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
      framelessMode(1),
      nworkers(1),
      worker_run_gl_test(false),
      display_run_gl_test(false),
      exit_scene(false),
      exit_engine(false),
      exit_scene_with_engine(true),
      hotSpotsMode(HotSpotsOff),
      frameMode(FullFrame)
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
    int framelessMode; // mode for frameless rendering...
    double Gjitter_vals[1000];
    double Gjitter_valsb[1000];
    unsigned int fact_table[NFACT];
    unsigned int C[NCOMB][NCOMB];
    int comb_table_inited;

    // This should only be modified by Dpy or before the rendering starts.
    int nworkers;
    int rtrt_nworkers() { return nworkers; }

    // these are for performance testing purposes
    bool worker_run_gl_test;
    bool display_run_gl_test;

    //////////////////////////////////////////////////////
    // Exit stuff
    bool exit_scene;
    bool exit_engine;
    // This will also exit the scene when the engine stops.
    bool exit_scene_with_engine;
    
    // This will stop all the scene relative windows and threads.
    // Caution should be made to make sure that stopping this won't
    // cause weird things to happen with the engine.
    void stop_scene() {
      exit_scene = true;
    }
    
    // This should cause the workers, Dpy thread, and other
    // rtrt_engine threads to exit gracefully.  If the flag
    // exit_scene_with_engine is true, then the scene stuff will also
    // be terminated.
    void stop_engine() {
      exit_engine = true;
      if (exit_scene_with_engine) stop_scene();
    }
    
    // This stops the world gracefully.
    void stop_all() {
      stop_scene();
      stop_engine();
    }

    enum {
      HotSpotsOff = 0,
      HotSpotsOn,
      HotSpotsHalfScreen
    };
    int hotSpotsMode; // defaults to HotSpotsOff

    enum {
      FullFrame = 0,
      Frameless,
      Interlaced,
      OddRows
    };
    int frameMode; // defaults to FullFrame
  };
  
} // end namespace rtrt

#endif
