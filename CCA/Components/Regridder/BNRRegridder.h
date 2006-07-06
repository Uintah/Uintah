#ifndef UINTAH_HOMEBREW_BNRREGRIDDER_H
#define UINTAH_HOMEBREW_BNRREGRIDDER_H
#include <Packages/Uintah/CCA/Components/Regridder/RegridderCommon.h>
#include <Packages/Uintah/CCA/Components/Regridder/BNRTask.h>
#include <Packages/Uintah/CCA/Components/Regridder/PatchFixer.h>
#include <queue>
#include <list>
using namespace std;

namespace Uintah {

/**************************************

CLASS
   BNRRegridder
   
	 Coarsened Berger-Rigoutsos regridding algorithm
	 
GENERAL INFORMATION

   BNRRegridder.h

	 Justin Luitjens
   Bryan Worthen
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   BNRRegridder

DESCRIPTION
 	 Creates a patchset from refinement flags using the Berger-Rigoutsos algorithm
	 over a set of coarse refinement flags.

WARNING
  
****************************************/
  //! Takes care of AMR Regridding, using the Berger-Rigoutsos algorithm
  class BNRRegridder : public RegridderCommon {
	friend class BNRTask;
  public:
    BNRRegridder(const ProcessorGroup* pg);
    virtual ~BNRRegridder();
    void SetTolerance(float tola, float tolb) {this->tola=tola;this->tolb=tolb;};
    //! Create a new Grid
    virtual Grid* regrid(Grid* oldGrid, SchedulerP& sched, 
                         const ProblemSpecP& ups);
		
    virtual void problemSetup(const ProblemSpecP& params,
			      const GridP& grid,
			      const SimulationStateP& state);

    /***** these should be private*******/
    void RunBR(vector<IntVector> &flags, vector<PseudoPatch> &patches);
  private:
    void problemSetup_BulletProofing(const int k);
    int task_count;								//number of tasks created on this proc
    MPI_Comm comm;								//mpi communicator
    float tola,tolb;							//Tolerance parameters
    /*
      int tag_start;								//beginning of my tag range
      int tags;											//number of tags I have
    */
    
//queues for tasks
    list<BNRTask> tasks;				//list of tasks created throughout the run
    queue<BNRTask*> immediate_q;  //tasks that are always ready to run
    queue<BNRTask*> delay_q;      //tasks that may be ready to run    
    queue<BNRTask*> tag_q;				//tasks that are waiting for tags to continue
    queue<int> tags;							//available tags
    PatchFixer patchfixer;
    SizeList d_minPatchSize;
  };

} // End namespace Uintah

#endif
