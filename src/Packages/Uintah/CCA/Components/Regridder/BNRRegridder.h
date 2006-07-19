#ifndef UINTAH_HOMEBREW_BNRREGRIDDER_H
#define UINTAH_HOMEBREW_BNRREGRIDDER_H
#include <Packages/Uintah/CCA/Components/Regridder/RegridderCommon.h>
#include <Packages/Uintah/CCA/Components/Regridder/BNRTask.h>
#include <Packages/Uintah/CCA/Components/Regridder/PatchFixer.h>
#include <queue>
#include <list>
#include <set>
#include <fstream>
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
    void SetTolerance(float tola, float tolb) {tola_=tola;tolb_=tolb;};
    //! Create a new Grid
    virtual Grid* regrid(Grid* oldGrid, SchedulerP& sched, 
                         const ProblemSpecP& ups);
		
    virtual void problemSetup(const ProblemSpecP& params,
			      const GridP& grid,
			      const SimulationStateP& state);

    /***** these should be private (public for testing)*******/
    void RunBR(vector<IntVector> &flags, vector<PseudoPatch> &patches);
    void PostFixup(vector<PseudoPatch> &patches,IntVector min_patch_size);
  private:
    //function for outputing grid in parsable format
    void writeGrid(Grid* grid,vector<vector<IntVector> > flag_sets);
    void problemSetup_BulletProofing(const int k);
    void AddSafetyLayer(const vector<PseudoPatch> patches, set<IntVector> &coarse_flags,
                        const vector<const Patch*>& coarse_patches, int level);

    int task_count_;								//number of tasks created on this proc
    double tola_,tolb_;							//Tolerance parameters
    unsigned int target_patches_;
    
    //queues for tasks
    list<BNRTask> tasks_;				//list of tasks created throughout the run
    queue<BNRTask*> immediate_q_;  //tasks that are always ready to run
    queue<BNRTask*> tag_q_;				//tasks that are waiting for tags to continue
    queue<int> tags_;							//available tags
    PatchFixer patchfixer_;
    IntVector d_minPatchSize;

    //request handeling variables
    vector<MPI_Request> requests_;
    vector<int> indicies_;
    vector<BNRTask*> request_to_task_;
    queue<int>  free_requests_;

    ofstream fout;
  };

} // End namespace Uintah

#endif
