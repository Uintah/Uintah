#ifndef UINTAH_HOMEBREW_BNRREGRIDDER_H
#define UINTAH_HOMEBREW_BNRREGRIDDER_H
#include <Packages/Uintah/CCA/Components/Regridder/RegridderCommon.h>
#include <Packages/Uintah/CCA/Components/Regridder/BNRTask.h>
#include <Packages/Uintah/CCA/Components/Regridder/PatchFixer.h>
#include <queue>
#include <stack>
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
    virtual Grid* regrid(Grid* oldGrid);
		
    virtual void problemSetup(const ProblemSpecP& params,
			      const GridP& grid,
			      const SimulationStateP& state);

    /***** these should be private (public for testing)*******/
    void RunBR(vector<IntVector> &flags, vector<Region> &patches);
    void PostFixup(vector<Region> &patches);
  protected:
    void problemSetup_BulletProofing(const int k);
    void AddSafetyLayer(const vector<Region> patches, set<IntVector> &coarse_flags,
                        const vector<const Patch*>& coarse_patches, int level);
    void CreateCoarseFlagSets(Grid *oldGrid, vector<set<IntVector> > &coarse_flag_sets);
    Grid* CreateGrid(Grid* oldGrid, vector< vector<Region> > &patch_sets );
    
    bool getTags(int &tag1, int &tag2);
    void OutputGridStats(vector< vector<Region> > &patch_sets, Grid* newGrid);

    bool d_loadBalance;             //should the regridder call the load balancer before creating the grid

    int task_count_;								//number of tasks created on this proc
    double tola_,tolb_;							//Tolerance parameters
    double d_patchRatioToTarget;    //percentage of target volume used to subdivide patches
    unsigned int target_patches_;   //Minimum number of patches the algorithm attempts to reach
   
    //tag information
    int free_tag_start_, free_tag_end_;
     
    //queues for tasks
    list<BNRTask> tasks_;				    //list of tasks created throughout the run
    queue<BNRTask*> immediate_q_;   //tasks that are always ready to run
    queue<BNRTask*> tag_q_;				  //tasks that are waiting for tags to continue
    queue<int> tags_;							  //available tags
    PatchFixer patchfixer_;         //Fixup class
    SizeList d_minPatchSize;       //minimum patch size in each dimension

    //request handeling variables
    vector<MPI_Request> requests_;    //MPI requests
    vector<MPI_Status>  statuses_;     //MPI statuses
    vector<int> indicies_;            //return value from waitsome
    vector<BNRTask*> request_to_task_;//maps requests to tasks using the indicies returned from waitsome
    queue<int>  free_requests_;       //list of free requests
  };

} // End namespace Uintah

#endif
