/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef UINTAH_HOMEBREW_BNRREGRIDDER_H
#define UINTAH_HOMEBREW_BNRREGRIDDER_H
#include <CCA/Components/Regridder/RegridderCommon.h>
#include <CCA/Components/Regridder/BNRTask.h>
#include <CCA/Components/Regridder/PatchFixer.h>
#include <queue>
#include <list>
#include <set>

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
    void RunBR(std::vector<IntVector> &flags, std::vector<Region> &patches);
    void PostFixup(std::vector<Region> &patches);
    std::vector<IntVector> getMinPatchSize() {return d_minPatchSize;}

    //! create and compare a checksum for the grid across all processors
    bool verifyGrid(Grid *grid);
  protected:
    void problemSetup_BulletProofing(const int k);
    void AddSafetyLayer(const std::vector<Region> patches, std::set<IntVector> &coarse_flags,
                        const std::vector<const Patch*>& coarse_patches, int level);
    void CreateCoarseFlagSets(Grid *oldGrid, std::vector<std::set<IntVector> > &coarse_flag_sets);
    Grid* CreateGrid(Grid* oldGrid, std::vector< std::vector<Region> > &patch_sets );
    
    bool getTags(int &tag1, int &tag2);
    void OutputGridStats(std::vector< std::vector<Region> > &patch_sets, Grid* newGrid);

    bool d_loadBalance;             //should the regridder call the load balancer before creating the grid

    int task_count_;								//number of tasks created on this proc
    double tola_,tolb_;							//Tolerance parameters
    double d_patchRatioToTarget;    //percentage of target volume used to subdivide patches
    unsigned int target_patches_;   //Minimum number of patches the algorithm attempts to reach
   
    //tag information
    int free_tag_start_, free_tag_end_;
     
    //queues for tasks
    std::list<BNRTask> tasks_;				    //list of tasks created throughout the run
    std::queue<BNRTask*> immediate_q_;   //tasks that are always ready to run
    std::queue<BNRTask*> tag_q_;				  //tasks that are waiting for tags to continue
    std::queue<int> tags_;							  //available tags
    PatchFixer patchfixer_;         //Fixup class
    SizeList d_minPatchSize;       //minimum patch size in each dimension

    //request handeling variables
    std::vector<MPI_Request> requests_;    //MPI requests
    std::vector<MPI_Status>  statuses_;     //MPI statuses
    std::vector<int> indicies_;            //return value from waitsome
    std::vector<BNRTask*> request_to_task_;//maps requests to tasks using the indicies returned from waitsome
    std::queue<int>  free_requests_;       //list of free requests
  };

} // End namespace Uintah

#endif
