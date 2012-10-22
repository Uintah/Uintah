/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */

#ifndef UINTAH_HOMEBREW_GBRV2REGRIDDER_H
#define UINTAH_HOMEBREW_GBRV2REGRIDDER_H
#include <sci_defs/mpi_defs.h>
#include <queue>
#include <stack>
#include <list>
#include <set>
#include <fstream>
#include <vector>
#include <algorithm>
#include <iomanip>

#include <testprograms/Regridders/BNRTask.h>

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
  class GBRv2Regridder  {
	friend class BNRTask;
  public:
    GBRv2Regridder(double tol, IntVector rr, int rank, int numprocs);
    ~GBRv2Regridder() {};
    //! Create a new Grid
    void regrid(const std::vector<IntVector> &flags, std::vector<Region> &patches);
    void RunBR(std::vector<IntVector> &flags, std::vector<Region> &patches);

  protected:
    bool getTags(int &tag1, int &tag2);

    int task_count_;								//number of tasks created on this proc
    double tol_;							      //Tolerance parameters
   
    //tag information
    int free_tag_start_, free_tag_end_;
     
    //queues for tasks
    std::list<BNRTask> tasks_;				    //list of tasks created throughout the run
    std::queue<BNRTask*> immediate_q_;   //tasks that are always ready to run
    std::queue<BNRTask*> tag_q_;				  //tasks that are waiting for tags to continue
    std::queue<int> tags_;							  //available tags

    //request handeling variables
    std::vector<MPI_Request> requests_;    //MPI requests
    std::vector<MPI_Status>  statuses_;     //MPI statuses
    std::vector<int> indicies_;            //return value from waitsome
    std::vector<BNRTask*> request_to_task_;//maps requests to tasks using the indicies returned from waitsome
    std::queue<int>  free_requests_;       //list of free requests

    int rank, numprocs;
    IntVector rr;
  };



} // End namespace Uintah

#endif
