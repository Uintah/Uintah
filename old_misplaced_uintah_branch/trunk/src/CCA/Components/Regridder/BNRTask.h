#ifndef UINTAH_HOMEBREW_BNRTASK_H
#define UINTAH_HOMEBREW_BNRTASK_H

#include <CCA/Components/Regridder/RegridderCommon.h>
#include <queue>
#include <Core/Grid/Region.h>

namespace Uintah {

/**************************************

CLASS
   BNRTask
   
         This is a single node of the Berger Rigoutos algorithm.
         
GENERAL INFORMATION

   BNRTask.h

   Justin Luitjens
   Bryan Worthen
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2006 SCI Institute

KEYWORDS
   BNRTask

DESCRIPTION
         This is a single node of the Berger Rigoutos algorithm.  These tasks
         are restartable and work closely with the BNRRegridder.  This class is 
         only to be used from within the BNR algorithm.
  
WARNING
  
****************************************/

  struct FlagsList
  {
    IntVector* locs;                                // flag location
    int size;                                       // number of flags
  };
  
  struct Split
  {
    int d;                                         //dimension of split
    unsigned int index;                            //index of split in patch coordinates
  };

  struct ChildTasks
  {
    Split split;                                   // location of split that created these tasks
    int ltag, rtag;                                // communication tags for patches
    Region left, right;                            // child patches
  };

  enum Task_Status { NEW, REDUCING_FLAG_INFO, UPDATING_FLAG_INFO, BROADCASTING_FLAG_INFO,
                     COMMUNICATING_SIGNATURES, SUMMING_SIGNATURES,
                     WAITING_FOR_TAGS, BROADCASTING_CHILD_TASKS, WAITING_FOR_CHILDREN,
                     WAITING_FOR_PATCH_COUNT, WAITING_FOR_PATCHES, TERMINATED };

  class BNRRegridder;

  class BNRTask
  {
    friend class BNRRegridder;

    private:
      BNRTask(Region patch,
              FlagsList flags,
              const vector<int> &p_group,
              int p_rank,
              BNRTask *parent,
              unsigned int tag);
    void continueTask();
    void continueTaskSerial();
    void setSibling(BNRTask *sibling) {sibling_=sibling;};   
    void ComputeLocalSignature();
    void BoundSignatures();
    void CheckTolA();       
    void CheckTolB();
    Split FindSplit();
    void CreateTasks();
    MPI_Request* getRequest();
                
    bool Broadcast(void *message, int count, MPI_Datatype datatype);

    // Task information
    Task_Status status_;                // Status of current task
    Region patch_;                      // patch that is being worked on
    FlagsList flags_;                   // list of flags inside this task
    vector<int> flag_info_;             // information on the flags on all processors
    BNRTask *parent_;                   // pointer to parent task
    BNRTask *sibling_;                  // pointer to sibling task
    BNRTask *left_, *right_;            // left and right child tasks
                
    unsigned int total_flags_;          // total number of flags on all processors within this patch
    bool acceptable_;                   // patch acceptablity
    IntVector offset_;                  // offset for indexing 

    // Signatures
    int sig_size_;                      // size of the signature
    vector<int>     count_;             // histogram signature
    IntVector sig_offset_;              // offset into count_ and sum_ for each dimension      
                
    // MPI Communication state
    int tag_;                           // unique message tag
    unsigned int remaining_requests_;   // remaining requests on this task
    int stage_;                         // hypercube send/recieve stage
    int d_;                             // dimension of hypercube
                
    // Communication buffers
    vector<int> flag_info_buffer_;       // buffer for reducing flag info
    vector<int> sum_;                   // buffer for calculating global histogram
    ChildTasks ctasks_;                 // structure of child tasks

    // Participating processor information
    vector<int> p_group_;               // particpating processor group
    int p_rank_;                        // rank within group
        
    // pointer to controlling algorithm
    static BNRRegridder *controller_;   // controlling algorithm;

    vector<Region> my_patches_;         // list of patches
    int my_size_;                       // number of patches on the parent
    int left_size_;                     // number of patches in left child
    int right_size_;                    // number of patches in right child
  };

} // End namespace Uintah

#endif
