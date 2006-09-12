#ifndef UINTAH_HOMEBREW_BNRTASK_H
#define UINTAH_HOMEBREW_BNRTASK_H

#include <Packages/Uintah/CCA/Components/Regridder/RegridderCommon.h>
#include <queue>

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

  struct FlagsCount 
  {
    int rank;
    int count;
  };

  inline bool operator<(FlagsCount f1, FlagsCount f2)
  {
    return f1.count>f2.count;
  }

  struct FlagsList
  {
    IntVector* locs;                                // flag location
    int size;                                       // number of flags
  };
  struct PseudoPatch
  {
    IntVector low;                                  // low point of patch
    IntVector high;                                 // high point of patch
    int volume;                                     // volume of patch
  };

  inline bool operator<(PseudoPatch p1, PseudoPatch p2)
  {
    return p2.volume>p1.volume;
  }
  inline ostream& operator<<(ostream& out, PseudoPatch p1)
  {
      out << "{" << p1.low << " " << p1.high << " (" << p1.volume << ")}";
      return out;
  }
  struct Split
  {
    int d;                                         //dimension of split
    unsigned int index;                            //index of split in patch coordinates
  };

  struct ChildTasks
  {
    Split split;                                   // location of split that created these tasks
    PseudoPatch left, right;                       // child patches
    int ltag, rtag;                                // communication tags for patches
  };

  enum Task_Status { NEW, GATHERING_FLAG_COUNT, BROADCASTING_FLAG_COUNT,
                     COMMUNICATING_SIGNATURES, SUMMING_SIGNATURES, BROADCASTING_ACCEPTABILITY,
                     WAITING_FOR_TAGS, BROADCASTING_CHILD_TASKS, WAITING_FOR_CHILDREN,
                     WAITING_FOR_PATCH_COUNT, WAITING_FOR_PATCHES, TERMINATED };

  class BNRRegridder;

  class BNRTask
  {
    friend class BNRRegridder;

    private:
      BNRTask(PseudoPatch patch,
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
                
    bool Broadcast(void *message, int count, MPI_Datatype datatype,unsigned int tag);

    // Task information
    Task_Status status_;                // Status of current task
    PseudoPatch patch_;                 // patch that is being worked on
    FlagsList flags_;                   // list of flags inside this task
    BNRTask *parent_;                   // pointer to parent task
    BNRTask *sibling_;                  // pointer to sibling task
    BNRTask *left_, *right_;            // left and right child tasks
                
    unsigned int total_flags_;          // total number of flags on all processors within this patch
    bool acceptable_;                   // patch acceptablity
    IntVector offset_;                  // offset for indexing 

    // Signatures
    vector<int>     count_[3];          // histogram signature
                
    // MPI Communication state
    unsigned int tag_;                  // unique message tag
    unsigned int remaining_requests_;   // remaining requests on this task
    int stage_;                         // hypercube send/recieve stage
    int d_;                             // dimension of hypercube
                
    // Communication buffers
    vector<FlagsCount> flagscount_;     // buffer for gathering the number of flags
    vector<int> sum_[3];                // buffer for calculating global histogram
    ChildTasks ctasks_;                 // structure of child tasks

    // Participating processor information
    vector<int> p_group_;               // particpating processor group
    int p_rank_;                        // rank within group
        
    // pointer to controlling algorithm
    static BNRRegridder *controller_;   // controlling algorithm;

    vector<PseudoPatch> my_patches_;    // list of patches
    unsigned int my_size_;              // number of patches on the parent
    unsigned int left_size_;            // number of patches in left child
    unsigned int right_size_;           // number of patches in right child
  };

} // End namespace Uintah

#endif
