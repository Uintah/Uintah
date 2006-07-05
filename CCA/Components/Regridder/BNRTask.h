#ifndef UINTAH_HOMEBREW_BNRTASK_H
#define UINTAH_HOMEBREW_BNRTASK_H

#include <Packages/Uintah/CCA/Components/Regridder/RegridderCommon.h>

using namespace Uintah;

#include <queue>
using namespace std;

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
  
   Copyright (C) 2000 SCI Group

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
	/*
	if(f1.count==0)  //zeros need to be at the end of the list
		return false;
	if(f2.count==0)
		return true;
	if(f1.count<f2.count)
	  return true;
	*/
	/*
	if(f2.count>f1.count)
		return false;
	if(f1.rank<r2.rank)
	  return true;
	else
	*/
		return false;
}

struct FlagsList
{
	IntVector* locs;				//flag location
	int size;								//number of flags
};
struct PseudoPatch
{
	IntVector low;					//low point of patch
	IntVector high;					//high point of patch
};
struct Split
{
	int d;									//dimension of split
	unsigned int index;			//index of split in patch coordinates
};

struct ChildTasks
{
	Split split;												//location of split that created these tasks
	PseudoPatch left, right;						//child patches
	unsigned int ltag, rtag;						//communication tags for patches
};

enum Task_Status {NEW,GATHERING_FLAG_COUNT,BROADCASTING_FLAG_COUNT,COMMUNICATING_SIGNATURES, SUMMING_SIGNATURES,BROADCASTING_ACCEPTABILITY,WAITING_FOR_TAGS,BROADCASTING_CHILD_TASKS,WAITING_FOR_CHILDREN,WAITING_FOR_PATCH_COUNT,WAITING_FOR_PATCHES,TERMINATED};

class BNRTask
{
	friend class BNRRegridder;
	public:
	private:
		BNRTask()
		{
			cout << "Error empty BNRTask constructor\n";
			exit(0);
		}
		BNRTask(PseudoPatch patch, FlagsList flags, const vector<int> &p_group, int p_rank, BNRTask *parent, unsigned int tag);/*: status(NEW), patch(patch), flags(flags), parent(parent), sibling(0), tag(tag), p_group(p_group), p_rank(p_rank) 
		{
			if(controller->task_count/controller->tags>0)
							cout << "WARNING REUSING TAGS\n";
			//calculate hypercube dimensions
			unsigned int p=1;
			d=0;
			while(p<p_group.size())
			{
			 	p<<=1;
				d++;
			}
		};
		*/
		void continueTask();
	  void setSibling(BNRTask *sibling) {this->sibling=sibling;};	
		void ComputeLocalSignature();
		void BoundSignatures();
		void CheckTolA();	
		void CheckTolB();
		Split FindSplit();
		void CreateTasks();
		
		bool Broadcast(void *message, int count, MPI_Datatype datatype,unsigned int tag);

		/*Task information*/
		Task_Status status;										//Status of current task
		PseudoPatch patch;										//patch that is being worked on
		FlagsList flags;											//list of flags inside this task
		BNRTask *parent;											//pointer to parent task
		BNRTask *sibling;											//pointer to sibling task
		BNRTask *left, *right;
		
		unsigned int total_flags;							//total number of flags on all processors within this patch
		unsigned int patch_vol;								//volume of the patch 
		bool acceptable;											//patch acceptablity
		IntVector offset;

		/*Signatures*/
		vector<int>	count[3];
		
		/*MPI Communication state*/
		unsigned int tag;											//unique message tag
		queue<MPI_Request> mpi_requests;			//requests that must be finished before task can continue
		int stage;														//hypercube send/recieve stage
		int d;																//dimension of hypercube
		
		/*Communication buffers*/
		vector<FlagsCount> flagscount;	
		vector<int> sum[3];
		ChildTasks ctasks;		

		/*Participating processor information*/
		vector<int> p_group;									//particpating processor group
		int p_rank;														//rank within group
	
		/*pointer to controlling algorithm*/
		static BNRRegridder *controller;					//controlling algorithm;

		vector<PseudoPatch> my_patches;						//list of patches
		unsigned int my_size;											//number of patches on the parent
		unsigned int left_size;										//number of patches in left child
		unsigned int right_size;									//number of patches in right child
};

	

} // End namespace Uintah

#endif
