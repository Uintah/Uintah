#include <Packages/Uintah/CCA/Components/Regridder/BNRRegridder.h>
#include <Packages/Uintah/CCA/Components/Regridder/BNRTask.h>

using namespace Uintah;
#include <vector>
#include <set>
#include <algorithm>
using namespace std;

int sign(int i)
{
		if(i>0)
						return 1;
		else if (i<0)
						return -1;
		else
						return 0;
}
BNRRegridder *BNRTask::controller_=0;

BNRTask::BNRTask(PseudoPatch patch, FlagsList flags, const vector<int> &p_group, int p_rank, BNRTask *parent, unsigned int tag): status_(NEW), patch_(patch), flags_(flags), parent_(parent), sibling_(0), tag_(tag), p_group_(p_group), p_rank_(p_rank)
{
		/*
	if(controller_->task_count_/controller_->tags_>0)
		cout << "WARNING REUSING TAGS\n";
	*/

	//calculate hypercube dimensions
 	unsigned int p=1;
  d_=0;
  while(p<p_group_.size())
  {
    p<<=1;
	  d_++;
  }
}


/*
 *	This function continues a task from where it left_ off
 *	Each task runs through the BR algorithm and performs
 *	communication where needed.  When the task is unable
 *	to make progress by waiting on communication it is
 *	placed on the delay_q_.  Tasks on the delay_q_ will 
 *  be continued after their communication is finished.
 *  I use goto's in this and I know they are bad form 
 *  but in this case goto's make the algorithm easier 
 *  to understand.
 */
void BNRTask::continueTask()
{
	int stride;
	int msg_size;
	unsigned int p;
	unsigned int partner;
	MPI_Request request;
	//cout << "rank:" << p_group_[p_rank_] << ": pid:" << tag_ << ": restarting task with status_: " << status_ << endl;
	switch (status_)
	{
					case NEW: 																														//0
									goto TASK_START;
					case GATHERING_FLAG_COUNT:																						//1
									goto GATHER_FLAG_COUNT;
					case BROADCASTING_FLAG_COUNT:																					//2
									goto BROADCAST_FLAG_COUNT;
					case COMMUNICATING_SIGNATURES:																				//3
									goto COMMUNICATE_SIGNATURES;
					case SUMMING_SIGNATURES:																							//4
									goto SUM_SIGNATURES;
					case BROADCASTING_ACCEPTABILITY:																			//5
									goto BROADCAST_ACCEPTABILITY;
					case WAITING_FOR_TAGS:																								//6
									goto WAIT_FOR_TAGS;																						
					case BROADCASTING_CHILD_TASKS:																				//7
									goto BROADCAST_CHILD_TASKS;
					case WAITING_FOR_CHILDREN:																						//8
									goto WAIT_FOR_CHILDREN;																	
					case WAITING_FOR_PATCH_COUNT:																					//9
									goto WAIT_FOR_PATCH_COUNT;
					case WAITING_FOR_PATCHES:																							//10
									goto WAIT_FOR_PATCHES;
					case TERMINATED:																											//11
									//cout << "rank:" << p_group_[p_rank_] << ": " << "pid:" << tag_  << ": task done\n" ;
									
								/*	
									cout << "rank:" << p_group_[p_rank_] << ": " << "pid:" << tag_  << ": reclaiming tag_\n" ;
									controller_->tags_.push(tag_);		//reclaim tag_
								*/	
									my_patches_.clear();
									return;
					default:
									cout << "rank:" << p_group_[p_rank_] << ": " << "pid:" << tag_  << ": error invalid status_: " << status_ << endl;
									return;
	}
									
	TASK_START:
	/*
	cout << "rank:" << p_group_[p_rank_] << ": " << "pid:" << tag_  << ": start patch_: {" 
					<< patch_.low[0] << "-" << patch_.high[0] << ", "
					<< patch_.low[1] << "-" << patch_.high[1] << ", "
					<< patch_.low[2] << "-" << patch_.high[2] << "} "
					<< "flags_: " << flags_.size << " ";
	for(int f=0;f<flags_.size;f++)
	{
			cout << "{" << flags_.locs[f][0] << ", " << flags_.locs[f][1] << ", " << flags_.locs[f][2] << "},"; 
	}
	cout << endl;
	*/
					
	offset_=-patch_.low;
	//cout <<"rank:" <<  p_group_[p_rank_] <<": " << "pid:" << tag_ <<": dimensions: " << d << " processors:" << p_group_.size() << endl;
	if(p_group_.size()>1)
	{
		//gather # of flags_ on root
		status_=GATHERING_FLAG_COUNT;
		//set mpi state
		stage_=0;
	
		//Allocate recieve buffer
		flagscount_.resize(1<<d_);		//make this big enough to recieve for entire hypercube
		flagscount_[0].count=flags_.size;
		flagscount_[0].rank=p_group_[p_rank_];

		//cout << "rank:" << p_group_[p_rank_] << ": " << "pid:" << tag_  <<": gathering flags_\n";
		GATHER_FLAG_COUNT:

		if(stage_<d_)
		{
			stride=1<<(d_-1-stage_);
			msg_size=1<<stage_;

			//cout << "rank:" << p_group_[p_rank_] << ": pid:" << tag_ << ": stride:" << stride << " msg_size:" << msg_size << " stage_:" << stage_ << endl;
			stage_++;
			if(p_rank_<stride)	//recieving
			{
				partner=p_rank_+stride;
				if(partner<p_group_.size())
				{
			  //cout <<"rank:" <<  p_group_[p_rank_] << ": pid:" << tag_ << ": recieving from " << p_group_[partner] << " msg_size=" << msg_size << " tag_=" << tag_ <<  endl;
					//Nonblocking recieve msg from partner
					mpi_requests_.push(request);
					MPI_Irecv(&flagscount_[msg_size],msg_size*sizeof(FlagsCount),MPI_BYTE,p_group_[partner],tag_,controller_->d_myworld->getComm(),&mpi_requests_.back());
					//delay_q_->push(this);
					controller_->delay_q_.push(this);
					return;
				}
				else
				{
					
					for(int f=0;f<msg_size;f++)
					{
							flagscount_[msg_size+f].count=0;
							flagscount_[msg_size+f].rank=-1;
					}		
					goto GATHER_FLAG_COUNT;
				}
			}
			else if(p_rank_ < (stride<<1) )	//sending
			{
				int partner=p_rank_-stride;
			
				//non blocking send msg of size size to partner
				//cout << "rank:" << p_group_[p_rank_] << ": pid:" << tag_ << ": sending to " << p_group_[partner] << " msg_size=" << msg_size << " tag_=" << tag_ <<  endl;
				//Nonblocking recieve msg from partner
				mpi_requests_.push(request);
				MPI_Isend(&flagscount_[0],msg_size*sizeof(FlagsCount),MPI_BYTE,p_group_[partner],tag_,controller_->d_myworld->getComm(),&mpi_requests_.back());
				//delay_q_->push(this);
				controller_->delay_q_.push(this);
				return;
			}
		}
		
		status_=BROADCASTING_FLAG_COUNT;
		stage_=0;
		
		//cout << "rank:" << p_group_[p_rank_] << ": " << "pid:" << tag_  << ": broadcasting flags_\n";
		BROADCAST_FLAG_COUNT:
	
		if(Broadcast(&flagscount_[0],flagscount_.size()*sizeof(FlagsCount),MPI_BYTE,1))
			return;
		
		if(flags_.size==0)	//if i don't have any flags_ don't participate any longer
		{
			if(parent_==0)
			{
					//sort flags_ so this processor knows who will be broadcasting the results out
					sort(flagscount_.begin(),flagscount_.end());
			}
			//cout << "rank:" << p_group_[p_rank_] << ": pid:" << tag_ << ": no flags_, terminating\n";
			p_rank_=-1;
			goto TERMINATE;		
		}
/*
		cout << "rank:" << p_group_[p_rank_] << ": pid:" << tag_ << ": flagscount_: ";
		for(int f=0;f<flagscount_.size();f++)
		{
			cout << flagscount_[f].rank << ":" << flagscount_[f].count_ << ", ";
		}
		cout << endl;
*/
//		cout << "rank:" << p_group_[p_rank_] << ": " << "pid:" << tag_  << ": creating new processor group\n";
		//sort ascending #flags_
		sort(flagscount_.begin(),flagscount_.end());

		//update p_rank_
		for(p=0;p<p_group_.size();p++)
		{
			if(flagscount_[p].rank==p_group_[p_rank_])
			{
					p_rank_=p;
					break;
			}
		}
		//update p_group_
		for(p=0;p<p_group_.size();p++)
		{
			if(flagscount_[p].count==0)
							break;
			p_group_[p]=flagscount_[p].rank;	
		}
		p_group_.resize(p);
		
		//calculate hypercube dimensions
		p=1;		
		d_=0;
		while(p<p_group_.size())
		{
			p<<=1;
			d_++;
		}
	/*
		cout << "rank:" << p_group_[p_rank_] << ": " << "pid:" << tag_  << ": p_group_: ";
		for(unsigned int p=0;p<p_group_.size();p++)
						cout << p_group_[p] << " ";
		cout << endl;
		cout << "rank:" << p_group_[p_rank_] << ": " << "pid:" << tag_  << ": p_group_.size():" << p_group_.size() << " p_rank_: " << p_rank_ << " d:" << d << endl;
	*/
		//compute #of flags_ on new root 
		if(p_rank_==0)
		{
//			cout << "rank:" << p_group_[p_rank_]<< ": pid:" << tag_ << ": count_ing flags_\n";
			total_flags_=0;
			for(unsigned int p=0;p<p_group_.size();p++)
			{
				total_flags_+=flagscount_[p].count;
			}
		}
	
//		cout << "rank:" << p_group_[p_rank_] << ": " << "pid:" << tag_  << ": clearing flagscount_\n";
		//give buffer back to OS
		flagscount_.clear();	
	}
	else
	{
			total_flags_=flags_.size;
	}
	
	//cout << "rank:" << p_group_[p_rank_]<< ": pid:" << tag_ << ": computing local signature\n";
	//compute local signatures
	ComputeLocalSignature();

	if(p_group_.size()>1)
	{
		sum_[0].resize(patch_.high[0]-patch_.low[0]);
		sum_[1].resize(patch_.high[1]-patch_.low[1]);
		sum_[2].resize(patch_.high[2]-patch_.low[2]);
		//sum_ signatures
		stage_=0;
		status_=COMMUNICATING_SIGNATURES;
		//cout << "rank:" << p_group_[p_rank_] << ": " << "pid:" << tag_  << ": d_myworld->getComm()unicating signatures\n";
		COMMUNICATE_SIGNATURES:
			
		//cout << "rank:" << p_group_[p_rank_] << ": " << "pid:" << tag_  << ": stage_:" << stage_ << " d:" << d << endl;
		
		//global reduce sum_ signatures
		if(stage_<d_)
		{
			stride=1<<(d_-1-stage_);
			//cout << "rank:" << p_group_[p_rank_] << ": " << "pid:" << tag_  << ": stage_:" << stage_ << " stride:" << stride << endl;
			stage_++;
			//determine if i'm a sender or a reciever
			if(p_rank_<stride)
			{
				partner=p_rank_+stride;
				if(partner<p_group_.size())
				{
					status_=SUMMING_SIGNATURES;
						
			  	//cout << "rank:" << p_group_[p_rank_] << ": pid:" << tag_ << ": recieving from " << p_group_[partner] <<  " tag_=" << tag_ <<  endl;
					//Nonblocking recieve msg from partner
					mpi_requests_.push(request);
					MPI_Irecv(&sum_[0][0],sum_[0].size(),MPI_INT,p_group_[partner],tag_,controller_->d_myworld->getComm(),&mpi_requests_.back());
					mpi_requests_.push(request);
					MPI_Irecv(&sum_[1][0],sum_[1].size(),MPI_INT,p_group_[partner],tag_,controller_->d_myworld->getComm(),&mpi_requests_.back());
					mpi_requests_.push(request);
					MPI_Irecv(&sum_[2][0],sum_[2].size(),MPI_INT,p_group_[partner],tag_,controller_->d_myworld->getComm(),&mpi_requests_.back());
					controller_->delay_q_.push(this);
					return;

					SUM_SIGNATURES:
						
					//cout << "rank:" << p_group_[p_rank_] << ": pid:" << tag_ << ": sum_ming count_\n";			
					for(int d=0;d<3;d++)
					{
					//	cout << "rank:" << p_group_[p_rank_] << ": pid:" << tag_ << ": dim:" << d << " count_[d].size():" << count_[d].size() << " {";			
						for(unsigned int i=0;i<count_[d].size();i++)
						{
					//		cout << count_[d][i] << "->";
							count_[d][i]+=sum_[d][i];
					//		cout << count_[d][i] << " ";
						}
					//	cout << "}" << endl;
					}
						
					status_=COMMUNICATING_SIGNATURES;
					goto COMMUNICATE_SIGNATURES;
				}
				else
				{
						goto COMMUNICATE_SIGNATURES;
				//	goto SUM_SIGNATURES;
				}
			}
			else if(p_rank_< (stride<<1))
			{
					partner=p_rank_-stride;
				  //cout << "rank:" << p_group_[p_rank_] <<": pid:" << tag_ <<  ": sending to " << p_group_[partner] <<  " tag_=" << tag_ <<  endl;
					//Nonblocking recieve msg from partner
					mpi_requests_.push(request);
					
					MPI_Isend(&count_[0][0],count_[0].size(),MPI_INT,p_group_[partner],tag_,controller_->d_myworld->getComm(),&mpi_requests_.back());
					mpi_requests_.push(request);
					MPI_Isend(&count_[1][0],count_[1].size(),MPI_INT,p_group_[partner],tag_,controller_->d_myworld->getComm(),&mpi_requests_.back());
					mpi_requests_.push(request);
					MPI_Isend(&count_[2][0],count_[2].size(),MPI_INT,p_group_[partner],tag_,controller_->d_myworld->getComm(),&mpi_requests_.back());
					controller_->delay_q_.push(this);
					return;
			}
		}
		
//		cout << "rank:" << p_group_[p_rank_] <<": pid:" << tag_ <<  ": deallocating sum_ array\n";
		//deallocate sum_ array
		sum_[0].clear();	
		sum_[1].clear();	
		sum_[2].clear();	
		
	}	
	
	if(p_rank_==0)
	{
/*
		cout << "rank:" << p_group_[p_rank_] << ": Patch: " << patch_.low[0] << "-" << patch_.high[0] << "," << patch_.low[1] << "-" << patch_.high[1] << "," << patch_.low[2] << "-" << patch_.high[2] << endl;
		cout << "rank:" << p_group_[p_rank_] << ": pid:" << tag_ << ": Global sum_: \n";
		for(int d=0;d<3;d++)
		{
	 		cout << "rank:" << p_group_[p_rank_] << ": pid:" << tag_ <<  ": dim:" <<  d << ": ";
			for(unsigned int i=0;i<count_[d].size();i++)
			{
				cout << count_[d][i] << " ";
			}
			cout << endl;
		}
*/		
//		cout << "rank:" << p_group_[p_rank_] << ": " << "pid:" << tag_  << ": bounding signatures\n";
		//bound signatures
		BoundSignatures();	
/*
		cout << "rank:" << p_group_[p_rank_] << ": " << "pid:" << tag_  << ": bounded patch_: ";
		for(int dim=0;dim<3;dim++)
			cout << patch_.low[dim] << "-" << patch_.high[dim] << ", ";
		cout << endl;
*/
		//check tolerance a
		CheckTolA();
	}	
	
	if(p_group_.size()>1)
	{
		stage_=0;
		status_=BROADCASTING_ACCEPTABILITY;
		//cout << "rank:" << p_group_[p_rank_] << ": " << "pid:" << tag_  << ": broadcasting acceptablity\n";
		BROADCAST_ACCEPTABILITY:
		//broadcast acceptablity	
		if(Broadcast(&acceptable_,1,MPI_INT,1))
		{
			return;
		}
	}	

	if(acceptable_)
	{
		//cout << "rank:" << p_group_[p_rank_] << ": " << "pid:" << tag_  << ": patch_ is acceptable_\n";
		if(p_rank_==0)
		{
			my_patches_.push_back(patch_);
		}
		//goto COMMUNICATE_PATCH_LIST;
	}
	else
	{
		//cout << "rank:" << p_group_[p_rank_] << ": " << "pid:" << tag_  << ": patch_ is not acceptable_\n";
		if(p_rank_==0)
		{
			ctasks_.split=FindSplit();
			//cout << "rank:" << p_group_[p_rank_] << ": " << "pid:" << tag_  << ": split: d:" << ctasks_.split.d << ", index:" << ctasks_.split.index << endl;
			ctasks_.left=ctasks_.right=patch_;
			ctasks_.left.high[ctasks_.split.d]=ctasks_.right.low[ctasks_.split.d]=ctasks_.split.index;

			if(controller_->tags_.size()<2)
			{
				status_=WAITING_FOR_TAGS;
				//cout << "rank:" << p_group_[p_rank_] << ": " << "pid:" << tag_  << ": not enough tags waiting\n";
				controller_->tag_q_.push(this);
				return;
			}
			WAIT_FOR_TAGS:
			
			ctasks_.ltag= controller_->tags_.front();
			controller_->task_count_++;
			controller_->tags_.pop();
			
			ctasks_.rtag=controller_->tags_.front();
			controller_->task_count_++;
			controller_->tags_.pop();
			
			//cout << "rank:" << p_group_[p_rank_] << ": " << "pid:" << tag_  << ": child tags: " << ctasks_.ltag << " " << ctasks_.rtag << endl;
		}

		if(p_group_.size()>1)
		{
			status_=BROADCASTING_CHILD_TASKS;
			stage_=0;
			//cout << "rank:" << p_group_[p_rank_] << ": pid:" << tag_ << ": broadcasting child tasks_\n";
			BROADCAST_CHILD_TASKS:
			//broadcast children tasks_
			if(Broadcast(&ctasks_,sizeof(ChildTasks),MPI_BYTE,0))
			{
				return;
			}
		}
		/*
		cout << "rank:" << p_group_[p_rank_] << ": Child patch_es:\n";
		cout << "left_: tag_:" << ctasks_.ltag << endl;
		for(int d=0;d<3;d++)
			cout << ctasks_.llow[d] << "-" << ctasks_.lhigh[d] << ", ";
		cout << endl;
		cout << "right_: tag_:" << ctasks_.rtag << endl;
		for(int d=0;d<3;d++)
			cout << ctasks_.rlow[d] << "-" << ctasks_.rhigh[d] << ", ";
		cout << endl;
		*/
		
//		cout << "rank:" << p_group_[p_rank_] << ": pid:" << tag_ << ": creating child tasks_\n";
		CreateTasks();
		
		if(p_rank_==0)
		{	
			status_=WAITING_FOR_CHILDREN;	
			
			
			//cout << "rank:" << p_group_[p_rank_] << ": " << "pid:" << tag_  << ": waiting for children\n";
		
			//don't place on delay_q_ child task will do that
			return;
		
			WAIT_FOR_CHILDREN:
			
			//cout << "rank:" << p_group_[p_rank_] << ": " << "pid:" << tag_  << ": waiting for patch_ count_: left_tag_:" << left_->tag_+1 << " right_tag_:" << right_->tag_+1 << endl;
			//begin # of patch_es recv
			mpi_requests_.push(request);
			MPI_Irecv(&left_size_,1,MPI_INT,MPI_ANY_SOURCE,left_->tag_+1,controller_->d_myworld->getComm(),&mpi_requests_.back());
			mpi_requests_.push(request);
			MPI_Irecv(&right_size_,1,MPI_INT,MPI_ANY_SOURCE,right_->tag_+1,controller_->d_myworld->getComm(),&mpi_requests_.back());
			//recv's might not be done yet so place back on delay_q_
			status_=WAITING_FOR_PATCH_COUNT;	
			controller_->delay_q_.push(this);
			return;
			
			WAIT_FOR_PATCH_COUNT:
			//cout << "rank:" << p_group_[p_rank_] << ": " << "pid:" << tag_  << ": activated by child\n";
			
		//	cout << "rank:" << p_group_[p_rank_] << ": pid:" << tag_ << ": waiting for patch_es: left_size_: " << left_size << " right_size_: " << right_size_ << " left_tag_: " << left_->tag_ << " right_tag_: " << right_->tag_ << endl;
			status_=WAITING_FOR_PATCHES;

			my_patches_.resize(left_size_+right_size_);
			
			//recieve patch_sets from children on child tag_
			if(left_size_>0)
			{
				mpi_requests_.push(request);
				MPI_Irecv(&my_patches_[0],left_size_*sizeof(PseudoPatch),MPI_BYTE,MPI_ANY_SOURCE,left_->tag_,controller_->d_myworld->getComm(),&mpi_requests_.back());		
			}
			if(right_size_>0)
			{
				mpi_requests_.push(request);
				MPI_Irecv(&my_patches_[0]+left_size_,right_size_*sizeof(PseudoPatch),MPI_BYTE,MPI_ANY_SOURCE,right_->tag_,controller_->d_myworld->getComm(),&mpi_requests_.back());		
			}		
			controller_->delay_q_.push(this);
			return;
			WAIT_FOR_PATCHES:
			
			//cout << "rank:" << p_group_[p_rank_] << ": " << "pid:" << tag_  << ": reclaiming tag_:" << left_->tag_ << endl ;
			controller_->tags_.push(left_->tag_);		//reclaim tag_
			//cout << "rank:" << p_group_[p_rank_] << ": " << "pid:" << tag_  << ": reclaiming tag_:" << right_->tag_ << endl ;
			controller_->tags_.push(right_->tag_);		//reclaim tag_
			
			//check tolerance b and take better patch_set
			CheckTolB();
			if(!acceptable_)
			{
				my_patches_.resize(0);
				my_patches_.push_back(patch_);
			}
		}
	}

	
	//COMMUNICATE_PATCH_LIST:	
	if(p_rank_==0 && parent_!=0)
	{
		//send up the chain or to the root processor
		
		//cout << "rank:" << p_group_[p_rank_] << ": " << "pid:" << tag_  << ": sending patch_list to parent_: " << parent_->p_group_[0]  << " on tag_: " << tag_+1 << " my_patches_.size():" << my_patches_.size() << " parent_ id:" << parent_->tag_<< endl;
		my_size_=my_patches_.size();
	
		mpi_requests_.push(request);
		//send patch_ count_ to parent_
		MPI_Isend(&my_size_,1,MPI_INT,parent_->p_group_[0],tag_+1,controller_->d_myworld->getComm(),&mpi_requests_.back());
 		
		if(my_size_>0)
		{
			mpi_requests_.push(request);
			//send patch_ list to parent_
			MPI_Isend(&my_patches_[0],my_size_*sizeof(PseudoPatch),MPI_BYTE,parent_->p_group_[0],tag_,controller_->d_myworld->getComm(),&mpi_requests_.back());
		}
	}
	
	
	TERMINATE:
	
	status_=TERMINATED;
	//cout << "rank:" << p_group_[p_rank_] << ": " << "pid:" << tag_  << ": terminating\n";
	//deallocate waisted space
	count_[0].clear();
	count_[1].clear();
	count_[2].clear();
	
	//if parent_ is waiting activiate parent_ 
	if(parent_!=0 && parent_->p_rank_==0 && sibling_->status_==TERMINATED )
	{
		//cout << "rank:" << p_group_[p_rank_] << ": " << "pid:" << tag_  << ": activating parent_ on tag_:" << parent_->tag_ << endl;
		//cout << "rank:" << p_group_[p_rank_] << ": " << "pid:" << tag_  << ": sibling_ tag_:" << sibling_->tag_ << " sibling_ status_:" << sibling_->status_ << endl;
		
		//place parent_ on delay queue (parent_ is waiting for d_myworld->getComm()unication from children)
		controller_->delay_q_.push(parent_);
	}

	
	//cout << "rank:" << p_group_[p_rank_] << ": " << "pid:" << tag_  << ": mpi_requests_.size():" << mpi_requests_.size() << endl;
	if(!mpi_requests_.empty())
	{
						
		//cout << "rank:" << p_group_[p_rank_] << ": " << "pid:" << tag_  << ": waiting to finish sends\n";
		//must wait for final d_myworld->getComm()unication to finish
		controller_->delay_q_.push(this);
	}
	else
	{
		//cout << "rank:" << p_group_[p_rank_] << ": pid:" << tag_  << ": task done\n";
	}
	if(parent_!=0)
	{
		p_group_.clear();
	}

	return; 

}

void BNRTask::ComputeLocalSignature()
{
	//resize signature count_
	count_[0].resize(patch_.high[0]-patch_.low[0]);
	count_[1].resize(patch_.high[1]-patch_.low[1]);
	count_[2].resize(patch_.high[2]-patch_.low[2]);

	//initialize signature
	count_[0].assign(count_[0].size(),0);
	count_[1].assign(count_[1].size(),0);
	count_[2].assign(count_[2].size(),0);
	
	//count_ flags_
	for(int f=0;f<flags_.size;f++)
	{
			IntVector loc=flags_.locs[f]+offset_;
			count_[0][loc[0]]++;
			count_[1][loc[1]]++;
			count_[2][loc[2]]++;
	}
	/*
	cout << "rank:" << p_group_[p_rank_] << ": pid:" << tag_ << ": Local sum_: \n";
	for(int d=0;d<3;d++)
	{
		cout << "rank:" << p_group_[p_rank_] << ": pid:" << tag_ <<  ": dim:" <<  d << ": ";
		for(unsigned int i=0;i<count_[d].size();i++)
		{
			cout << count_[d][i] << " ";
		}
		cout << endl;
	}
	*/
}
void BNRTask::BoundSignatures()
{
		IntVector low;
		IntVector high;
		IntVector size=patch_.high-patch_.low;
		//for each dimension
		for(int d=0;d<3;d++)
		{
			int i;
			//search for first non zero
			for(i=0;i<size[d];i++)
			{
				if(count_[d][i]!=0)
					break;
			}
			low[d]=i+patch_.low[d];
			//search for last non zero
			for(i=size[d]-1;i>=0;i--)
			{
				if(count_[d][i]!=0)
							break;	
			}
			high[d]=i+1+patch_.low[d];
		}
		patch_.low=low;
		patch_.high=high;
}

void BNRTask::CheckTolA()
{
	patch_vol_=1;
	IntVector size=patch_.high-patch_.low;
	patch_vol_=size[0]*size[1]*size[2];
	acceptable_= float(total_flags_)/patch_vol_>=controller_->tola_;
}

void BNRTask::CheckTolB()
{
		//calculate patch_ volume of children
		int children_vol=0;
		for(unsigned int p=0;p<my_patches_.size();p++)
		{
				IntVector size=my_patches_[p].high-my_patches_[p].low;
				children_vol+=size[0]*size[1]*size[2];
		}
		
		//compare to patch_ volume of parent_
		if(float(children_vol)/patch_vol_>=controller_->tolb_)
		{
			acceptable_=false;
		}
		else
		{
			acceptable_=true;
		}
}
Split BNRTask::FindSplit()
{
/*
	cout << "rank:" << p_group_[p_rank_] << ": " << "pid:" << tag_  << ": patch_.low:{"
						<< patch_.low[0] << " " << patch_.low[1] << " " << patch_.low[2] << "} offset_:{"
						<< offset_[0] << " " << offset[1] << " " << offset[2] << "}\n";
*/	
	Split split;
	split.d=-1;
	IntVector size=patch_.high-patch_.low;
	//search for zero split in each dimension
	for(int d=0;d<3;d++)
	{
		int index=patch_.low[d]+offset_[d]+1;
		for(int i=1;i<size[d]-1;i++,index++)
		{
			if(count_[d][index]==0)
			{
					split.d=d;
					split.index=index-offset_[d];
					//cout << "rank:" << p_group_[p_rank_] << ": " << "pid:" << tag_  << ": zero split found\n";
					return split;
			}		
		}
	}
	//no zero split found	
	//search for second derivitive split
	IntVector mid=(patch_.low+patch_.high)/IntVector(2,2,2);
	int max_change=-1,max_dist=INT_MAX;
		
//	cout << "rank:" << p_group_[p_rank_] << ": pid:" << tag_ << ": offset_: {" << offset[0] << " " << offset[1] << " " << offset[2] << "}\n";
	for(int d=0;d<3;d++)
	{
		if(size[d]>2)
		{
//			cout << "rank:" << p_group_[p_rank_] << ": pid:" << tag_ << ": d2: dim:" << d << "{";
			int d2, last_d2;
			int s;
			
			int index=patch_.low[d]+offset_[d];
			last_d2=count_[d][index+1]-count_[d][index];
//			cout << last_d2 << " ";
			int last_s=sign(last_d2);
			index++;
			for(int i=1;i<size[d]-1;i++,index++)
			{
				d2=count_[d][index-1]+count_[d][index+1]-2*count_[d][index];
//				cout << d2 << " ";
				s=sign(d2);
				
				//if sign change
				if(last_s!=s)
				{
					int change=abs(last_d2-d2);
					int dist=abs(mid[d]-index);
					//compare to max found sign change and update max
					if(change>=max_change)
					{
						if(change>max_change)
						{
							max_change=change;
							split.d=d;
							split.index=index-offset_[d];
							max_dist=dist;
						}
						else
						{
							//tie breaker - take longest dimension
							if(size[d]>=size[split.d])
							{
								if(size[d]>size[split.d])
								{
									max_change=change;
									split.d=d;
									split.index=index-offset_[d];
									max_dist=dist;
								}
								else
								{
									//tie breaker - closest to center
									if(dist<max_dist)
									{
										max_change=change;
										split.d=d;
										split.index=index-offset_[d];
										max_dist=dist;
									}
								} 
							}	
						}
					}
				}
				
				last_d2=d2;
				last_s=s;
			}
			d2=count_[d][index-1]-count_[d][index];
			s=sign(d2);
							
//			cout << d2 << "}\n";

			//if sign change
			if(last_s!=s)
			{
				int change=abs(last_d2-d2);
				int dist=abs(mid[d]-index);
				//compare to max found sign change and update max
				if(change>=max_change)
				{
					if(change>max_change)
					{
						max_change=change;
						split.d=d;
						split.index=index-offset_[d];
						max_dist=dist;
					}
					else
					{
						//tie breaker - take longest dimension
						if(size[d]>=size[split.d])
						{
							if(size[d]>size[split.d])
							{
								max_change=change;
								split.d=d;
								split.index=index-offset_[d];
								max_dist=dist;
							}
						}	
					}
				}
			}
		}
	}
	
	if(split.d>=0)
	{
		//cout << "rank:" << p_group_[p_rank_] << ": " << "pid:" << tag_  << ": d2 split found, max_change was: " << max_change << endl;
		return split;
	}
	//no second derivitive split found 
	//take middle of longest dim
	int max_d=0;
	for(int d=1;d<3;d++)
	{
		if(size[d]>size[max_d])
						max_d=d;
	}
	split.d=max_d;
	split.index=mid[max_d];
	//cout << "rank:" << p_group_[p_rank_] << ": " << "pid:" << tag_  << ": middle split\n";
	return split;
}

//return value indicates if there is more broadcasting to perform on this processor
bool BNRTask::Broadcast(void *message, int count_, MPI_Datatype datatype,unsigned int tag_inc)
{
	unsigned int partner;
	MPI_Request request;
	//broadcast flagscount_ back to procs
//	cout << "rank:" << p_group_[p_rank_] << ": pid:" << tag_ << ": stage_: " << stage_ << " d: " << d << " p_rank_: " << p_rank_ << " p_group_.size(): " << p_group_.size() << " tag_inc:" << tag_inc <<  endl;
	if(stage_<d_)
	{
		int stride=1<<stage_;
		stage_++;
		if(p_rank_<stride)
		{
			partner=p_rank_+stride;
			if(partner<p_group_.size())
			{
				//Nonblocking send msg to partner
			//cout << "rank:" << p_group_[p_rank_]  << ": pid:" << tag_<< ": sending to " << p_group_[partner] << " on tag_: " << tag_+tag_inc << " message length:" << count_ << endl;
				mpi_requests_.push(request);
				MPI_Isend(message,count_,datatype,p_group_[partner],tag_+tag_inc,controller_->d_myworld->getComm(),&mpi_requests_.back());
				controller_->delay_q_.push(this);
				return true;
			}
		}
		else if(p_rank_< (stride<<1))
		{
			partner=p_rank_-stride;
				
			//Nonblocking recieve msg from partner
			//cout << "rank:" << p_group_[p_rank_]  << ": pid:" << tag_<< ": recieving from " << p_group_[partner] << " on tag_: " << tag_+tag_inc << " message length:" << count_ << endl;
			mpi_requests_.push(request);
			MPI_Irecv(message,count_,datatype,p_group_[partner],tag_+tag_inc,controller_->d_myworld->getComm(),&mpi_requests_.back());	
			
			controller_->delay_q_.push(this);
			return true;
		}
		else
		{
			controller_->immediate_q_.push(this);
			//cout << "rank:" << p_group_[p_rank_]  << ": pid:" << tag_<< ": not active this stage_ moving to next" << endl;
			return true;
		}
	}
		
	//cout << "rank:" << p_group_[p_rank_]  << ": pid:" << tag_<< ": broadcast done\n";
	return false;		
}

void BNRTask::CreateTasks()
{
	FlagsList leftflags_,rightflags_;
		
	//output flags_ before
	//split the flags_
	int front=0, back=flags_.size-1;	
	int d=ctasks_.split.d, v=ctasks_.split.index;
	while(front<back)
	{
		if(flags_.locs[front][d]<v) //place at front
		{
			front++;
		}
		else
		{
			swap(flags_.locs[front],flags_.locs[back]);
			back--;
		}
	}
	if(flags_.locs[front][d]<v)
		front++;
	
	leftflags_.locs=flags_.locs;
	leftflags_.size=front;

	rightflags_.locs=flags_.locs+front;
	rightflags_.size=flags_.size-front;
	/*
	cout << "rank:" << p_group_[p_rank_] << ": pid:" << tag_ << ": left_ patch_: tag_:" << ctasks_.ltag << " {" 
					<< ctasks_.left_.low[0] << "-" << ctasks_.left_.high[0] << ", "
					<< ctasks_.left_.low[1] << "-" << ctasks_.left_.high[1] << ", "
					<< ctasks_.left_.low[2] << "-" << ctasks_.left_.high[2] << "}\n";
	
	
	cout << "rank:" << p_group_[p_rank_] << ": pid:" << tag_ << ": right_ patch_: tag_:" << ctasks_.rtag << " {" 
					<< ctasks_.right_.low[0] << "-" << ctasks_.right_.high[0] << ", "
					<< ctasks_.right_.low[1] << "-" << ctasks_.right_.high[1] << ", "
					<< ctasks_.right_.low[2] << "-" << ctasks_.right_.high[2] << "}\n";
	*/
	/*	
	//output left_ flags_
	cout << "rank:" << p_group_[p_rank_] << ": left_ flags_:";
	for(int f=0;f<leftflags_.size;f++)
	{
			cout << "{" << leftflags_.locs[f][0] << "," << leftflags_.locs[f][1] << "," << leftflags_.locs[f][2] << "},";
	}
	cout << endl;
	//output right_ flags_
	cout << "rank:" << p_group_[p_rank_] << ": right_ flags_:";
	for(int f=0;f<rightflags_.size;f++)
	{
			cout << "{" << rightflags_.locs[f][0] << "," << rightflags_.locs[f][1] << "," << rightflags_.locs[f][2] << "},";
	}
	cout << endl;
	*/	
		
	//create new tasks_		
	
	controller_->tasks_.push_back(BNRTask(ctasks_.left,leftflags_,p_group_,p_rank_,this,ctasks_.ltag));
	left_=&controller_->tasks_.back();
	
	controller_->tasks_.push_back(BNRTask(ctasks_.right,rightflags_,p_group_,p_rank_,this,ctasks_.rtag));
	right_=&controller_->tasks_.back();
	
	left_->setSibling(right_);	
	right_->setSibling(left_);	

	controller_->immediate_q_.push(left_);
	controller_->immediate_q_.push(right_);
	
}

	
