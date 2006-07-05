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
BNRRegridder *BNRTask::controller=0;

BNRTask::BNRTask(PseudoPatch patch, FlagsList flags, const vector<int> &p_group, int p_rank, BNRTask *parent, unsigned int tag): status(NEW), patch(patch), flags(flags), parent(parent), sibling(0), tag(tag), p_group(p_group), p_rank(p_rank)
{
		/*
	if(controller->task_count/controller->tags>0)
		cout << "WARNING REUSING TAGS\n";
	*/

	//calculate hypercube dimensions
 	unsigned int p=1;
  d=0;
  while(p<p_group.size())
  {
    p<<=1;
	  d++;
  }
}


/*
 *	This function continues a task from where it left off
 *	Each task runs through the BR algorithm and performs
 *	communication where needed.  When the task is unable
 *	to make progress by waiting on communication it is
 *	placed on the delay_q.  Tasks on the delay_q will 
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
	//cout << "rank:" << p_group[p_rank] << ": pid:" << tag << ": restarting task with status: " << status << endl;
	switch (status)
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
									//cout << "rank:" << p_group[p_rank] << ": " << "pid:" << tag  << ": task done\n" ;
									
								/*	
									cout << "rank:" << p_group[p_rank] << ": " << "pid:" << tag  << ": reclaiming tag\n" ;
									controller->tags.push(tag);		//reclaim tag
								*/	
									my_patches.clear();
									return;
					default:
									cout << "rank:" << p_group[p_rank] << ": " << "pid:" << tag  << ": error invalid status: " << status << endl;
									return;
	}
									
	TASK_START:
	/*
	cout << "rank:" << p_group[p_rank] << ": " << "pid:" << tag  << ": start patch: {" 
					<< patch.low[0] << "-" << patch.high[0] << ", "
					<< patch.low[1] << "-" << patch.high[1] << ", "
					<< patch.low[2] << "-" << patch.high[2] << "} "
					<< "flags: " << flags.size << " ";
	for(int f=0;f<flags.size;f++)
	{
			cout << "{" << flags.locs[f][0] << ", " << flags.locs[f][1] << ", " << flags.locs[f][2] << "},"; 
	}
	cout << endl;
	*/
					
	offset=-patch.low;
	//cout <<"rank:" <<  p_group[p_rank] <<": " << "pid:" << tag <<": dimensions: " << d << " processors:" << p_group.size() << endl;
	if(p_group.size()>1)
	{
		//gather # of flags on root
		status=GATHERING_FLAG_COUNT;
		//set mpi state
		stage=0;
	
		//Allocate recieve buffer
		flagscount.resize(1<<d);		//make this big enough to recieve for entire hypercube
		flagscount[0].count=flags.size;
		flagscount[0].rank=p_group[p_rank];

		//cout << "rank:" << p_group[p_rank] << ": " << "pid:" << tag  <<": gathering flags\n";
		GATHER_FLAG_COUNT:

		if(stage<d)
		{
			stride=1<<(d-1-stage);
			msg_size=1<<stage;

			//cout << "rank:" << p_group[p_rank] << ": pid:" << tag << ": stride:" << stride << " msg_size:" << msg_size << " stage:" << stage << endl;
			stage++;
			if(p_rank<stride)	//recieving
			{
				partner=p_rank+stride;
				if(partner<p_group.size())
				{
			  //cout <<"rank:" <<  p_group[p_rank] << ": pid:" << tag << ": recieving from " << p_group[partner] << " msg_size=" << msg_size << " tag=" << tag <<  endl;
					//Nonblocking recieve msg from partner
					mpi_requests.push(request);
					MPI_Irecv(&flagscount[msg_size],msg_size*sizeof(FlagsCount),MPI_BYTE,p_group[partner],tag,controller->comm,&mpi_requests.back());
					//delay_q->push(this);
					controller->delay_q.push(this);
					return;
				}
				else
				{
					
					for(int f=0;f<msg_size;f++)
					{
							flagscount[msg_size+f].count=0;
							flagscount[msg_size+f].rank=-1;
					}		
					goto GATHER_FLAG_COUNT;
				}
			}
			else if(p_rank < (stride<<1) )	//sending
			{
				int partner=p_rank-stride;
			
				//non blocking send msg of size size to partner
				//cout << "rank:" << p_group[p_rank] << ": pid:" << tag << ": sending to " << p_group[partner] << " msg_size=" << msg_size << " tag=" << tag <<  endl;
				//Nonblocking recieve msg from partner
				mpi_requests.push(request);
				MPI_Isend(&flagscount[0],msg_size*sizeof(FlagsCount),MPI_BYTE,p_group[partner],tag,controller->comm,&mpi_requests.back());
				//delay_q->push(this);
				controller->delay_q.push(this);
				return;
			}
		}
		
		status=BROADCASTING_FLAG_COUNT;
		stage=0;
		
		//cout << "rank:" << p_group[p_rank] << ": " << "pid:" << tag  << ": broadcasting flags\n";
		BROADCAST_FLAG_COUNT:
	
		if(Broadcast(&flagscount[0],flagscount.size()*sizeof(FlagsCount),MPI_BYTE,1))
			return;
		
		if(flags.size==0)	//if i don't have any flags don't participate any longer
		{
			if(parent==0)
			{
					//sort flags so this processor knows who will be broadcasting the results out
					sort(flagscount.begin(),flagscount.end());
			}
			//cout << "rank:" << p_group[p_rank] << ": pid:" << tag << ": no flags, terminating\n";
			p_rank=-1;
			goto TERMINATE;		
		}
/*
		cout << "rank:" << p_group[p_rank] << ": pid:" << tag << ": flagscount: ";
		for(int f=0;f<flagscount.size();f++)
		{
			cout << flagscount[f].rank << ":" << flagscount[f].count << ", ";
		}
		cout << endl;
*/
//		cout << "rank:" << p_group[p_rank] << ": " << "pid:" << tag  << ": creating new processor group\n";
		//sort ascending #flags
		sort(flagscount.begin(),flagscount.end());

		//update p_rank
		for(p=0;p<p_group.size();p++)
		{
			if(flagscount[p].rank==p_group[p_rank])
			{
					p_rank=p;
					break;
			}
		}
		//update p_group
		for(p=0;p<p_group.size();p++)
		{
			if(flagscount[p].count==0)
							break;
			p_group[p]=flagscount[p].rank;	
		}
		p_group.resize(p);
		
		//calculate hypercube dimensions
		p=1;		
		d=0;
		while(p<p_group.size())
		{
			p<<=1;
			d++;
		}
	/*
		cout << "rank:" << p_group[p_rank] << ": " << "pid:" << tag  << ": p_group: ";
		for(unsigned int p=0;p<p_group.size();p++)
						cout << p_group[p] << " ";
		cout << endl;
		cout << "rank:" << p_group[p_rank] << ": " << "pid:" << tag  << ": p_group.size():" << p_group.size() << " p_rank: " << p_rank << " d:" << d << endl;
	*/
		//compute #of flags on new root 
		if(p_rank==0)
		{
//			cout << "rank:" << p_group[p_rank]<< ": pid:" << tag << ": counting flags\n";
			total_flags=0;
			for(unsigned int p=0;p<p_group.size();p++)
			{
				total_flags+=flagscount[p].count;
			}
		}
	
//		cout << "rank:" << p_group[p_rank] << ": " << "pid:" << tag  << ": clearing flagscount\n";
		//give buffer back to OS
		flagscount.clear();	
	}
	else
	{
			total_flags=flags.size;
	}
	
	//cout << "rank:" << p_group[p_rank]<< ": pid:" << tag << ": computing local signature\n";
	//compute local signatures
	ComputeLocalSignature();

	if(p_group.size()>1)
	{
		sum[0].resize(patch.high[0]-patch.low[0]);
		sum[1].resize(patch.high[1]-patch.low[1]);
		sum[2].resize(patch.high[2]-patch.low[2]);
		//sum signatures
		stage=0;
		status=COMMUNICATING_SIGNATURES;
		//cout << "rank:" << p_group[p_rank] << ": " << "pid:" << tag  << ": communicating signatures\n";
		COMMUNICATE_SIGNATURES:
			
		//cout << "rank:" << p_group[p_rank] << ": " << "pid:" << tag  << ": stage:" << stage << " d:" << d << endl;
		
		//global reduce sum signatures
		if(stage<d)
		{
			stride=1<<(d-1-stage);
			//cout << "rank:" << p_group[p_rank] << ": " << "pid:" << tag  << ": stage:" << stage << " stride:" << stride << endl;
			stage++;
			//determine if i'm a sender or a reciever
			if(p_rank<stride)
			{
				partner=p_rank+stride;
				if(partner<p_group.size())
				{
					status=SUMMING_SIGNATURES;
						
			  	//cout << "rank:" << p_group[p_rank] << ": pid:" << tag << ": recieving from " << p_group[partner] <<  " tag=" << tag <<  endl;
					//Nonblocking recieve msg from partner
					mpi_requests.push(request);
					MPI_Irecv(&sum[0][0],sum[0].size(),MPI_INT,p_group[partner],tag,controller->comm,&mpi_requests.back());
					mpi_requests.push(request);
					MPI_Irecv(&sum[1][0],sum[1].size(),MPI_INT,p_group[partner],tag,controller->comm,&mpi_requests.back());
					mpi_requests.push(request);
					MPI_Irecv(&sum[2][0],sum[2].size(),MPI_INT,p_group[partner],tag,controller->comm,&mpi_requests.back());
					controller->delay_q.push(this);
					return;

					SUM_SIGNATURES:
						
					//cout << "rank:" << p_group[p_rank] << ": pid:" << tag << ": summing count\n";			
					for(int d=0;d<3;d++)
					{
					//	cout << "rank:" << p_group[p_rank] << ": pid:" << tag << ": dim:" << d << " count[d].size():" << count[d].size() << " {";			
						for(unsigned int i=0;i<count[d].size();i++)
						{
					//		cout << count[d][i] << "->";
							count[d][i]+=sum[d][i];
					//		cout << count[d][i] << " ";
						}
					//	cout << "}" << endl;
					}
						
					status=COMMUNICATING_SIGNATURES;
					goto COMMUNICATE_SIGNATURES;
				}
				else
				{
						goto COMMUNICATE_SIGNATURES;
				//	goto SUM_SIGNATURES;
				}
			}
			else if(p_rank< (stride<<1))
			{
					partner=p_rank-stride;
				  //cout << "rank:" << p_group[p_rank] <<": pid:" << tag <<  ": sending to " << p_group[partner] <<  " tag=" << tag <<  endl;
					//Nonblocking recieve msg from partner
					mpi_requests.push(request);
					
					MPI_Isend(&count[0][0],count[0].size(),MPI_INT,p_group[partner],tag,controller->comm,&mpi_requests.back());
					mpi_requests.push(request);
					MPI_Isend(&count[1][0],count[1].size(),MPI_INT,p_group[partner],tag,controller->comm,&mpi_requests.back());
					mpi_requests.push(request);
					MPI_Isend(&count[2][0],count[2].size(),MPI_INT,p_group[partner],tag,controller->comm,&mpi_requests.back());
					controller->delay_q.push(this);
					return;
			}
		}
		
//		cout << "rank:" << p_group[p_rank] <<": pid:" << tag <<  ": deallocating sum array\n";
		//deallocate sum array
		sum[0].clear();	
		sum[1].clear();	
		sum[2].clear();	
		
	}	
	
	if(p_rank==0)
	{
/*
		cout << "rank:" << p_group[p_rank] << ": Patch: " << patch.low[0] << "-" << patch.high[0] << "," << patch.low[1] << "-" << patch.high[1] << "," << patch.low[2] << "-" << patch.high[2] << endl;
		cout << "rank:" << p_group[p_rank] << ": pid:" << tag << ": Global sum: \n";
		for(int d=0;d<3;d++)
		{
	 		cout << "rank:" << p_group[p_rank] << ": pid:" << tag <<  ": dim:" <<  d << ": ";
			for(unsigned int i=0;i<count[d].size();i++)
			{
				cout << count[d][i] << " ";
			}
			cout << endl;
		}
*/		
//		cout << "rank:" << p_group[p_rank] << ": " << "pid:" << tag  << ": bounding signatures\n";
		//bound signatures
		BoundSignatures();	
/*
		cout << "rank:" << p_group[p_rank] << ": " << "pid:" << tag  << ": bounded patch: ";
		for(int dim=0;dim<3;dim++)
			cout << patch.low[dim] << "-" << patch.high[dim] << ", ";
		cout << endl;
*/
		//check tolerance a
		CheckTolA();
	}	
	
	if(p_group.size()>1)
	{
		stage=0;
		status=BROADCASTING_ACCEPTABILITY;
		//cout << "rank:" << p_group[p_rank] << ": " << "pid:" << tag  << ": broadcasting acceptablity\n";
		BROADCAST_ACCEPTABILITY:
		//broadcast acceptablity	
		if(Broadcast(&acceptable,1,MPI_INT,1))
		{
			return;
		}
	}	

	if(acceptable)
	{
		//cout << "rank:" << p_group[p_rank] << ": " << "pid:" << tag  << ": patch is acceptable\n";
		if(p_rank==0)
		{
			my_patches.push_back(patch);
		}
		//goto COMMUNICATE_PATCH_LIST;
	}
	else
	{
		//cout << "rank:" << p_group[p_rank] << ": " << "pid:" << tag  << ": patch is not acceptable\n";
		if(p_rank==0)
		{
			ctasks.split=FindSplit();
			//cout << "rank:" << p_group[p_rank] << ": " << "pid:" << tag  << ": split: d:" << ctasks.split.d << ", index:" << ctasks.split.index << endl;
			ctasks.left=ctasks.right=patch;
			ctasks.left.high[ctasks.split.d]=ctasks.right.low[ctasks.split.d]=ctasks.split.index;

			if(controller->tags.size()<2)
			{
				status=WAITING_FOR_TAGS;
				//cout << "rank:" << p_group[p_rank] << ": " << "pid:" << tag  << ": not enough tags waiting\n";
				controller->tag_q.push(this);
				return;
			}
			WAIT_FOR_TAGS:
			
			ctasks.ltag= controller->tags.front();
			controller->task_count++;
			controller->tags.pop();
			
			ctasks.rtag=controller->tags.front();
			controller->task_count++;
			controller->tags.pop();
			
			//cout << "rank:" << p_group[p_rank] << ": " << "pid:" << tag  << ": child tags: " << ctasks.ltag << " " << ctasks.rtag << endl;
		}

		if(p_group.size()>1)
		{
			status=BROADCASTING_CHILD_TASKS;
			stage=0;
			//cout << "rank:" << p_group[p_rank] << ": pid:" << tag << ": broadcasting child tasks\n";
			BROADCAST_CHILD_TASKS:
			//broadcast children tasks
			if(Broadcast(&ctasks,sizeof(ChildTasks),MPI_BYTE,0))
			{
				return;
			}
		}
		/*
		cout << "rank:" << p_group[p_rank] << ": Child patches:\n";
		cout << "left: tag:" << ctasks.ltag << endl;
		for(int d=0;d<3;d++)
			cout << ctasks.llow[d] << "-" << ctasks.lhigh[d] << ", ";
		cout << endl;
		cout << "right: tag:" << ctasks.rtag << endl;
		for(int d=0;d<3;d++)
			cout << ctasks.rlow[d] << "-" << ctasks.rhigh[d] << ", ";
		cout << endl;
		*/
		
//		cout << "rank:" << p_group[p_rank] << ": pid:" << tag << ": creating child tasks\n";
		CreateTasks();
		
		if(p_rank==0)
		{	
			status=WAITING_FOR_CHILDREN;	
			
			
			//cout << "rank:" << p_group[p_rank] << ": " << "pid:" << tag  << ": waiting for children\n";
		
			//don't place on delay_q child task will do that
			return;
		
			WAIT_FOR_CHILDREN:
			
			//cout << "rank:" << p_group[p_rank] << ": " << "pid:" << tag  << ": waiting for patch count: left_tag:" << left->tag+1 << " right_tag:" << right->tag+1 << endl;
			//begin # of patches recv
			mpi_requests.push(request);
			MPI_Irecv(&left_size,1,MPI_INT,MPI_ANY_SOURCE,left->tag+1,controller->comm,&mpi_requests.back());
			mpi_requests.push(request);
			MPI_Irecv(&right_size,1,MPI_INT,MPI_ANY_SOURCE,right->tag+1,controller->comm,&mpi_requests.back());
			//recv's might not be done yet so place back on delay_q
			status=WAITING_FOR_PATCH_COUNT;	
			controller->delay_q.push(this);
			return;
			
			WAIT_FOR_PATCH_COUNT:
			//cout << "rank:" << p_group[p_rank] << ": " << "pid:" << tag  << ": activated by child\n";
			
		//	cout << "rank:" << p_group[p_rank] << ": pid:" << tag << ": waiting for patches: left_size: " << left_size << " right_size: " << right_size << " left_tag: " << left->tag << " right_tag: " << right->tag << endl;
			status=WAITING_FOR_PATCHES;

			my_patches.resize(left_size+right_size);
			
			//recieve patchsets from children on child tag
			if(left_size>0)
			{
				mpi_requests.push(request);
				MPI_Irecv(&my_patches[0],left_size*sizeof(PseudoPatch),MPI_BYTE,MPI_ANY_SOURCE,left->tag,controller->comm,&mpi_requests.back());		
			}
			if(right_size>0)
			{
				mpi_requests.push(request);
				MPI_Irecv(&my_patches[0]+left_size,right_size*sizeof(PseudoPatch),MPI_BYTE,MPI_ANY_SOURCE,right->tag,controller->comm,&mpi_requests.back());		
			}		
			controller->delay_q.push(this);
			return;
			WAIT_FOR_PATCHES:
			
			//cout << "rank:" << p_group[p_rank] << ": " << "pid:" << tag  << ": reclaiming tag:" << left->tag << endl ;
			controller->tags.push(left->tag);		//reclaim tag
			//cout << "rank:" << p_group[p_rank] << ": " << "pid:" << tag  << ": reclaiming tag:" << right->tag << endl ;
			controller->tags.push(right->tag);		//reclaim tag
			
			//check tolerance b and take better patchset
			CheckTolB();
			if(!acceptable)
			{
				my_patches.resize(0);
				my_patches.push_back(patch);
			}
		}
	}

	
	//COMMUNICATE_PATCH_LIST:	
	if(p_rank==0 && parent!=0)
	{
		//send up the chain or to the root processor
		
		//cout << "rank:" << p_group[p_rank] << ": " << "pid:" << tag  << ": sending patchlist to parent: " << parent->p_group[0]  << " on tag: " << tag+1 << " my_patches.size():" << my_patches.size() << " parent id:" << parent->tag<< endl;
		my_size=my_patches.size();
	
		mpi_requests.push(request);
		//send patch count to parent
		MPI_Isend(&my_size,1,MPI_INT,parent->p_group[0],tag+1,controller->comm,&mpi_requests.back());
 		
		if(my_size>0)
		{
			mpi_requests.push(request);
			//send patch list to parent
			MPI_Isend(&my_patches[0],my_size*sizeof(PseudoPatch),MPI_BYTE,parent->p_group[0],tag,controller->comm,&mpi_requests.back());
		}
	}
	
	
	TERMINATE:
	
	status=TERMINATED;
	//cout << "rank:" << p_group[p_rank] << ": " << "pid:" << tag  << ": terminating\n";
	//deallocate waisted space
	count[0].clear();
	count[1].clear();
	count[2].clear();
	
	//if parent is waiting activiate parent 
	if(parent!=0 && parent->p_rank==0 && sibling->status==TERMINATED )
	{
		//cout << "rank:" << p_group[p_rank] << ": " << "pid:" << tag  << ": activating parent on tag:" << parent->tag << endl;
		//cout << "rank:" << p_group[p_rank] << ": " << "pid:" << tag  << ": sibling tag:" << sibling->tag << " sibling status:" << sibling->status << endl;
		
		//place parent on delay queue (parent is waiting for communication from children)
		controller->delay_q.push(parent);
	}

	
	//cout << "rank:" << p_group[p_rank] << ": " << "pid:" << tag  << ": mpi_requests.size():" << mpi_requests.size() << endl;
	if(!mpi_requests.empty())
	{
						
		//cout << "rank:" << p_group[p_rank] << ": " << "pid:" << tag  << ": waiting to finish sends\n";
		//must wait for final communication to finish
		controller->delay_q.push(this);
	}
	else
	{
		//cout << "rank:" << p_group[p_rank] << ": pid:" << tag  << ": task done\n";
	}
	if(parent!=0)
	{
		p_group.clear();
	}

	return; 

}

void BNRTask::ComputeLocalSignature()
{
	//resize signature count
	count[0].resize(patch.high[0]-patch.low[0]);
	count[1].resize(patch.high[1]-patch.low[1]);
	count[2].resize(patch.high[2]-patch.low[2]);

	//initialize signature
	count[0].assign(count[0].size(),0);
	count[1].assign(count[1].size(),0);
	count[2].assign(count[2].size(),0);
	
	//count flags
	for(int f=0;f<flags.size;f++)
	{
			IntVector loc=flags.locs[f]+offset;
			count[0][loc[0]]++;
			count[1][loc[1]]++;
			count[2][loc[2]]++;
	}
	/*
	cout << "rank:" << p_group[p_rank] << ": pid:" << tag << ": Local sum: \n";
	for(int d=0;d<3;d++)
	{
		cout << "rank:" << p_group[p_rank] << ": pid:" << tag <<  ": dim:" <<  d << ": ";
		for(unsigned int i=0;i<count[d].size();i++)
		{
			cout << count[d][i] << " ";
		}
		cout << endl;
	}
	*/
}
void BNRTask::BoundSignatures()
{
		IntVector low;
		IntVector high;
		IntVector size=patch.high-patch.low;
		//for each dimension
		for(int d=0;d<3;d++)
		{
			int i;
			//search for first non zero
			for(i=0;i<size[d];i++)
			{
				if(count[d][i]!=0)
					break;
			}
			low[d]=i+patch.low[d];
			//search for last non zero
			for(i=size[d]-1;i>=0;i--)
			{
				if(count[d][i]!=0)
							break;	
			}
			high[d]=i+1+patch.low[d];
		}
		patch.low=low;
		patch.high=high;
}

void BNRTask::CheckTolA()
{
	patch_vol=1;
	IntVector size=patch.high-patch.low;
	patch_vol=size[0]*size[1]*size[2];
	acceptable= float(total_flags)/patch_vol>=controller->tola;
}

void BNRTask::CheckTolB()
{
		//calculate patch volume of children
		int children_vol=0;
		for(unsigned int p=0;p<my_patches.size();p++)
		{
				IntVector size=my_patches[p].high-my_patches[p].low;
				children_vol+=size[0]*size[1]*size[2];
		}
		
		//compare to patch volume of parent
		if(float(children_vol)/patch_vol>=controller->tolb)
		{
			acceptable=false;
		}
		else
		{
			acceptable=true;
		}
}
Split BNRTask::FindSplit()
{
/*
	cout << "rank:" << p_group[p_rank] << ": " << "pid:" << tag  << ": patch.low:{"
						<< patch.low[0] << " " << patch.low[1] << " " << patch.low[2] << "} offset:{"
						<< offset[0] << " " << offset[1] << " " << offset[2] << "}\n";
*/	
	Split split;
	split.d=-1;
	IntVector size=patch.high-patch.low;
	//search for zero split in each dimension
	for(int d=0;d<3;d++)
	{
		int index=patch.low[d]+offset[d]+1;
		for(int i=1;i<size[d]-1;i++,index++)
		{
			if(count[d][index]==0)
			{
					split.d=d;
					split.index=index-offset[d];
					//cout << "rank:" << p_group[p_rank] << ": " << "pid:" << tag  << ": zero split found\n";
					return split;
			}		
		}
	}
	//no zero split found	
	//search for second derivitive split
	IntVector mid=(patch.low+patch.high)/IntVector(2,2,2);
	int max_change=-1,max_dist=INT_MAX;
		
//	cout << "rank:" << p_group[p_rank] << ": pid:" << tag << ": offset: {" << offset[0] << " " << offset[1] << " " << offset[2] << "}\n";
	for(int d=0;d<3;d++)
	{
		if(size[d]>2)
		{
//			cout << "rank:" << p_group[p_rank] << ": pid:" << tag << ": d2: dim:" << d << "{";
			int d2, last_d2;
			int s;
			
			int index=patch.low[d]+offset[d];
			last_d2=count[d][index+1]-count[d][index];
//			cout << last_d2 << " ";
			int last_s=sign(last_d2);
			index++;
			for(int i=1;i<size[d]-1;i++,index++)
			{
				d2=count[d][index-1]+count[d][index+1]-2*count[d][index];
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
							split.index=index-offset[d];
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
									split.index=index-offset[d];
									max_dist=dist;
								}
								else
								{
									//tie breaker - closest to center
									if(dist<max_dist)
									{
										max_change=change;
										split.d=d;
										split.index=index-offset[d];
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
			d2=count[d][index-1]-count[d][index];
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
						split.index=index-offset[d];
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
								split.index=index-offset[d];
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
		//cout << "rank:" << p_group[p_rank] << ": " << "pid:" << tag  << ": d2 split found, max_change was: " << max_change << endl;
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
	//cout << "rank:" << p_group[p_rank] << ": " << "pid:" << tag  << ": middle split\n";
	return split;
}

//return value indicates if there is more broadcasting to perform on this processor
bool BNRTask::Broadcast(void *message, int count, MPI_Datatype datatype,unsigned int tag_inc)
{
	unsigned int partner;
	MPI_Request request;
	//broadcast flagscount back to procs
//	cout << "rank:" << p_group[p_rank] << ": pid:" << tag << ": stage: " << stage << " d: " << d << " p_rank: " << p_rank << " p_group.size(): " << p_group.size() << " tag_inc:" << tag_inc <<  endl;
	if(stage<d)
	{
		int stride=1<<stage;
		stage++;
		if(p_rank<stride)
		{
			partner=p_rank+stride;
			if(partner<p_group.size())
			{
				//Nonblocking send msg to partner
			//cout << "rank:" << p_group[p_rank]  << ": pid:" << tag<< ": sending to " << p_group[partner] << " on tag: " << tag+tag_inc << " message length:" << count << endl;
				mpi_requests.push(request);
				MPI_Isend(message,count,datatype,p_group[partner],tag+tag_inc,controller->comm,&mpi_requests.back());
				controller->delay_q.push(this);
				return true;
			}
		}
		else if(p_rank< (stride<<1))
		{
			partner=p_rank-stride;
				
			//Nonblocking recieve msg from partner
			//cout << "rank:" << p_group[p_rank]  << ": pid:" << tag<< ": recieving from " << p_group[partner] << " on tag: " << tag+tag_inc << " message length:" << count << endl;
			mpi_requests.push(request);
			MPI_Irecv(message,count,datatype,p_group[partner],tag+tag_inc,controller->comm,&mpi_requests.back());	
			
			controller->delay_q.push(this);
			return true;
		}
		else
		{
			controller->immediate_q.push(this);
			//cout << "rank:" << p_group[p_rank]  << ": pid:" << tag<< ": not active this stage moving to next" << endl;
			return true;
		}
	}
		
	//cout << "rank:" << p_group[p_rank]  << ": pid:" << tag<< ": broadcast done\n";
	return false;		
}

void BNRTask::CreateTasks()
{
	FlagsList leftflags,rightflags;
		
	//output flags before
	//split the flags
	int front=0, back=flags.size-1;	
	int d=ctasks.split.d, v=ctasks.split.index;
	while(front<back)
	{
		if(flags.locs[front][d]<v) //place at front
		{
			front++;
		}
		else
		{
			swap(flags.locs[front],flags.locs[back]);
			back--;
		}
	}
	if(flags.locs[front][d]<v)
		front++;
	
	leftflags.locs=flags.locs;
	leftflags.size=front;

	rightflags.locs=flags.locs+front;
	rightflags.size=flags.size-front;
	/*
	cout << "rank:" << p_group[p_rank] << ": pid:" << tag << ": left patch: tag:" << ctasks.ltag << " {" 
					<< ctasks.left.low[0] << "-" << ctasks.left.high[0] << ", "
					<< ctasks.left.low[1] << "-" << ctasks.left.high[1] << ", "
					<< ctasks.left.low[2] << "-" << ctasks.left.high[2] << "}\n";
	
	
	cout << "rank:" << p_group[p_rank] << ": pid:" << tag << ": right patch: tag:" << ctasks.rtag << " {" 
					<< ctasks.right.low[0] << "-" << ctasks.right.high[0] << ", "
					<< ctasks.right.low[1] << "-" << ctasks.right.high[1] << ", "
					<< ctasks.right.low[2] << "-" << ctasks.right.high[2] << "}\n";
	*/
	/*	
	//output left flags
	cout << "rank:" << p_group[p_rank] << ": left flags:";
	for(int f=0;f<leftflags.size;f++)
	{
			cout << "{" << leftflags.locs[f][0] << "," << leftflags.locs[f][1] << "," << leftflags.locs[f][2] << "},";
	}
	cout << endl;
	//output right flags
	cout << "rank:" << p_group[p_rank] << ": right flags:";
	for(int f=0;f<rightflags.size;f++)
	{
			cout << "{" << rightflags.locs[f][0] << "," << rightflags.locs[f][1] << "," << rightflags.locs[f][2] << "},";
	}
	cout << endl;
	*/	
		
	//create new tasks		
	
	controller->tasks.push_back(BNRTask(ctasks.left,leftflags,p_group,p_rank,this,ctasks.ltag));
	left=&controller->tasks.back();
	
	controller->tasks.push_back(BNRTask(ctasks.right,rightflags,p_group,p_rank,this,ctasks.rtag));
	right=&controller->tasks.back();
	
	left->setSibling(right);	
	right->setSibling(left);	

	controller->immediate_q.push(left);
	controller->immediate_q.push(right);
	
}

	
