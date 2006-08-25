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

BNRTask::BNRTask(PseudoPatch patch, FlagsList flags, const vector<int> &p_group, int p_rank, BNRTask *parent, unsigned int tag): status_(NEW), patch_(patch), flags_(flags), parent_(parent), sibling_(0), tag_(tag), remaining_requests_(0),p_group_(p_group), p_rank_(p_rank)
{
  //calculate hypercube dimensions
   unsigned int p=1;
  d_=0;
  while(p<p_group_.size())
  {
    p<<=1;
    d_++;
  }
}

MPI_Request* BNRTask::getRequest()
{
  remaining_requests_++;
  if(controller_->free_requests_.empty())
  {
    int index=controller_->requests_.size();
    
    //allocate a new request
    MPI_Request request;
    controller_->requests_.push_back(request);
    controller_->indicies_.push_back(0);
    
    //assign request
    controller_->request_to_task_.push_back(this);
    return &controller_->requests_[index]; 
  }
  else
  {
    //get a free request
    int index=controller_->free_requests_.front();
    
    //assign request
    controller_->free_requests_.pop();
    controller_->request_to_task_[index]=this;
    return &controller_->requests_[index]; 
  }
}

/*
 *  This function continues a task from where it left off
 *  Each task runs through the BR algorithm and performs
 *  communication where needed.  When the task is unable
 *  to make progress by waiting on communication it terminates 
 *  and is restarted later by the main controll loop in
 *  BNRRegridder::RunBR().  I use goto's in this and I know 
 *  they are bad form but in this case goto's make the algorithm 
 *  easier to understand.
 *
 *  Most local variables are stored as class variables so the 
 *  state will remain the same when a task is restarted.  
 */
void BNRTask::continueTask()
{
  int stride;
  int msg_size;
  unsigned int p;
  unsigned int partner;
  
  switch (status_)
  {
    case NEW:                                                             //0
      goto TASK_START;
    case GATHERING_FLAG_COUNT:                                            //1
      goto GATHER_FLAG_COUNT;
    case BROADCASTING_FLAG_COUNT:                                         //2
      goto BROADCAST_FLAG_COUNT;
    case COMMUNICATING_SIGNATURES:                                        //3
      goto COMMUNICATE_SIGNATURES;
    case SUMMING_SIGNATURES:                                              //4
      goto SUM_SIGNATURES;
    case BROADCASTING_ACCEPTABILITY:                                      //5
      goto BROADCAST_ACCEPTABILITY;
    case WAITING_FOR_TAGS:                                                //6
      goto WAIT_FOR_TAGS;                                            
    case BROADCASTING_CHILD_TASKS:                                        //7
      goto BROADCAST_CHILD_TASKS;
    case WAITING_FOR_CHILDREN:                                            //8
      goto WAIT_FOR_CHILDREN;                                  
    case WAITING_FOR_PATCH_COUNT:                                         //9
      goto WAIT_FOR_PATCH_COUNT;
    case WAITING_FOR_PATCHES:                                             //10
      goto WAIT_FOR_PATCHES;
    case TERMINATED:                                                      //11
      return;
     default:
      cerr << "rank:" << p_group_[p_rank_] << ": " << "pid:" << tag_  << ": error invalid status_: " << status_ << endl;
      return;
  }
                  
  TASK_START:
  
  offset_=-patch_.low;
  
  if(p_group_.size()>1)
  {
    //gather # of flags_ on root
    status_=GATHERING_FLAG_COUNT;
    //set mpi state
    stage_=0;
  
    //Allocate recieve buffer
    flagscount_.resize(1<<d_);    //make this big enough to recieve for entire hypercube
    flagscount_[0].count=flags_.size;
    flagscount_[0].rank=p_group_[p_rank_];

    //Gather the flags onto the root processor without blocking
    GATHER_FLAG_COUNT:

    if(stage_<d_)
    {
      stride=1<<(d_-1-stage_);
      msg_size=1<<stage_;

      stage_++;
      if(p_rank_<stride)  //recieving
      {
        partner=p_rank_+stride;
        if(partner<p_group_.size())
        {
          //Nonblocking recieve msg from partner
          MPI_Irecv(&flagscount_[msg_size],msg_size*sizeof(FlagsCount),MPI_BYTE,p_group_[partner],tag_,controller_->d_myworld->getComm(),getRequest());
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
      else if(p_rank_ < (stride<<1) )  //sending
      {
        int partner=p_rank_-stride;
      
        //non blocking send msg of size size to partner
        MPI_Isend(&flagscount_[0],msg_size*sizeof(FlagsCount),MPI_BYTE,p_group_[partner],tag_,controller_->d_myworld->getComm(),getRequest());
        return;
      }
    }
    
    status_=BROADCASTING_FLAG_COUNT;
    stage_=0;
    
    BROADCAST_FLAG_COUNT:
  
    if(Broadcast(&flagscount_[0],flagscount_.size()*sizeof(FlagsCount),MPI_BYTE,1))
      return;
    
    if(flags_.size==0)  //if i don't have any flags don't participate any longer
    {
      if(parent_==0)
      {
        //sort flags_ so this processor knows who will be broadcasting the results out
        sort(flagscount_.begin(),flagscount_.end());
        p_group_[0]=flagscount_[0].rank;        
      }
      p_rank_=-1;
      goto TERMINATE;    
    }
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
    //compute total # of flags on new root 
    if(p_rank_==0)
    {
      total_flags_=0;
      for(unsigned int p=0;p<p_group_.size();p++)
      {
        total_flags_+=flagscount_[p].count;
      }
    }
  
    //give buffer back to OS
    flagscount_.clear();  
  }
  else
  {
    total_flags_=flags_.size;
  }
  
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
    COMMUNICATE_SIGNATURES:
      
    //global reduce sum signatures
    if(stage_<d_)
    {
      stride=1<<(d_-1-stage_);
      stage_++;
      //determine if i'm a sender or a reciever
      if(p_rank_<stride)
      {
        partner=p_rank_+stride;
        if(partner<p_group_.size())
        {
          status_=SUMMING_SIGNATURES;
          
          //Nonblocking recieve msg from partner
          MPI_Irecv(&sum_[0][0],sum_[0].size(),MPI_INT,p_group_[partner],tag_,controller_->d_myworld->getComm(),getRequest());
          MPI_Irecv(&sum_[1][0],sum_[1].size(),MPI_INT,p_group_[partner],tag_,controller_->d_myworld->getComm(),getRequest());
          MPI_Irecv(&sum_[2][0],sum_[2].size(),MPI_INT,p_group_[partner],tag_,controller_->d_myworld->getComm(),getRequest());
          return;

          SUM_SIGNATURES:
            
          for(int d=0;d<3;d++)
          {
            for(unsigned int i=0;i<count_[d].size();i++)
            {
              count_[d][i]+=sum_[d][i];
            }
          }
            
          status_=COMMUNICATING_SIGNATURES;
          goto COMMUNICATE_SIGNATURES;
        }
        else
        {
            goto COMMUNICATE_SIGNATURES;
        }
      }
      else if(p_rank_< (stride<<1))
      {
          partner=p_rank_-stride;
          
          //Nonblocking recieve msg from partner
          MPI_Isend(&count_[0][0],count_[0].size(),MPI_INT,p_group_[partner],tag_,controller_->d_myworld->getComm(),getRequest());
          MPI_Isend(&count_[1][0],count_[1].size(),MPI_INT,p_group_[partner],tag_,controller_->d_myworld->getComm(),getRequest());
          MPI_Isend(&count_[2][0],count_[2].size(),MPI_INT,p_group_[partner],tag_,controller_->d_myworld->getComm(),getRequest());
          return;
      }
    }
    //deallocate sum_ array
    sum_[0].clear();  
    sum_[1].clear();  
    sum_[2].clear();  
  }  
  
  if(p_rank_==0)
  {
    //bound signatures
    BoundSignatures();  
    
    //check tolerance a
    CheckTolA();
  }  
  
  if(p_group_.size()>1)
  {
    stage_=0;
    status_=BROADCASTING_ACCEPTABILITY;
    BROADCAST_ACCEPTABILITY:
    //broadcast acceptablity  
    if(Broadcast(&acceptable_,1,MPI_INT,1))
    {
      return;
    }
  }  

  if(acceptable_)
  {
    if(p_rank_==0)
    {
      my_patches_.push_back(patch_);
    }
    //signature is no longer needed so free memory
    count_[0].clear();
    count_[1].clear();
    count_[2].clear();
  }
  else
  {
    if(p_rank_==0)
    {
      ctasks_.split=FindSplit();
      ctasks_.left=ctasks_.right=patch_;
      ctasks_.left.high[ctasks_.split.d]=ctasks_.right.low[ctasks_.split.d]=ctasks_.split.index;
    
      //signature is no longer needed so free memory
      count_[0].clear();
      count_[1].clear();
      count_[2].clear();

      //check if tags are available
      if(controller_->tags_.size()<2)
      {
        status_=WAITING_FOR_TAGS;
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
    }
    else
    {
      //signature is no longer needed so free memory
      count_[0].clear();
      count_[1].clear();
      count_[2].clear();
    }

    if(p_group_.size()>1)
    {
      status_=BROADCASTING_CHILD_TASKS;
      stage_=0;
      BROADCAST_CHILD_TASKS:
      //broadcast children tasks
      if(Broadcast(&ctasks_,sizeof(ChildTasks),MPI_BYTE,0))
      {
        return;
      }
    }
    
    CreateTasks();
    
    status_=WAITING_FOR_CHILDREN;  
    return;
    
    WAIT_FOR_CHILDREN:
    
    if(p_rank_==0)
    {  
      
      //begin # of patches recv
      MPI_Irecv(&left_size_,1,MPI_INT,MPI_ANY_SOURCE,left_->tag_+1,controller_->d_myworld->getComm(),getRequest());
      MPI_Irecv(&right_size_,1,MPI_INT,MPI_ANY_SOURCE,right_->tag_+1,controller_->d_myworld->getComm(),getRequest());
      //recv's might not be done yet so place back on delay_q
      status_=WAITING_FOR_PATCH_COUNT;  
      return;
      
      WAIT_FOR_PATCH_COUNT:
      status_=WAITING_FOR_PATCHES;
      
      my_patches_.resize(left_size_+right_size_);
      
      //recieve patch_sets from children on child tag
      if(left_size_>0)
      {
        MPI_Irecv(&my_patches_[0],left_size_*sizeof(PseudoPatch),MPI_BYTE,MPI_ANY_SOURCE,left_->tag_,controller_->d_myworld->getComm(),getRequest());    
      }
      if(right_size_>0)
      {
        MPI_Irecv(&my_patches_[0]+left_size_,right_size_*sizeof(PseudoPatch),MPI_BYTE,MPI_ANY_SOURCE,right_->tag_,controller_->d_myworld->getComm(),getRequest());    
      }    
      return;
      WAIT_FOR_PATCHES:
      
      controller_->tags_.push(left_->tag_);    //reclaim tag
      controller_->tags_.push(right_->tag_);    //reclaim tag
      
      //check tolerance b and take better patchset
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
    my_size_=my_patches_.size();
  
    //send patch_ count to parent
    MPI_Isend(&my_size_,1,MPI_INT,parent_->p_group_[0],tag_+1,controller_->d_myworld->getComm(),getRequest());
     
    if(my_size_>0)
    {
      //send patch list to parent
      MPI_Isend(&my_patches_[0],my_size_*sizeof(PseudoPatch),MPI_BYTE,parent_->p_group_[0],tag_,controller_->d_myworld->getComm(),getRequest());
    }
  }
  
  TERMINATE:
  
  status_=TERMINATED;
  
  //if parent is waiting activiate parent 
  if(parent_!=0 && sibling_->status_==TERMINATED )
  {
    
    //place parent_ on delay queue (parent is waiting for communication from children)
    controller_->immediate_q_.push(parent_);
  }
  
  return; 
}

/***************************************************
 * Same as continue task but on 1 processor only
 * ************************************************/
void BNRTask::continueTaskSerial()
{
  
  switch (status_)
  {
          case NEW:                                                             //0
                  goto TASK_START;
          case WAITING_FOR_CHILDREN:                                            //8
                  goto WAIT_FOR_CHILDREN;
          case GATHERING_FLAG_COUNT:                                            //1
          case BROADCASTING_FLAG_COUNT:                                         //2
          case COMMUNICATING_SIGNATURES:                                        //3
          case SUMMING_SIGNATURES:                                              //4
          case BROADCASTING_ACCEPTABILITY:                                      //5
          case WAITING_FOR_TAGS:                                                //6
          case BROADCASTING_CHILD_TASKS:                                        //7
          case WAITING_FOR_PATCH_COUNT:                                         //9
          case WAITING_FOR_PATCHES:                                             //10
          case TERMINATED:                                                      //11
          default:
                  cout << "Error invalid status(" << status_ << ") in serial task\n";
                  exit(0);
                  return;
  }
                  
  TASK_START:
          
  offset_=-patch_.low;
  
  //compute local signatures
  ComputeLocalSignature();

  BoundSignatures();  
  
  total_flags_=flags_.size;
  
  CheckTolA();
  if(acceptable_)
  {
    my_patches_.push_back(patch_);
      
    //signature is no longer needed so free memory
    count_[0].clear();
    count_[1].clear();
    count_[2].clear();
  }
  else
  {
    ctasks_.split=FindSplit();
     
    //signature is no longer needed so free memory
    count_[0].clear();
    count_[1].clear();
    count_[2].clear();

    ctasks_.left=ctasks_.right=patch_;
    ctasks_.left.high[ctasks_.split.d]=ctasks_.right.low[ctasks_.split.d]=ctasks_.split.index;
    ctasks_.ltag=0;
    ctasks_.rtag=0;
    controller_->task_count_++;
    controller_->task_count_++;
      
    CreateTasks();
    
    status_=WAITING_FOR_CHILDREN;  

    return;
    
    WAIT_FOR_CHILDREN:
      
    //copy patches from left children
    for(unsigned int p=0;p<left_->my_patches_.size();p++)
    {
      my_patches_.push_back(left_->my_patches_[p]);
    }
    //copy patches from left and right children
    for(unsigned int p=0;p<right_->my_patches_.size();p++)
    {
      my_patches_.push_back(right_->my_patches_[p]);
    }
        
    //check tolerance b and take better patchset
    CheckTolB();
    if(!acceptable_)
    {
      my_patches_.resize(0);
      my_patches_.push_back(patch_);
    }
  }
  
  //COMMUNICATE_PATCH_LIST:  
  if( parent_!=0 && parent_->p_group_.size()!=1)
  {
    //send up the chain or to the root processor
    my_size_=my_patches_.size();
  
    //send patch count to parent
    MPI_Isend(&my_size_,1,MPI_INT,parent_->p_group_[0],tag_+1,controller_->d_myworld->getComm(),getRequest());
     
    if(my_size_>0)
    {
      //send patch list to parent
      MPI_Isend(&my_patches_[0],my_size_*sizeof(PseudoPatch),MPI_BYTE,parent_->p_group_[0],tag_,controller_->d_myworld->getComm(),getRequest());
    }
  }
  
  status_=TERMINATED;
  
  //if parent is waiting activiate parent 
  if(parent_!=0 && sibling_->status_==TERMINATED )
  {
    //place parent on immediate queue 
    controller_->immediate_q_.push(parent_);
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
  
  //count flags
  for(int f=0;f<flags_.size;f++)
  {
      IntVector loc=flags_.locs[f]+offset_;
      count_[0][loc[0]]++;
      count_[1][loc[1]]++;
      count_[2][loc[2]]++;
  }
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
    size=high-low;
    patch_.volume=size[0]*size[1]*size[2];
}

void BNRTask::CheckTolA()
{
  IntVector size=patch_.high-patch_.low;
  acceptable_= float(total_flags_)/patch_.volume>=controller_->tola_;
}

void BNRTask::CheckTolB()
{
  //calculate patch_ volume of children
  int children_vol=0;
  for(unsigned int p=0;p<my_patches_.size();p++)
  {
      children_vol+=my_patches_[p].volume;
  }
  //compare to patch volume of parent
  if(float(children_vol)/patch_.volume>=controller_->tolb_)
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
          return split;
      }    
    }
  }
  //no zero split found  
  //search for second derivitive split
  IntVector mid=(patch_.low+patch_.high)/IntVector(2,2,2);
  int max_change=-1,max_dist=INT_MAX;
    
  for(int d=0;d<3;d++)
  {
    if(size[d]>2)
    {
      int d2, last_d2;
      int s;
      
      int index=patch_.low[d]+offset_[d];
      last_d2=count_[d][index+1]-count_[d][index];
      int last_s=sign(last_d2);
      index++;
      for(int i=1;i<size[d]-1;i++,index++)
      {
        d2=count_[d][index-1]+count_[d][index+1]-2*count_[d][index];
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
  return split;
}
/************************************************************
 * Broadcast message of size count in a non blocking fashion
 * the return value indicates if there is more broadcasting to 
 * perform on this processor
 * *********************************************************/
bool BNRTask::Broadcast(void *message, int count_, MPI_Datatype datatype,unsigned int tag_inc)
{
  unsigned int partner;
  //broadcast flagscount_ back to procs
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
        MPI_Isend(message,count_,datatype,p_group_[partner],tag_+tag_inc,controller_->d_myworld->getComm(),getRequest());
        return true;
      }
    }
    else if(p_rank_< (stride<<1))
    {
      partner=p_rank_-stride;
        
      //Nonblocking recieve msg from partner
      MPI_Irecv(message,count_,datatype,p_group_[partner],tag_+tag_inc,controller_->d_myworld->getComm(),getRequest());  
      return true;
    }
    else
    {
      controller_->immediate_q_.push(this);
      return true;
    }
  }
    
  return false;    
}

void BNRTask::CreateTasks()
{
  FlagsList leftflags_,rightflags_;
    
  //split the flags
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
    
  //create new tasks   
  
  controller_->tasks_.push_back(BNRTask(ctasks_.left,leftflags_,p_group_,p_rank_,this,ctasks_.ltag));
  left_=&controller_->tasks_.back();
  
  controller_->tasks_.push_back(BNRTask(ctasks_.right,rightflags_,p_group_,p_rank_,this,ctasks_.rtag));
  right_=&controller_->tasks_.back();
  
  left_->setSibling(right_);  
  right_->setSibling(left_);  

  controller_->immediate_q_.push(left_);
  controller_->immediate_q_.push(right_);
}

  
