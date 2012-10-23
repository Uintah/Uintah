/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
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

#include <CCA/Components/Regridder/BNRRegridder.h>
#include <CCA/Components/Regridder/BNRTask.h>

using namespace Uintah;
#include <vector>
#include <set>
#include <algorithm>
#include <stdio.h>

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

BNRTask::BNRTask(Region patch, FlagsList flags, const vector<int> &p_group, int p_rank, BNRTask *parent, unsigned int tag): status_(NEW), patch_(patch), flags_(flags), parent_(parent), sibling_(0), tag_(tag), remaining_requests_(0),p_group_(p_group), p_rank_(p_rank)
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
    controller_->statuses_.push_back(MPI_Status());
    
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
  unsigned int p;
  unsigned int partner;
  int start;
  int num_ints;
  int index, shift;
  int mask;
  
  switch (status_)
  {
    case NEW:                                                             //0
      goto TASK_START;
    case REDUCING_FLAG_INFO:                                              //1
      goto REDUCE_FLAG_INFO;
    case UPDATING_FLAG_INFO:                                              //2
      goto UPDATE_FLAG_INFO;
    case BROADCASTING_FLAG_INFO:                                         //3
      goto BROADCAST_FLAG_INFO;
    case COMMUNICATING_SIGNATURES:                                        //4
      goto COMMUNICATE_SIGNATURES;
    case SUMMING_SIGNATURES:                                              //5
      goto SUM_SIGNATURES;
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
      char error[100];
      sprintf(error,"Error invalid status (%d) in parallel task\n",status_);
      throw InternalError(error,__FILE__,__LINE__);
  }
                  
  TASK_START:
  
  offset_=-patch_.getLow();
  
  if(p_group_.size()>1)
  {
    //create flag_info_

    //determine the number of integers needed for bitfield
    num_ints=p_group_.size()/(sizeof(int)*8);
    if(p_group_.size()%(sizeof(int)*8)!=0)
      num_ints++;
    
    flag_info_.resize(3+num_ints);    //1 for sum, 2 for max info, and one for each required int
    flag_info_buffer_.resize(3+num_ints);

    flag_info_.assign(3+num_ints,0);  //initialize to 0
    
    //initialize flag info
    flag_info_[0]=flag_info_[2]=flags_.size; //set sum and max
    flag_info_[1]=p_rank_;                  //set location of max
    //set bitfield
    if(flags_.size>0)
    {
      index=p_rank_/(sizeof(int)*8);    //index into flag_info_
      flag_info_[3+index]=1<<(p_rank_-index*(sizeof(int)*8));    //place a 1 in the bit field to represent me
    }
    //reduce flag info_ onto root
    status_=REDUCING_FLAG_INFO;
    //set mpi state
    stage_=0;
  
    //Gather the flags onto the root processor without blocking
    REDUCE_FLAG_INFO:

    if(stage_<d_)
    {
      stride=1<<(d_-1-stage_);

      stage_++;
      if(p_rank_<stride)  //recieving
      {
        partner=p_rank_+stride;
        if(partner<p_group_.size())
        {
          //Nonblocking recieve msg from partner
          MPI_Irecv(&flag_info_buffer_[0],flag_info_buffer_.size(),MPI_INT,p_group_[partner],tag_,controller_->d_myworld->getComm(),getRequest());
        
          status_=UPDATING_FLAG_INFO;
          return;
          
          UPDATE_FLAG_INFO:
          //update sum
          flag_info_[0]+=flag_info_buffer_[0];
          //update max
          if(flag_info_buffer_[2]>flag_info_[2] || (flag_info_buffer_[2]==flag_info_[2] && flag_info_buffer_[1]<flag_info_[1]))
          {
            flag_info_[1]=flag_info_buffer_[1];   //set new rank of max
            flag_info_[2]=flag_info_buffer_[2];   //set new  max
          }
          //update bit field
          for(int i=3;i<(int)flag_info_.size();i++)
          {
            flag_info_[i]|=flag_info_buffer_[i];
          }
        }

        status_=REDUCING_FLAG_INFO;
        goto REDUCE_FLAG_INFO;
      }
      else if(p_rank_ < (stride<<1) )  //sending
      {
        int partner=p_rank_-stride;
      
        //non blocking send msg of size size to partner
        MPI_Isend(&flag_info_[0],flag_info_.size(),MPI_INT,p_group_[partner],tag_,controller_->d_myworld->getComm(),getRequest());
        return;
      }
    }
    
    status_=BROADCASTING_FLAG_INFO;
    stage_=0;
    
    BROADCAST_FLAG_INFO:
  
    if(Broadcast(&flag_info_[0],flag_info_.size(),MPI_INT))
      return;
    
    total_flags_=flag_info_[0];
    
    //remove processors from p_group_ that have zero flags
    p=0;
    mask=1;
    for(int i=0;i<(int)p_group_.size();i++,index++)
    {
      index=i/(sizeof(int)*8);
      shift=i-(index)*(sizeof(int)*8);
      //if the bit for this local rank is set
      if( (flag_info_[3+index]>>shift)&mask )
      {
        if(i==p_rank_)          //update p_rank_
          p_rank_=p;    
        p_group_[p]=p_group_[i];  //update p_group_
        
        if(i==flag_info_[1])  //if this is the master processor
        {
            swap(p_group_[0],p_group_[p]); //place it at the front of the p_group_
            if((unsigned int)p_rank_==p)                //if i'm master
              p_rank_=0;                      //set my rank to 0
            else if(p_rank_==0)           //if i'm rank 0
              p_rank_=p;                      //set my rank to p
        }
        p++;
      }
    }
    p_group_.resize(p);
    
    //clear buffers
    flag_info_.clear();
    flag_info_buffer_.clear();
    
    if(flags_.size==0)  //if i don't have any flags don't participate any longer
    {
      p_rank_=-1;
      goto TERMINATE;    
    }
    
    //calculate hypercube dimensions
    p=1;    
    d_=0;
    while(p<p_group_.size())
    {
      p<<=1;
      d_++;
    }
  }
  else
  {
    total_flags_=flags_.size;
  }
  
  //compute local signatures
  ComputeLocalSignature();

  if(p_group_.size()>1)
  {
    sum_.resize(sig_size_);
    //sum signatures
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
          MPI_Irecv(&sum_[0],sig_size_,MPI_INT,p_group_[partner],tag_,controller_->d_myworld->getComm(),getRequest());
          return;

          SUM_SIGNATURES:
          
          for(int i=0;i<sig_size_;i++)
          {
            count_[i]+=sum_[i];
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
        MPI_Isend(&count_[0],sig_size_,MPI_INT,p_group_[partner],tag_,controller_->d_myworld->getComm(),getRequest());
        return;
      }
    }
    //deallocate sum_ array
    sum_.clear();  
  }  
  
  //controlling task determines if the current patch is good or not
  //if not find the location to split this patch
  if(p_rank_==0)
  {
    //bound signatures
    BoundSignatures();  
    
    //check tolerance a
    CheckTolA();

    if(acceptable_)
    {
      //set d=-1 to signal that the patch is acceptable
      ctasks_.split.d=-1;
    }
    else
    {
      //find split
      ctasks_.split=FindSplit();
      
      //split the current patch
      ctasks_.left=ctasks_.right=patch_;
      ctasks_.left.high()[ctasks_.split.d]=ctasks_.right.low()[ctasks_.split.d]=ctasks_.split.index;

      WAIT_FOR_TAGS:
      //check if tags are available
      if(!controller_->getTags(ctasks_.ltag,ctasks_.rtag) )
      {
        status_=WAITING_FOR_TAGS;
        controller_->tag_q_.push(this);
        return;
      }
    }
  }  
  
  //signature is no longer needed so free memory
  count_.clear();
  //broadcast child tasks
  if(p_group_.size()>1)
  {
    status_=BROADCASTING_CHILD_TASKS;
    stage_=0;
    BROADCAST_CHILD_TASKS:
    //broadcast children tasks
    if(Broadcast(&ctasks_,sizeof(ChildTasks),MPI_BYTE))
    {
      return;
    }
  }
  
  if(ctasks_.split.d==-1)
  {
      //current patch is acceptable
      my_patches_.push_back(patch_);
  }
  else
  {
    //create tasks
    CreateTasks();
  
    //Wait for childern to reactivate parent
    status_=WAITING_FOR_CHILDREN;  
    return;
    WAIT_FOR_CHILDREN:
    
    if(p_rank_==0)
    {  
      //begin # of patches recv

      //if i'm also the master of the children tasks copy the patches from the child task
      if(left_->p_group_[0]==p_group_[0])
      {
         left_size_=left_->my_patches_.size();
         my_patches_.assign(left_->my_patches_.begin(),left_->my_patches_.end());
      }
      else
      {
        MPI_Irecv(&left_size_,1,MPI_INT,left_->p_group_[0],left_->tag_,controller_->d_myworld->getComm(),getRequest());
      }
      if(right_->p_group_[0]==p_group_[0])
      {
         right_size_=right_->my_patches_.size();
         my_patches_.insert(my_patches_.end(),right_->my_patches_.begin(),right_->my_patches_.end());
      }
      else
      {
        MPI_Irecv(&right_size_,1,MPI_INT,right_->p_group_[0],right_->tag_,controller_->d_myworld->getComm(),getRequest());
      }
      //recv's might not be done yet so place back on delay_q
      status_=WAITING_FOR_PATCH_COUNT;  
      if(remaining_requests_>0)
        return;
      
      WAIT_FOR_PATCH_COUNT:
      status_=WAITING_FOR_PATCHES;
      
      start=my_patches_.size();               //start of receive buff
      //resize my_patches_ buffer to recieve
      my_patches_.resize(left_size_+right_size_);
     
      //recieve patch_sets from children on child tag only if it hasn't been copied already
      if(left_->p_group_[0]!=p_group_[0])
      {
        MPI_Irecv(&my_patches_[start],left_size_*sizeof(Region),MPI_BYTE,left_->p_group_[0],left_->tag_,controller_->d_myworld->getComm(),getRequest());    
        start+=left_size_;                        //move recieve buffer forward
      }
      if(right_->p_group_[0]!=p_group_[0])
      {
        MPI_Irecv(&my_patches_[start],right_size_*sizeof(Region),MPI_BYTE,right_->p_group_[0],right_->tag_,controller_->d_myworld->getComm(),getRequest());    
      }    
      if(remaining_requests_>0)
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
    } //if(p_rank_==0)
  }
  
  //COMMUNICATE_PATCH_LIST:  
  if(p_rank_==0 && parent_!=0)
  {
    if(p_group_[0]!=parent_->p_group_[0]) //if I am not the same rank as the parent master process
    {
      //send up to the master using mpi
      my_size_=my_patches_.size();
  
      //send patch_ count to parent
      MPI_Isend(&my_size_,1,MPI_INT,parent_->p_group_[0],tag_,controller_->d_myworld->getComm(),getRequest());
     
      if(my_size_>0)
      {
        //send patch list to parent
        MPI_Isend(&my_patches_[0],my_size_*sizeof(Region),MPI_BYTE,parent_->p_group_[0],tag_,controller_->d_myworld->getComm(),getRequest());
      }
    }
  }
  
  TERMINATE:
  
  status_=TERMINATED;
  
  //if parent is waiting activiate parent 
  if(parent_!=0 && sibling_->status_==TERMINATED )
  {
    //place parent_ on immediate queue (parent is waiting for communication from children and both are done sending)
    controller_->immediate_q_.push(parent_);
  }
  
  return; 
}

/***************************************************
 * Same as continueTask but on 1 processor only
 * ************************************************/
void BNRTask::continueTaskSerial()
{
  switch (status_)
  {
          case NEW:                                                             //0
                  goto TASK_START;
          case WAITING_FOR_CHILDREN:                                            //8
                  goto WAIT_FOR_CHILDREN;
          default:
                  char error[100];
                  sprintf(error,"Error invalid status (%d) in serial task\n",status_);
                  throw InternalError(error,__FILE__,__LINE__);
  }
                  
  TASK_START:
          
  offset_=-patch_.getLow();
  
  //compute local signatures
  ComputeLocalSignature();

  BoundSignatures();  
  
  total_flags_=flags_.size;
  
  CheckTolA();
  if(acceptable_)
  {
    my_patches_.push_back(patch_);
      
    //signature is no longer needed so free memory
    count_.clear();
  }
  else
  {
    ctasks_.split=FindSplit();
      
    //split the current patch
    ctasks_.left=ctasks_.right=patch_;
    ctasks_.left.high()[ctasks_.split.d]=ctasks_.right.low()[ctasks_.split.d]=ctasks_.split.index;
    
    ctasks_.ltag=-1;
    ctasks_.rtag=-1;
     
    //signature is no longer needed so free memory
    count_.clear();
      
    CreateTasks();
    
    status_=WAITING_FOR_CHILDREN;  

    return;
    
    WAIT_FOR_CHILDREN:
    
    if(left_->tag_!=-1 || right_->tag_!=-1)
    {
      controller_->tags_.push(left_->tag_);    //reclaim tag
      controller_->tags_.push(right_->tag_);    //reclaim tag
    }
    //copy patches from left children
    my_patches_.assign(left_->my_patches_.begin(),left_->my_patches_.end());
    my_patches_.insert(my_patches_.end(),right_->my_patches_.begin(),right_->my_patches_.end());
    
    //check tolerance b and take better patchset
    CheckTolB();
    if(!acceptable_)
    {
      my_patches_.resize(0);
      my_patches_.push_back(patch_);
    }
  }
  
  //COMMUNICATE_PATCH_LIST:  
  if( parent_!=0 && parent_->p_group_[0]!=p_group_[0])  //if parent exists and I am not also the master on the parent
  {
    {
      //send up the chain or to the root processor
      my_size_=my_patches_.size();
  
      //send patch count to parent
      MPI_Isend(&my_size_,1,MPI_INT,parent_->p_group_[0],tag_,controller_->d_myworld->getComm(),getRequest());
     
      if(my_size_>0)
      {
        //send patch list to parent
        MPI_Isend(&my_patches_[0],my_size_*sizeof(Region),MPI_BYTE,parent_->p_group_[0],tag_,controller_->d_myworld->getComm(),getRequest());
      }
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
  IntVector size=patch_.getHigh()-patch_.getLow();
  sig_offset_[0]=0;
  sig_offset_[1]=size[0];
  sig_offset_[2]=size[0]+size[1];
  sig_size_=size[0]+size[1]+size[2];
  
  //resize signature count_
  count_.resize(sig_size_);

  //initialize signature
  count_.assign(sig_size_,0);
  
  //count flags
  for(int f=0;f<flags_.size;f++)
  {
      IntVector loc=flags_.locs[f]+offset_;
      count_[loc[0]]++;
      count_[sig_offset_[1]+loc[1]]++;
      count_[sig_offset_[2]+loc[2]]++;
  }
}
void BNRTask::BoundSignatures()
{
    IntVector low;
    IntVector high;
    IntVector size=patch_.getHigh()-patch_.getLow();
    //for each dimension
    for(int d=0;d<3;d++)
    {
      int i;
      //search for first non zero
      for(i=0;i<size[d];i++)
      {
        if(count_[sig_offset_[d]+i]!=0)
          break;
      }
      low[d]=i+patch_.getLow()[d];
      //search for last non zero
      for(i=size[d]-1;i>=0;i--)
      {
        if(count_[sig_offset_[d]+i]!=0)
              break;  
      }
      high[d]=i+1+patch_.getLow()[d];
    }
    patch_=Region(low,high);
}

void BNRTask::CheckTolA()
{
  IntVector size=patch_.getHigh()-patch_.getLow();
  acceptable_= float(total_flags_)/patch_.getVolume()>=controller_->tola_;
}

void BNRTask::CheckTolB()
{
  //calculate patch_ volume of children
  int children_vol=0;
  for(unsigned int p=0;p<my_patches_.size();p++)
  {
      children_vol+=my_patches_[p].getVolume();
  }
  //compare to patch volume of parent
  if(float(children_vol)/patch_.getVolume()>=controller_->tolb_)
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
  split.index=0;

  IntVector size=patch_.getHigh()-patch_.getLow();
  //search for zero split in each dimension
  for(int d=0;d<3;d++)
  {
    int index=patch_.getLow()[d]+offset_[d]+1;
    for(int i=1;i<size[d]-1;i++,index++)
    {
      if(count_[sig_offset_[d]+index]==0)
      {
          split.d=d;
          split.index=index-offset_[d];
          return split;
      }    
    }
  }
  //no zero split found  
  //search for second derivitive split
  IntVector mid=(patch_.getLow()+patch_.getHigh())/IntVector(2,2,2);
  int max_change=-1,max_dist=INT_MAX;
    
  for(int d=0;d<3;d++)
  {
    if(size[d]>2)
    {
      int d2, last_d2;
      int s;
      
      int index=patch_.getLow()[d]+offset_[d];
      last_d2=count_[sig_offset_[d]+index+1]-count_[sig_offset_[d]+index];
      int last_s=sign(last_d2);
      index++;
      for(int i=1;i<size[d]-1;i++,index++)
      {
        d2=count_[sig_offset_[d]+index-1]+count_[sig_offset_[d]+index+1]-2*count_[sig_offset_[d]+index];
        s=sign(d2);
        
        //if sign change
        if(last_s!=s)
        {
          int change=abs(last_d2-d2);
          int dist=abs(mid[d]-index+offset_[d]);
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
      d2=count_[sig_offset_[d]+index-1]-count_[sig_offset_[d]+index];
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
bool BNRTask::Broadcast(void *message, int count_, MPI_Datatype datatype)
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
        MPI_Isend(message,count_,datatype,p_group_[partner],tag_,controller_->d_myworld->getComm(),getRequest());
        return true;
      }
    }
    else if(p_rank_< (stride<<1))
    {
      partner=p_rank_-stride;
        
      //Nonblocking recieve msg from partner
      MPI_Irecv(message,count_,datatype,p_group_[partner],tag_,controller_->d_myworld->getComm(),getRequest());  
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
  controller_->task_count_+=2;
}

  
