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

#include <testprograms/Regridders/GBRv2Regridder.h>

using namespace std;

namespace Uintah {

bool GBRv2Regridder::getTags(int &tag1, int &tag2)
{

  int free_tags=free_tag_end_-free_tag_start_;
  
  //check if queue has tags
  if(tags_.size()>1)
  {
    tag1=tags_.front(); 
    tags_.pop();
    tag2=tags_.front();
    tags_.pop(); 
    return true;  
  }
  //check if tags can be allocated 
  else if(free_tags>1)
  {
    tag1=free_tag_start_; free_tag_start_++;
    tag2=free_tag_start_; free_tag_start_++;
    return true;
  }
  //check if 1 tag is on the queue and 1 avialable at the end
  else if(tags_.size()==1 && free_tags==1)
  {
    tag1=tags_.front();
    tags_.pop();
    tag2=free_tag_start_; free_tag_start_++;
    return true;
  }
  //no more tags available
  else
  {
    return false;
  }
}
GBRv2Regridder::GBRv2Regridder(double tol, IntVector rr, int rank, int num_procs) : task_count_(0),tol_(tol), rank(rank), numprocs(num_procs), rr(rr)
{
  
  int *tag_ub, maxtag_ ,flag;

  //generate tag lists for processors
  if(numprocs>1)
  {  
    MPI_Attr_get(MPI_COMM_WORLD,MPI_TAG_UB,&tag_ub,&flag);
    if(flag)
      maxtag_=*tag_ub;
    else
      maxtag_=32767;

    int div=maxtag_/numprocs;
    int rem=maxtag_%numprocs;
  
    free_tag_start_=div*rank;
  
    if(rank<rem)
      free_tag_start_+=rank;
    else
      free_tag_start_+=rem;
  
    if(rank<rem)
      free_tag_end_=free_tag_start_+div+1;
    else
      free_tag_end_=free_tag_start_+div;

    //don't have zero in the tag list  
    if(rank==0)
      free_tag_start_++;
  }
}

void GBRv2Regridder::regrid( const vector<IntVector> &flags, vector<Region> &patches)
{
  vector<IntVector> flags2(flags);
  patches.resize(0);
  RunBR(flags2,patches);
  for(size_t i=0;i<patches.size();i++)
  {
    Region &patch=patches[i];
    patch.low()=patch.low()*rr;
    patch.high()=patch.high()*rr;
  }
}
void GBRv2Regridder::RunBR( vector<IntVector> &flags, vector<Region> &patches)
{
  vector<int> procs(numprocs);
  BNRTask *root=0;  
  //bound local flags
  Region patch;
  if(flags.size()>0)
  {
    patch.low()=patch.high()=flags[0];
    for(unsigned int f=1;f<flags.size();f++)
    {
      patch.low()=Min(patch.getLow(),flags[f]);
      patch.high()=Max(patch.getHigh(),flags[f]);
    }
    //make high bounds non-inclusive
    patch.high()=patch.getHigh()+IntVector(1,1,1);
  }
  else
  {
    //use INT_MAX to signal no patch;
    patch.low()[0]=INT_MAX;
  }
  int prank=-1;
  //Calculate global bounds
  if(numprocs>1)
  {
    vector<Region> bounds(numprocs);
    MPI_Allgather(&patch,sizeof(Region),MPI_BYTE,&bounds[0],sizeof(Region),MPI_BYTE,MPI_COMM_WORLD);

    //calculate participating processor set
    int count=0;
    for(int p=0;p<numprocs;p++)
    {
      if(bounds[p].getLow()[0]!=INT_MAX)
      {
        if(p==rank)
        {
          prank=count;
        }
        procs[count++]=p;   
      }
    }
  
    if(count==0) 
    {
      //no flags on any processors so exit
      return;   
    }
            
    procs.resize(count);
    
    //find the bounds
    patch=bounds[procs[0]];
    for(int p=1;p<count;p++)
    {
      patch.low()=Min(patch.getLow(),bounds[procs[p]].getLow());
      patch.high()=Max(patch.getHigh(),bounds[procs[p]].getHigh());
    }
  }
  
  if(flags.size()>0)
  {
    //create initial task
    BNRTask::controller_=this;
    FlagsList flagslist;
 
    flagslist.locs=&flags[0];
    flagslist.size=flags.size();
    tasks_.push_back(BNRTask(patch,flagslist,procs,prank,0,0));
    root=&tasks_.back();
 
    //place on immediate_q_
    immediate_q_.push(root);                  
    //control loop
    //MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    while(true)
    {
      BNRTask *task;
      //check tag_q for processors waiting for tags

      if(!tag_q_.empty() && tags_.size() + free_tag_end_ - free_tag_start_>1 )
      {
        //2 tags are available continue the task
        task=tag_q_.front();
        tag_q_.pop();
        task->continueTask();
      }
      else if(!immediate_q_.empty())  //check for tasks that are able to make progress
      {
        task=immediate_q_.front();
        immediate_q_.pop();
        //runable task found, continue task
        if(task->p_group_.size()==1)
          task->continueTaskSerial();
        else
          task->continueTask();
      }
      else if(free_requests_.size()<requests_.size())  //no tasks can make progress finish communication
      {
        int count;
        //wait on requests
        //MPI_STATUSES_IGNORE
        if(MPI_Waitsome(requests_.size(),&requests_[0],&count,&indicies_[0],&statuses_[0])==MPI_ERR_IN_STATUS)
        {
                BNRTask *task;
                cerr << "rank:" << rank << " error in MPI_Waitsome status\n";
                for(int c=0;c<count;c++)
                {
                  if(statuses_[c].MPI_ERROR!=MPI_SUCCESS)
                  {
                    char message[MPI_MAX_ERROR_STRING];
                    int length;
                    
                    MPI_Error_string(statuses_[c].MPI_ERROR,message,&length);
                    cerr << "Error message" << ": '" << message << "'\n";
                  
                    task=request_to_task_[indicies_[c]];
                    cerr << "Task status:" << task->status_ << " patch:" << task->patch_ << endl;
                  }
                }
                cerr << "Entering infinite loop so debugger can be attached\n";
                while(1); //hang so debugger can be attached
        }
      
        //handle each request
        for(int c=0;c<count;c++)
        {
          BNRTask *task=request_to_task_[indicies_[c]];
          free_requests_.push(indicies_[c]);
          if(--(task->remaining_requests_)==0)  //task has completed communication
          {
            if(task->status_!=TERMINATED)     //if task needs more work
            {
              immediate_q_.push(task);        //place it on the immediate_q 
            }
          }
        }
      }
      else if(tag_q_.empty())  //no tasks remaining, no communication waiting, algorithm is done
      {
        break; 
      }
      else
      {
        //no tasks on the immediate_q, tasks are on the taq_q
        if(tags_.size() + free_tag_end_ - free_tag_start_ < 2) //this if might not be needed 
        {
          cout << "Error not enough tags\n";
          exit(0);
        }
      }
    }
 
    //check for controlling processors 
    if(rank==root->p_group_[0])
    {
      //assign the patches to my list
      patches.assign(root->my_patches_.begin(),root->my_patches_.end());
    }
  }
  
  if(numprocs>1)
  {
    //communicate the patchset to rank 0 for broadcasting
    if(root!=0 && rank==root->p_group_[0] && rank!=0) //if I am the root and not rank 0
    {
      int size=patches.size();
      //send to rank 0
      MPI_Send(&size,1,MPI_INT,0,0,MPI_COMM_WORLD);
      MPI_Send(&patches[0],size*sizeof(Region),MPI_BYTE,0,0,MPI_COMM_WORLD);
    }
    else if(rank==0 && (root==0 || rank!=root->p_group_[0])) //if I am rank 0 and not the root
    {
      MPI_Status status;
      int size;
      //receive from any rank
      MPI_Recv(&size,1,MPI_INT,MPI_ANY_SOURCE,0,MPI_COMM_WORLD,&status);
      patches.resize(size);
      MPI_Recv(&patches[0],size*sizeof(Region),MPI_BYTE,MPI_ANY_SOURCE,0,MPI_COMM_WORLD,&status);
    }
          
    //broadcast size out
    unsigned int size=patches.size();
    MPI_Bcast(&size,1,MPI_INT,0,MPI_COMM_WORLD);
    //resize patchlist
    patches.resize(size);
    //broadcast patches
    MPI_Bcast(&patches[0],size*sizeof(Region),MPI_BYTE,0,MPI_COMM_WORLD);
  }
 
  tasks_.clear();
}

} // End namespace Uintah

