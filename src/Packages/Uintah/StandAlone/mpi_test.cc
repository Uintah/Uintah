#include "mpi.h"
#include <vector>
#include <iostream>
#include <string>
#include <sstream>

using namespace std;
	
int rank;
int procs;

stringstream error_stream;

int allreduce_test();
int reduce_test();
int broadcast_test();
int allgather_test();
int gather_test();
int point2pointasync_test();
int point2pointsync_test();

int testme(int (*testfunc)(void),char* name)
{

  if(rank==0)
  {
    cout << "Testing '" << name << "': ";
    cout.flush();
  }

  int pass=testfunc();
  int all_pass=false;
 
  MPI_Allreduce(&pass,&all_pass,1,MPI_INT,MPI_LOR,MPI_COMM_WORLD);
 
  if(rank==0)
  {
    if(all_pass)
      cout << "Passed\n" ;
    else
      cout << "Failed\n" ;
            
  }
  
  if(!all_pass)
  {
    //sync processors so output is in sync
    MPI_Barrier(MPI_COMM_WORLD);
    cout << error_stream.str();
    cout.flush();
    error_stream.str("");
    //sync processors so output is in sync
    MPI_Barrier(MPI_COMM_WORLD);
  }
  return all_pass;
        
}

int main(int argc, char** argv)
{
	MPI_Init(&argc,&argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &procs);

  if(rank==0)
  {
    cout << "Testing mpi communication on " << procs << " processors." << endl;
  }
 
  int pass;
 
  testme(allreduce_test,"MPI_Allreduce");
  testme(reduce_test,"MPI_Reduce");
  testme(broadcast_test,"MPI_Bcast");
  testme(allgather_test,"MPI_Allgather");
  testme(gather_test,"MPI_Gather");
  testme(point2pointasync_test,"PointToPointAsync");
  testme(point2pointsync_test,"PointToPointSync");
  
	MPI_Finalize();
	return 0;
}
//Each Processor all reduces their rank
int allreduce_test()
{
  int pass=true;
  int message;
  int n=procs-1;
  MPI_Allreduce(&rank,&message,1,MPI_INT,MPI_SUM,MPI_COMM_WORLD);

  if(message!=(n*(n+1))/2)
  {
    pass=false;
    error_stream << "     rank " << rank << ": Allreduce incorrect\n";
    //cout << "     rank " << rank << ": Allreduce incorrect\n";
  }
  return pass; 
}
//Each processor broadcasts their rank
int broadcast_test()
{
  int pass=true;
  int message;
  for(int p=0;p<procs;p++)
  {
    message=rank;
    MPI_Bcast(&message,1,MPI_INT,p,MPI_COMM_WORLD);
    if(message!=p)
    {
      pass=false;
      error_stream << "     rank " << rank << ": Bcast from rank " << p << " incorrect\n";
    }
  }
  return pass;
}

//Each processor reduce sums their rank
int reduce_test()
{
  int pass=true;
  int message;
  int n=procs-1;
  for(int p=0;p<procs;p++)
  {
    message=rank;
    MPI_Reduce(&rank,&message,1,MPI_INT,MPI_SUM,p,MPI_COMM_WORLD);
    
    if(p==rank && message!=(n*(n+1))/2)
    {
      pass=false;
      error_stream << "     rank " << rank << ": Reduce on rank " << p << " incorrect\n";
    }
  }
  return pass;
}

//Each Processor allgathers its rank
int allgather_test()
{
  int pass=true;
  vector<int> message(procs,0);
  
  MPI_Allgather(&rank,1,MPI_INT,&message[0],1,MPI_INT,MPI_COMM_WORLD);

  for(int p=0;p<procs;p++)
  {
    if(message[p]!=p)
    {
      pass=false;
      error_stream << "     rank " << rank << ": Allgather entry from " << p << " is invalid\n";
    }
  }
  return pass; 
}

//Each Processor gathers its rank
int gather_test()
{
  int pass=true;
  
  for(int p=0;p<procs;p++)
  {
    vector<int> message(procs,0);
  
    MPI_Gather(&rank,1,MPI_INT,&message[0],1,MPI_INT,p,MPI_COMM_WORLD);

    if(rank==p)
    {
      for(int p=0;p<procs;p++)
      {
        if(message[p]!=p)
        {
          pass=false;
          error_stream << "     rank " << rank << ": gather entry from " << p << " is invalid\n";
        }
      }
    }
  }
  return pass; 
}

//Each Processor sends its rank to each other processor
int point2pointasync_test()
{
  int pass=true;
  vector<int> messages(procs);
  vector<MPI_Request> srequest(procs),rrequest(procs);
  
  for(int p=0;p<procs;p++)
  {
    //start send
    MPI_Isend(&rank,1,MPI_INT,p,p,MPI_COMM_WORLD,&srequest[p]);
    
    //start recv
    MPI_Irecv(&messages[p],1,MPI_INT,p,rank,MPI_COMM_WORLD,&rrequest[p]);
  }
  
  //waitsome loop
  for(int p=0;p<procs;p++)
  {
    MPI_Status status;
    MPI_Wait(&srequest[p],&status);

    MPI_Wait(&rrequest[p],&status);
    if(messages[p]!=p)
    {
      pass=false;
      error_stream << "     rank " << rank << ": point to point async from  " << p << " is invalid\n";
     }
  }
  return pass;
}

//Each Processor gathers its rank
  //this should probably be improved to work in parallel but right now it is done sequentially
int point2pointsync_test()
{
  int pass=true;
  int message;
  for(int pp=0;pp<procs;pp++)
  {
    if(pp==rank) //sending
    {
      for(int p=0;p<procs;p++)
      {
        MPI_Send(&rank,1,MPI_INT,p,p,MPI_COMM_WORLD);
      }
    }
    else //recieving
    {
        MPI_Status status;
        message=-1;
        MPI_Recv(&message,1,MPI_INT,pp,rank,MPI_COMM_WORLD,&status);
        if(message!=pp)
        {
          pass=false;
          error_stream << "     rank " << rank << ": point to point sync from  " << pp << " is invalid\n";
        }
    }
  }

  return pass;
}
