/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
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

#include <Core/Parallel/UintahMPI.h>

#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <vector>
#include <queue>
#include <unistd.h>
#include <sys/time.h>

#define debug_main
#define debug_main_thread
#define debug_mpi_thread

using namespace std;

int message_size = 5*1024*1024;

void do_some_work(int myid);

int
main(int argc, char** argv)
{
  int thread_supported = 0;
  char *send_buf; 
  char *send_buf2;
  char *recv_buf; 
  char *recv_buf2;
  int myid = 99;
  int dest = 0;
  int numprocs = 99;
  int tag = 1;
  
  Uintah::MPI::Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &thread_supported);
#ifdef debug_main
  cout<<"Thread supported is "<<thread_supported<<endl;
#endif
  Uintah::MPI::Comm_size(MPI_COMM_WORLD, &numprocs);
  Uintah::MPI::Comm_rank(MPI_COMM_WORLD, &myid);
  
  srand(myid*10);

  do_some_work(myid);

  if (myid == 0){
    dest = 1;
  }
  else {
    dest = 0;
  }

  send_buf = new char[message_size];
  send_buf2 = new char[message_size];
  recv_buf = new char[message_size];
  recv_buf2 = new char[message_size];
  MPI_Request rq1;
  MPI_Request rq2;
  MPI_Request rq3;
  MPI_Status st1;
  MPI_Status st2;
  MPI_Status st3;
  
  if (myid == 1){
    sprintf((char*)send_buf, "this a message sent from myid1, signed Bruce R. Kanobi");
    Uintah::MPI::Isend(send_buf, message_size, MPI_CHAR, dest, tag, MPI_COMM_WORLD, &rq1);
    sprintf((char*)send_buf2, "this a 2nd message sent from myid1, signed Bruce R. Kanobi");
    Uintah::MPI::Isend(send_buf2, message_size, MPI_CHAR, dest, tag+5, MPI_COMM_WORLD, &rq2);
  }
  else{
    Uintah::MPI::Recv(recv_buf, message_size, MPI_CHAR, dest, tag, MPI_COMM_WORLD, &st1);
    cout<<"0 Got message "<<recv_buf<<endl;
  }

  do_some_work(myid);

  if (myid == 1){
    Uintah::MPI::Recv(recv_buf, message_size, MPI_CHAR, dest, tag, MPI_COMM_WORLD,&st2);
    cout<<"1 Got message "<<recv_buf<<endl;
  }
  else{
    sprintf(send_buf, "this a message sent from myid0, signed Thomas S. Duku");
    Uintah::MPI::Isend(send_buf, message_size, MPI_CHAR, dest, tag, MPI_COMM_WORLD, &rq3);

    Uintah::MPI::Recv(recv_buf2, message_size, MPI_CHAR, dest, tag+5, MPI_COMM_WORLD,&st3);
    cout<<"0 Got message "<<recv_buf2<<endl;
  }
  
  do_some_work(myid);

  //MPI_Status status2[1];
  //int* probe_flag = new int(0);
  //Uintah::MPI::Iprobe(MPI_ANY_SOURCE,MPI_ANY_TAG,MPI_COMM_WORLD,probe_flag,status2);
  //if (*probe_flag){
  //  cout<<"myid"<<myid<<" has outstading communications"<<endl;
  //}
#ifdef debug_main
  //cout<<"myid"<<myid<<" mpiCallQueue"<<MPICommObj.mpiCallQueue.size()<<endl;
#endif
  Uintah::MPI::Finalize();

  return 0;
}

void
do_some_work(int myid)
{
  const int sleep_time_constant = 5000000;
  //const int sleep_time_constant = 1000000;
  int sleep_time_total = 0;
  sleep_time_total = sleep_time_constant + (rand()%1000000);
  cout<<myid<<" is sleeping for "<<sleep_time_total<<endl;
  usleep(sleep_time_total);
}

