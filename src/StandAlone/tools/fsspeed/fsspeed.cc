/*
 * The MIT License
 *
 * Copyright (c) 1997-2020 The University of Utah
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

#include <iostream>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <unistd.h>

using namespace std;

#define CSTYLE

int main(int argc,char *argv[])
{
  Uintah::MPI::Init(&argc,&argv);
  int rank,processors;
  Uintah::MPI::Comm_rank(MPI_COMM_WORLD,&rank);
  Uintah::MPI::Comm_size(MPI_COMM_WORLD,&processors);

  if(argc!=2)
  {
    if(rank==0)
    {
      cout << "Command Line Example: mpirun -np X fsspeed 16GB\n";
      cout << "acceptable file sizes include B (bytes), MB (megabytes), GB (gigabytes)\n";
    }
    Uintah::MPI::Finalize();
    return 1;
  }
  stringstream str;
 
  //write argument into stringstream
  str << argv[1];

  double size;
  string type;

  //read in size and type
  str >> size;
  str >> type;

  size/=processors;

  if(type=="GB")
  {
    size*=1073741824;
  }
  else if(type=="MB")
  {
    size*=1048576;
  }
  else if(type!="B")
  {
    if(rank==0)
    {
      cout << "Error invalid size type\n";
      cout << "Command Line Example: mpirun -np X fsspeed 16GB\n";
      cout << "acceptable file sizes include bytes (B), megabytes (MB), gigabytes (GB)\n";
    }
    Uintah::MPI::Finalize();
    return 1;
  }
  
  long long isize=(long long)size;
  char *buff=new char[isize];
  for(int i=0;i<isize;i++)
    buff[i]=0;

  char filename[100];
  sprintf(filename,".tmpfile.%d",rank);

#ifdef CSTYLE
   FILE* fout=fopen(filename,"w");
#else
  ofstream fout(filename,ios::binary);
#endif
  double start,finish;

  if(rank==0)
  {
    cout << "Writing " << isize*processors/1048576.0 << " MB" << endl;
  }
  Uintah::MPI::Barrier(MPI_COMM_WORLD);
  start=Uintah::MPI::Wtime();
#ifdef CSTYLE
  fwrite(buff,sizeof(char),isize,fout);
  fflush(fout);
  fclose(fout);
#else
  fout.write(buff,isize);
  fout.flush();
  fout.close();
#endif
  Uintah::MPI::Barrier(MPI_COMM_WORLD);
  finish=Uintah::MPI::Wtime();
  
  char command[100];
  sprintf(command,"rm -f %s",filename);
  
  delete [] buff;
  if( rank == 0 ) {
    cout << "Writing Total Time: " << finish-start << " seconds" << endl;
    cout << "Writing Throughput: " <<  (isize*processors/1048576.0)/(finish-start) << " MB/s" << endl;
  
    cout << "Cleaning up datafiles\n";
  }
  Uintah::MPI::Barrier(MPI_COMM_WORLD);
  start=Uintah::MPI::Wtime();
  system(command);
  Uintah::MPI::Barrier(MPI_COMM_WORLD);
  finish=Uintah::MPI::Wtime();
  if( rank == 0 ) {
    cout << "Deleting Total Time: " << finish-start << " seconds" << endl;
    cout << "Deleting Throughput: " <<  (isize*processors/1048576.0)/(finish-start) << " MB/s" << endl;
  }
  Uintah::MPI::Finalize();

  return 0;
}
