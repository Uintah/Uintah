#include <sci_defs/mpi_defs.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdlib.h>
using namespace std;

int main(int argc,char *argv[])
{
  MPI_Init(&argc,&argv);
  int rank,processors;
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);
  MPI_Comm_size(MPI_COMM_WORLD,&processors);

  if(argc!=2)
  {
    if(rank==0)
    {
      cout << "Command Line Example: mpirun -np X fsspeed 16GB\n";
      cout << "acceptable file sizes include B (bytes), MB (megabytes), GB (gigabytes)\n";
    }
    MPI_Finalize();
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
    MPI_Finalize();
    return 1;
  }
  
  long long isize=size;
  char *buff=new char[isize];

  char filename[100];
  sprintf(filename,".tmpfile.%d",rank);

  ofstream fout(filename,ios::binary);
  double start,finish;

  if(rank==0)
  {
    cout << "Writing " << isize*processors/1048576.0 << " MB" << endl;
  }
  MPI_Barrier(MPI_COMM_WORLD);
  start=MPI_Wtime();
  fout.write(buff,isize);
  MPI_Barrier(MPI_COMM_WORLD);
  finish=MPI_Wtime();
  
  char command[100];
  sprintf(command,"rm -f %s",filename);


  delete buff;
  if(rank==0)
  {
    cout << "Total Time: " << finish-start << " seconds" << endl;
    cout << "Throughput: " <<  (isize*processors/1048576.0)/(finish-start) << " MB/s" << endl;
  
    cout << "Cleaning up datafiles\n";
  }
  system(command);
  MPI_Finalize();
  return 0;
}
