#include <sci_defs/mpi_defs.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#include <unistd.h>
using namespace std;

#define CSTYLE

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
  MPI_Barrier(MPI_COMM_WORLD);
  start=MPI_Wtime();
#ifdef CSTYLE
  fwrite(buff,sizeof(char),isize,fout);
  fflush(fout);
  fclose(fout);
#else
  fout.write(buff,isize);
  fout.flush();
  fout.close();
#endif
  MPI_Barrier(MPI_COMM_WORLD);
  finish=MPI_Wtime();
  
  char command[100];
  sprintf(command,"rm -f %s",filename);
  
  delete buff;
  if(rank==0)
  {
    cout << "Writing Total Time: " << finish-start << " seconds" << endl;
    cout << "Writing Throughput: " <<  (isize*processors/1048576.0)/(finish-start) << " MB/s" << endl;
  
    cout << "Cleaning up datafiles\n";
  }
  MPI_Barrier(MPI_COMM_WORLD);
  start=MPI_Wtime();
  system(command);
  MPI_Barrier(MPI_COMM_WORLD);
  finish=MPI_Wtime();
  if(rank==0)
  {
    cout << "Deleting Total Time: " << finish-start << " seconds" << endl;
    cout << "Deleting Throughput: " <<  (isize*processors/1048576.0)/(finish-start) << " MB/s" << endl;
  }
  MPI_Finalize();

  return 0;
}
