
#include <iostream>
#include <ostream>
#include <fstream>
using namespace std;

#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/Vector.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Patch.h>

using namespace SCIRun;
using namespace Uintah;

#include "mpi.h"
#include <testprograms/Regridders/TiledRegridder.h>
#include <testprograms/Regridders/BNRRegridder.h>
#include <testprograms/Regridders/LBNRRegridder.h>
#include <testprograms/Regridders/GBRv1Regridder.h>
#include <testprograms/Regridders/common.h>

int main(int argc, char **argv) 
{
  MPI_Init(&argc,&argv);

  MPI_Comm_size(MPI_COMM_WORLD,&num_procs);
  MPI_Comm_rank(MPI_COMM_WORLD,&rank);

  if(argc!=5)
  {
    if(rank==0)
      cout << "Usage: benchmark patch_size max_patches flag_inner_rad(0-1) flag_outter_rad(0-1)\n";
    MPI_Finalize();
    return 1;
  }
    
  IntVector patch_size;
  patch_size[0]=patch_size[1]=patch_size[2]=atoi(argv[1]);
  int max_p=atoi(argv[2]);
  IntVector rr(4,4,4);

 
  for(int np=6;np<max_p;np++)
  {
    //get command line arguements
    IntVector num_patches;
    num_patches[0]=num_patches[1]=num_patches[2]=np;
    IntVector cells;
    cells[0]=cells[1]=cells[2]=num_patches[0]*patch_size[0];
    double radin=cells[0]/2.0*atof(argv[3]);
    double radout=cells[0]/2.0*atof(argv[4]);

#if 0
    num_patches[2]=1;
    cells[2]=1;
    rr[2]=1;
#endif 


    Sphere2 s(cells.asVector()/Vector(2,2,2),radin,radout);

    //cout << "num patches:" << num_patches << endl;
    //cout << "cells: " << cells << endl;
    //cout << "rad: " << rad << endl;

    //create coarse patch set
    vector<Region> patches;
    vector<CCVariable<int> * > flags;
    vector<IntVector> gflags;
    vector<list<IntVector> > lflags;

    int total_patches=num_patches.x()*num_patches.y()*num_patches.z();
    int div=total_patches/num_procs;
    int mod=total_patches%num_procs;

    //cout << "total patches: " << total_patches << endl;
    //cout << "div: " << div << " mod: " << mod << endl;
    int p=0;
    int p_assigned=0;
    int to_assign=div+int(mod>0);
    int idx=0;
    //cout << "to_assign=" << to_assign << endl;

    for(int i=0;i<num_patches.x();i++)
    {
      for(int j=0;j<num_patches.y();j++)
      {
        for(int k=0;k<num_patches.z();k++)
        {
          IntVector low=IntVector(i,j,k)*patch_size;
          IntVector high=low+patch_size;

          if(p==rank)
          {
            patches.push_back(Region(low,high));
          }

          p_assigned++;

          if(p_assigned==to_assign)
          { 
            //if(rank==0)
            //  cout << p << " assigned: " << to_assign << " patches\n"; 
            p++;
            p_assigned=0;
            to_assign=div+int(mod>p);
          }

          idx++;
        }
      }
    }

    //for(unsigned int i=0;i<patches.size();i++)
    //  cout << rank << " patch: " << patches[i] << endl;

    //create refinement flags
    flags.resize(patches.size());
    lflags.resize(patches.size());

    //for each local patch
    for(unsigned int patch=0;patch<patches.size();patch++)
    {
      //allocate the variable
      CCVariable<int> *f=new CCVariable<int>();

      //allocate the memory in the variable
      f->allocate(patches[patch].getLow(),patches[patch].getHigh());
      flags[patch]=f;

      //determine flag set
      for(CellIterator iter(patches[patch].getLow(),patches[patch].getHigh());!iter.done();iter++)
      {
        if(s.intersects( (*iter).asVector()+Vector(.5,.5,.5) ))
        {
          //add to CC variable flags
          (*f)[*iter]=1;
          //add to per patch flags list
          lflags[patch].push_back(*iter);
          //add to per processor flags list
          gflags.push_back(*iter);
        }
        else
          (*f)[*iter]=0;
      }
    }
    ofstream fout;
#if 0
    fout.open("flags");
    outputFlags(gflags,fout);
    fout.close();
    fout.open("cpatches");
    outputPatches(patches,fout);
    fout.close();
#endif

    long long fflags=getTotalNumFlags(gflags)*rr[0]*rr[1]*rr[2];


    vector<Region> fine_patches,global_patches;

    TiledRegridder tiled(patch_size,rr);
    BNRRegridder bnr(.85,rr);
    LBNRRegridder lbnr(.85,rr);
    GBRv1Regridder gbrv1(.85,rr,rank,num_procs);
    long long vol;
#if 1
    tiled.regrid(patches,flags,fine_patches);
    gatherPatches(fine_patches,global_patches);
    if(rank==0)
    {
      //fout.open("tiledpatches");
      //outputPatches(global_patches,fout);
      //fout.close();
      vol=0;
      for(size_t i=0;i<global_patches.size();i++)
        vol+=global_patches[i].getVolume();
      cout << "0 " << cells[0]*cells[1]*cells[2] << " " << global_patches.size() <<  " " << vol << " " << fflags << endl;
      if(vol<fflags)
      {
        cout << "Error\n";
        MPI_Abort(MPI_COMM_WORLD,0);
      }
    }
#endif

#if 0  //only works in serial
    bnr.regrid(gflags,fine_patches);
    if(rank==0)
    {
      //fout.open("bnrpatches");
      //outputPatches(fine_patches,fout);
      //fout.close();
      vol=0;
      for(size_t i=0;i<fine_patches.size();i++)
        vol+=fine_patches[i].getVolume();
      cout << "1 " << cells[0]*cells[1]*cells[2] << " " << fine_patches.size() <<  " " << vol << " " << fflags << endl;
    }
#endif

#if 1
    gbrv1.regrid(gflags,fine_patches);
    if(rank==0)
    {
      //fout.open("gbrv1patches");
      //outputPatches(fine_patches,fout);
      //fout.close();
      vol=0;
      for(size_t i=0;i<fine_patches.size();i++)
        vol+=fine_patches[i].getVolume();
      cout << "1 " << cells[0]*cells[1]*cells[2] << " " << fine_patches.size() <<  " " << vol << " " << fflags << endl;
    }
#endif

#if 1
    lbnr.regrid(lflags,fine_patches);
    gatherPatches(fine_patches,global_patches);
    if(rank==0)
    {
      //fout.open("lbnrpatches");
      //outputPatches(global_paches,fout);
      //fout.close();
      vol=0;
      for(size_t i=0;i<global_patches.size();i++)
        vol+=global_patches[i].getVolume();
      cout << "2 " << cells[0]*cells[1]*cells[2] << " " << global_patches.size() <<  " " << vol << " " << fflags << endl;
    }
#endif
  }
  MPI_Finalize();
}

