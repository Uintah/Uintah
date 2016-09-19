#include <Core/Geometry/IntVector.h>
#include <Core/Geometry/Vector.h>
#include <Core/Grid/Variables/CellIterator.h>
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Patch.h>

#include "mpi.h"

#include <testprograms/Regridders/TiledRegridder.h>
#include <testprograms/Regridders/common.h>

#include <iostream>
#include <ostream>
#include <fstream>

using namespace Uintah;

int
main(int argc, char **argv)
{
  Uintah::MPI::Init(&argc, &argv);

  Uintah::MPI::Comm_size(MPI_COMM_WORLD, &num_procs);
  Uintah::MPI::Comm_rank(MPI_COMM_WORLD, &rank);

  if (argc != 5) {
    if (rank == 0) {
      std::cout << "Usage: benchmark patch_size patches_in_each_dim flag_inner_rad(0-1) flag_outter_rad(0-1)\n";
    }
    Uintah::MPI::Finalize();
    return 1;
  }

  //get command line arguements
  IntVector patch_size;
  patch_size[0] = patch_size[1] = patch_size[2] = atoi(argv[1]);
  IntVector num_patches;
  num_patches[0] = num_patches[1] = num_patches[2] = atoi(argv[2]);
  IntVector cells;
  cells[0] = cells[1] = cells[2] = num_patches[0] * patch_size[0];
  IntVector rr(4, 4, 4);

#if 1
  patch_size[2] = 1;
  num_patches[2] = 1;
  cells[2] = 1;
  rr[2] = 1;
#endif 

  double radin = cells[0] / 2.0 * atof(argv[3]);
  double radout = cells[0] / 2.0 * atof(argv[4]);

  Sphere2 s(cells.asVector() / Vector(2, 2, 2), radin, radout);

//  std::cout << "num patches:" << num_patches << std::endl;
//  std::cout << "cells: " << cells << std::endl;
//  std::cout << "rad: " << rad << std::endl;

//create coarse patch set
  std::vector<Region> patches;
  std::vector<CCVariable<int> *> flags;
  std::vector<IntVector> gflags;
  std::vector<std::list<IntVector> > lflags;

  int total_patches = num_patches.x() * num_patches.y() * num_patches.z();
  int div = total_patches / num_procs;
  int mod = total_patches % num_procs;

//  std::cout << "total patches: " << total_patches << std::endl;
//  std::cout << "div: " << div << " mod: " << mod << std::endl;
  int p = 0;
  int p_assigned = 0;
  int to_assign = div + int(mod > 0);
  int idx = 0;
//  std::cout << "to_assign=" << to_assign <<std:: endl;

  for (int i = 0; i < num_patches.x(); i++) {
    for (int j = 0; j < num_patches.y(); j++) {
      for (int k = 0; k < num_patches.z(); k++) {
        IntVector low = IntVector(i, j, k) * patch_size;
        IntVector high = low + patch_size;
        if (p == rank) {
          patches.push_back(Region(low, high));
        }

        p_assigned++;

        if (p_assigned == to_assign) {
//          if(rank==0) {
//            cout << p << " assigned: " << to_assign << " patches\n";
//          }
          p++;
          p_assigned = 0;
          to_assign = div + int(mod > p);
        }
        idx++;
      }
    }
  }

//  for(unsigned int i=0;i<patches.size();i++) {
//    cout << rank << " patch: " << patches[i] << endl;
//  }

  //create refinement flags
  flags.resize(patches.size());
  lflags.resize(patches.size());

  //for each local patch
  for (unsigned int patch = 0; patch < patches.size(); patch++) {
    //allocate the variable
    CCVariable<int> *f = new CCVariable<int>();

    //allocate the memory in the variable
    f->allocate(patches[patch].getLow(), patches[patch].getHigh());
    flags[patch] = f;

    //determine flag set
    for (CellIterator iter(patches[patch].getLow(), patches[patch].getHigh()); !iter.done(); iter++) {
      if (s.intersects((*iter).asVector() + Vector(.5, .5, .5))) {
        //add to CC variable flags
        (*f)[*iter] = 1;
        //add to per patch flags list
        lflags[patch].push_back(*iter);
        //add to per processor flags list
        gflags.push_back(*iter);
      }
      else {
        (*f)[*iter] = 0;
      }
    }
  }

  int num_flags = getTotalNumFlags(gflags);
  int fflags = num_flags * rr[0] * rr[1] * rr[2];

  std::ofstream fout;
  fout.open("flags");
  outputFlags(gflags, fout);
  fout.close();
  fout.open("cpatches");
  outputPatches(patches, fout);
  fout.close();

  std::vector<Region> fine_patches, global_patches;
  TiledRegridder tiled(patch_size, rr);
  long long vol;

#if 1
  Uintah::MPI::Barrier(MPI_COMM_WORLD );
  tiled.regrid(patches, flags, fine_patches);
  gatherPatches(fine_patches, global_patches);
  if (rank == 0) {
    fout.open("tiledpatches");
    outputPatches(global_patches, fout);
    fout.close();

    vol = 0;
    for (size_t i = 0; i < global_patches.size(); i++) {
      vol += global_patches[i].getVolume();
    }
    std::cout << "Tiled number of patches: " << global_patches.size() << " number of cells: "
              << vol << " Flags: " << fflags << std::endl;
  }
#endif

  Uintah::MPI::Finalize();
}

