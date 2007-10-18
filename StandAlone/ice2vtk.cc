/*
 *  ice2vtk.cc: Print out a uintah data archive
 *
 *  Written by:
 *   Martin Denison
 *   Reaction Engineering International
 *   July 2006
 *   Jim Guilkey
 *   Department of Mechancial Engineering 
 *   by stealing timeextract from:
 *   James L. Bigler
 *   Bryan J. Worthen
 *   Department of Computer Science
 *   University of Utah
 *   June 2004
 *
 *  Copyright (C) 2004 U of U
 */

#include <Packages/Uintah/Core/DataArchive/DataArchive.h>
#include <Packages/Uintah/Core/Disclosure/TypeDescription.h>
#include <Packages/Uintah/Core/Math/Matrix3.h>
#include <Packages/Uintah/Core/Grid/Box.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/Level.h>
#include <Packages/Uintah/Core/Grid/Variables/NodeIterator.h>
#include <Packages/Uintah/Core/Grid/Variables/CellIterator.h>
#include <Packages/Uintah/Core/Grid/Variables/ShareAssignParticleVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCXVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCYVariable.h>
#include <Packages/Uintah/Core/Grid/Variables/SFCZVariable.h>

#include <Core/Math/MinMax.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/OS/Dir.h>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <algorithm>

#include <stdio.h>

using namespace SCIRun;
using namespace std;
using namespace Uintah;

bool verbose = false;
bool quiet = false;
bool d_printCell_coords = false;
  
void
usage(const std::string& badarg, const std::string& progname)
{
    if(badarg != "")
        cerr << "Error parsing argument: " << badarg << endl;
    cerr << "Usage: " << progname << " [options] "
         << "-uda <archive file>\n\n";
    cerr << "Valid options are:\n";
    cerr << "  -h,--help\n";
    cerr << "  -v,--variable <variable name>\n";
    cerr << "  -m,--material <material number> [defaults to 0]\n";
    cerr << "  -tlow,--timesteplow [int] (sets start output timestep to int) [defaults to 0]\n";
    cerr << "  -thigh,--timestephigh [int] (sets end output timestep to int) [defaults to last timestep]\n";
    cerr << "  -timestep,--timestep [int] (only outputs from timestep int) [defaults to 0]\n";
    cerr << "  -istart,--indexs <x> <y> <z> (cell index) [defaults to 0,0,0]\n";
    cerr << "  -iend,--indexe <x> <y> <z> (cell index) [defaults to 0,0,0]\n";
    cerr << "  -l,--level [int] (level index to query range from) [defaults to 0]\n";
    cerr << "  -o,--out <outputfilename> [defaults to stdout]\n"; 
    cerr << "  -vv,--verbose (prints status of output)\n";
    cerr << "  -q,--quiet (only print data values)\n";
    cerr << "  -cellCoords (prints the cell centered coordinates on that level)\n";
    cerr << "  --cellIndexFile <filename> (file that contains a list of cell indices)\n";
    cerr << "                                   [int 100, 43, 0]\n";
    cerr << "                                   [int 101, 43, 0]\n";
    cerr << "                                   [int 102, 44, 0]\n";
    exit(1);
}
///////////////////
void swap_4(char* data)
{
  char b;
  b = data[0]; data[0] = data[3]; data[3] = b;
  b = data[1]; data[1] = data[2]; data[2] = b;
}

// arguments are the dataarchive, the successive arguments are the same as 
// the arguments to archive->query for data values.  Then comes a type 
// dexcription of the variable being queried, and last is an output stream.
//______________________________________________________________________
//
template<class T>
void printData(DataArchive* archive, string& variable_name, const Uintah::TypeDescription* variable_type,
               int material, int levelIndex,
               IntVector& var_start, IntVector& var_end, int pbegin, int pend,
               unsigned long time_step, vector<vector<vector<T> > >& phi) 

{
  // query time info from dataarchive
  vector<int> index;
  vector<double> times;

  archive->queryTimesteps(index, times);
  ASSERTEQ(index.size(), times.size());
    
  bool cellNotFound = false;
  //__________________________________
  // loop over timesteps
  //for (unsigned long time_step = time_start; time_step <= time_end; time_step++) {
  

    //__________________________________
    //  does the requested level exist
    bool levelExists = false;
    GridP grid = archive->queryGrid(times[time_step]); 
    int numLevels = grid->numLevels();
   
    for (int L = 0;L < numLevels; L++) {
      const LevelP level = grid->getLevel(L);
      if (level->getIndex() == levelIndex){
        levelExists = true;
      }
    }
    if (!levelExists){
      cerr<< " Level " << levelIndex << " does not exist at this timestep " << time_step << endl;
    }

    int i, j, k, ni = var_end.x()+1, nj = var_end.y()+1, nk = var_end.z()+1;
    
    if(levelExists){   // only extract data if the level exists
      const LevelP level = grid->getLevel(levelIndex);
      //__________________________________
      // User input starting and ending indicies    
          
        // find the corresponding patches
        Level::selectType patches;
	level->selectPatches(IntVector(0,0,0), IntVector(1001,1001,1001), patches);
        if( patches.size() == 0){
          cerr << " Could not find any patches on Level " << level->getIndex()
               << " Double check the starting and ending indices "<< endl;
          exit(1);
        }

        // query all the data up front
        vector<Variable*> vars(patches.size());
        for (int p = pbegin; p < pend; p++) {
          switch (variable_type->getType()) {
          case Uintah::TypeDescription::CCVariable:
            vars[p] = scinew CCVariable<T>;
            archive->query( *(CCVariable<T>*)vars[p], variable_name, 
                            material, patches[p], times[time_step]);
            break;
          case Uintah::TypeDescription::NCVariable:
            vars[p] = scinew NCVariable<T>;
            archive->query( *(NCVariable<T>*)vars[p], variable_name, 
                            material, patches[p], times[time_step]);
            break;
          case Uintah::TypeDescription::SFCXVariable:
            vars[p] = scinew SFCXVariable<T>;
            archive->query( *(SFCXVariable<T>*)vars[p], variable_name, 
                            material, patches[p], times[time_step]);
            break;
          case Uintah::TypeDescription::SFCYVariable:
            vars[p] = scinew SFCYVariable<T>;
            archive->query( *(SFCYVariable<T>*)vars[p], variable_name, 
                            material, patches[p], times[time_step]);
            break;
          case Uintah::TypeDescription::SFCZVariable:
            vars[p] = scinew SFCZVariable<T>;
            archive->query( *(SFCZVariable<T>*)vars[p], variable_name, 
                            material, patches[p], times[time_step]);
            break;
          default:
            cerr << "Unknown variable type: " << variable_type->getName() << endl;
          }
          
        }
        if(levelIndex>0){
	  var_start = patches[pbegin]->getLowIndex();// + IntVector(1,1,1);
	  var_end = patches[pbegin]->getHighIndex();// - IntVector(1,1,1);
	  //	  if(var_end.z()==2) var_end = var_end - IntVector(0,0,1);
	}

        for (CellIterator ci(var_start, var_end); !ci.done(); ci++) {
          IntVector c = *ci;

          // find out which patch it's on (to keep the printing in sorted order.
          // alternatively, we could just iterate through the patches)
          IntVector low, high;
          int p = pbegin;
          for (; p < pend; p++) {
            low = patches[p]->getLowIndex();
            high = patches[p]->getHighIndex();
            if (c.x() >= low.x() && c.y() >= low.y() && c.z() >= low.z() && 
                c.x() < high.x() && c.y() < high.y() && c.z() < high.z())
              break;
          }
          if (p == pend) {
            cellNotFound = true;
            continue;
          }
          
          T val;
          Vector dx = patches[p]->dCell();
          Vector shift(0,0,0);  // shift the cellPosition if it's a (X,Y,Z)FC variable
          switch (variable_type->getType()) {
          case Uintah::TypeDescription::CCVariable: 
            val = (*dynamic_cast<CCVariable<T>*>(vars[p]))[c]; 
          break;
          case Uintah::TypeDescription::NCVariable: 
            val = (*dynamic_cast<NCVariable<T>*>(vars[p]))[c]; 
          break;
          case Uintah::TypeDescription::SFCXVariable: 
            val = (*dynamic_cast<SFCXVariable<T>*>(vars[p]))[c];
            shift.x(-dx.x()/2.0); 
          break;
          case Uintah::TypeDescription::SFCYVariable: 
            val = (*dynamic_cast<SFCYVariable<T>*>(vars[p]))[c];
            shift.y(-dx.y()/2.0); 
          break;
          case Uintah::TypeDescription::SFCZVariable: 
            val = (*dynamic_cast<SFCZVariable<T>*>(vars[p]))[c];
            shift.z(-dx.z()/2.0); 
          break;
          default: break;
          }
          if(levelIndex==0){
	    i = c.x();
	    j = c.y();
	    k = c.z();
	  }else{
	    i = c.x() - var_start.x();
	    j = c.y() - var_start.y();
	    k = c.z() - var_start.z();
	  }
          phi[i][j][k] = val;
        }
        for (unsigned m = pbegin; m < pend; m++)
          delete vars[m];

    } // if level exists
    
} 
//______________________________________________________________________
//
template<class T>
int maxIntVect(DataArchive* archive, string& variable_name, const Uintah::TypeDescription* variable_type,
               int material, int levelIndex,
               IntVector& nijk0, IntVector& dimen,
               unsigned long time_step, FILE* VTK_FILE, int ifirst, int& nummat) 

{
  // query time info from dataarchive
  vector<int> index;
  vector<double> times;

  archive->queryTimesteps(index, times);
  ASSERTEQ(index.size(), times.size());
  
    
  bool cellNotFound = false;

    cerr <<time_step + ifirst<<" %outputting for times["<<time_step<<"] = " << times[time_step]<< endl;

    //__________________________________
    //  does the requested level exist
    bool levelExists = false;
    GridP grid = archive->queryGrid(times[time_step]); 
    int numLevels = grid->numLevels();

    int nimax=1, njmax=1, nkmax=1;
    vector<vector<int> > nptsv(numLevels);
   
    int npts = 0, nc = 0, nc0, type;
    for (int L = 0;L < numLevels; L++) {
      levelIndex = L;
      const LevelP level = grid->getLevel(L);
      if (level->getIndex() == levelIndex){
        levelExists = true;
      }
      //}
    if (!levelExists){
      cerr<< " Level " << levelIndex << " does not exist at this timestep " << time_step << endl;
    }

    int ni = 1, nj = 1, nk = 1;
    if(levelExists){   // only extract data if the level exists
      const LevelP level = grid->getLevel(levelIndex);
      //__________________________________
      // User input starting and ending indicies    

      // find the corresponding patches
      Level::selectType patches;
      level->selectPatches(IntVector(0,0,0), IntVector(1001,1001,1001), patches);
      if( patches.size() == 0){
	cerr << " Could not find any patches on Level " << level->getIndex()
	     << " Double check the starting and ending indices "<< endl;
	exit(1);
      }
      nptsv[L].resize(patches.size(),0);

      
      // query all the data up front
      if(levelIndex==0){
        for (int p=0; p < patches.size(); p++) {
	  IntVector low = patches[p]->getLowIndex();
	  IntVector high = patches[p]->getHighIndex();
	  if(high.x()>ni) ni = high.x();
	  if(high.y()>nj) nj = high.y();
	  if(high.z()>nk) nk = high.z();

	  // loop thru all the materials
	  ConsecutiveRangeSet matls = archive->queryMaterials("vol_frac_CC", patches[p], times[time_step]);
	  ConsecutiveRangeSet::iterator matlIter = matls.begin(); 
	  nummat = 0;
	  for(; matlIter != matls.end(); matlIter++){
	    int matl = *matlIter;
	    nummat++;
	  }
	}
	nijk0 = IntVector(ni,nj,nk);
	if(ni>nimax) nimax = ni;
	if(nj>njmax) njmax = nj;
	if(nk>nkmax) nkmax = nk;
	npts += (ni-1)*(nj-1)*(nk-1);
	if(nk==2){
	  nc += (ni-2)*(nj-2);
	  nc0 = 4;
	  type = 9;
	}else{
	  nc += (ni-2)*(nj-2)*(nk-2);
	  nc0 = 8;
	  type = 12;
	}
      }else{
	for (int p = 0; p < patches.size(); p++) {
	  // loop thru all the materials
	  ConsecutiveRangeSet matls = archive->queryMaterials("vol_frac_CC", patches[p], times[time_step]);
	  ConsecutiveRangeSet::iterator matlIter = matls.begin(); 
	  nummat = 0;
	  for(; matlIter != matls.end(); matlIter++){
	    int matl = *matlIter;
	    nummat++;
	  }
	  IntVector low = patches[p]->getLowIndex();
	  IntVector high = patches[p]->getHighIndex();
	  
	  ni = high.x() - low.x();
	  nj = high.y() - low.y();
	  nk = high.z() - low.z();
	  
          if(ni>nimax) nimax = ni;
          if(nj>njmax) njmax = nj;
          if(nk>nkmax) nkmax = nk;
	  
          if(nk==3) nk = 1;
	  
          nptsv[L][p] = npts;
	  npts += ni*nj*nk;
	  if(nk==1){
	    nc += (ni-1)*(nj-1);
	    nc0 = 4;
	    type = 9;
	  }else{
	    nc += (ni-1)*(nj-1)*(nk-1);
	    nc0 = 8;
	    type = 12;
	  }
	}  // for(p
      }
    } // if(levelExists
    } // if(L

    fprintf(VTK_FILE, "POINTS %d float\n", npts);

    npts = 0;

    for (int L = 0;L < numLevels; L++) {
      levelIndex = L;
      const LevelP level = grid->getLevel(L);
      if (level->getIndex() == levelIndex){
        levelExists = true;
      }
      //}
    if (!levelExists){
      cerr<< " Level " << levelIndex << " does not exist at this timestep " << time_step << endl;
    }

    int ni = 1, nj = 1, nk = 1;
    if(levelExists){   // only extract data if the level exists
      const LevelP level = grid->getLevel(levelIndex);
      //__________________________________
      // User input starting and ending indicies    

      // find the corresponding patches
      Level::selectType patches;
      level->selectPatches(IntVector(0,0,0), IntVector(1001,1001,1001), patches);
      if( patches.size() == 0){
	cerr << " Could not find any patches on Level " << level->getIndex()
	     << " Double check the starting and ending indices "<< endl;
	exit(1);
      }
      if(levelIndex==0){
        ni = 1; nj = 1; nk = 1;
        for (int p=0; p < patches.size(); p++) {
	  IntVector low = patches[p]->getLowIndex();
	  IntVector high = patches[p]->getHighIndex();
	  if(high.x()>ni) ni = high.x();
	  if(high.y()>nj) nj = high.y();
	  if(high.z()>nk) nk = high.z();
        }

	//# Write out POINTS
	int i, j, k;
	for (k=0; k<nk-1; k++)
	  for (j=0; j<nj-1; j++)
	    for (i=0; i<ni-1; i++) {
	      npts++;
	      IntVector c(i,j,k);
	      Point pt = level->getCellPosition(c);
	      Vector here = pt.asVector();
	      float rl;
	      rl = (float)here.x();
	      swap_4((char*)&rl);
	      fwrite(&rl,sizeof(float),1,VTK_FILE);
	      rl = (float)here.y();
	      swap_4((char*)&rl);
	      fwrite(&rl,sizeof(float),1,VTK_FILE);
	      rl = (float)here.z();
	      swap_4((char*)&rl);
	      fwrite(&rl,sizeof(float),1,VTK_FILE);
	    }
      }else{
	for (int p = 0; p < patches.size(); p++) {
	  IntVector low = patches[p]->getLowIndex();
	  IntVector high = patches[p]->getHighIndex();
	  
	  nk = high.z() - low.z() - 1;
          int nk1 = high.z()-1, klo = low.z();
          if(nk==2){
            nk1 = low.z()+2;
            klo = low.z()+1;
	  }
	  
	  //# Write out POINTS
	  int i, j, k;
	  for (k=klo; k<nk1; k++)
	    for (j=low.y(); j<high.y(); j++)
	      for (i=low.x(); i<high.x(); i++) {
		npts++;
		IntVector c(i,j,k);
		Point pt = level->getCellPosition(c);
		Vector here = pt.asVector();
		float rl;
		rl = (float)here.x();
		swap_4((char*)&rl);
		fwrite(&rl,sizeof(float),1,VTK_FILE);
		rl = (float)here.y();
		swap_4((char*)&rl);
		fwrite(&rl,sizeof(float),1,VTK_FILE);
		rl = (float)here.z();
		swap_4((char*)&rl);
		fwrite(&rl,sizeof(float),1,VTK_FILE);
	      } // for(i j k
	}  // for(p
      }// if(levelIndex>0
      
    }// if(levelExists
    } // for(L

    int num = nc*(nc0+1);
    fprintf(VTK_FILE,"\nCELLS %d %d\n",nc,num);
    nc = 0;

    for (int L = 0;L < numLevels; L++) {
      levelIndex = L;
      const LevelP level = grid->getLevel(L);
      if (level->getIndex() == levelIndex){
        levelExists = true;
      }
      //}
    if (!levelExists){
      cerr<< " Level " << levelIndex << " does not exist at this timestep " << time_step << endl;
    }

    int ni = 1, nj = 1, nk = 1, i, j, k;
    if(levelExists){   // only extract data if the level exists
      const LevelP level = grid->getLevel(levelIndex);
      //__________________________________
      // User input starting and ending indicies    

      // find the corresponding patches
      Level::selectType patches;
      level->selectPatches(IntVector(0,0,0), IntVector(1001,1001,1001), patches);
      if( patches.size() == 0){
	cerr << " Could not find any patches on Level " << level->getIndex()
	     << " Double check the starting and ending indices "<< endl;
	exit(1);
      }

      if(levelIndex==0){
        ni = 1; nj = 1; nk = 1;
        for (int p=0; p < patches.size(); p++) {
	  IntVector low = patches[p]->getLowIndex();
	  IntVector high = patches[p]->getHighIndex();
	  if(high.x()>ni) ni = high.x();
	  if(high.y()>nj) nj = high.y();
	  if(high.z()>nk) nk = high.z();
        }

        int nc1 = nc0;
        swap_4((char*)&nc1);
        int nk1 = nk - 2;
        if(nk==2) nk1 = 1;
        for(k=0; k<nk1; k++)
	  for(j=0; j<nj-2; j++)
	    for(i=0; i<ni-2; i++){
	      nc++;
              fwrite(&nc1,sizeof(int),1,VTK_FILE);
	      int i1[8] = {i   + (ni-1)*j     + (ni-1)*(nj-1)*k,
	                   i+1 + (ni-1)*j     + (ni-1)*(nj-1)*k,
	                   i+1 + (ni-1)*(j+1) + (ni-1)*(nj-1)*k,
	                   i   + (ni-1)*(j+1) + (ni-1)*(nj-1)*k,
                           i   + (ni-1)*j     + (ni-1)*(nj-1)*(k+1),
	                   i+1 + (ni-1)*j     + (ni-1)*(nj-1)*(k+1),
	                   i+1 + (ni-1)*(j+1) + (ni-1)*(nj-1)*(k+1),
	                   i   + (ni-1)*(j+1) + (ni-1)*(nj-1)*(k+1)};
              int iv0;
              for(iv0=0; iv0<nc0; iv0++){
		int iv = i1[iv0];
                swap_4((char*)&iv);
                fwrite(&iv,sizeof(int),1,VTK_FILE);
	      }
	    }
      }else{
	  int nc1 = nc0;
	  swap_4((char*)&nc1);
	  for (int p=0; p < patches.size(); p++) {
	    IntVector low = patches[p]->getLowIndex();
	    IntVector high = patches[p]->getHighIndex();
            ni = high.x() - low.x();
            nj = high.y() - low.y();
            nk = high.z() - low.z();
	    int nk1 = nk - 1;
	    if(nk==3) nk1 = 1;
            int i,j,k;
	    for(k=0; k<nk1; k++)
	      for(j=0; j<nj-1; j++)
		for(i=0; i<ni-1; i++){
		  nc++;
		  fwrite(&nc1,sizeof(int),1,VTK_FILE);
		  int i1[8] = {i   + ni*j     + ni*nj*k + nptsv[L][p],
			       i+1 + ni*j     + ni*nj*k + nptsv[L][p],
			       i+1 + ni*(j+1) + ni*nj*k + nptsv[L][p],
			       i   + ni*(j+1) + ni*nj*k + nptsv[L][p],
			       i   + ni*j     + ni*nj*(k+1) + nptsv[L][p],
			       i+1 + ni*j     + ni*nj*(k+1) + nptsv[L][p],
			       i+1 + ni*(j+1) + ni*nj*(k+1) + nptsv[L][p],
			       i   + ni*(j+1) + ni*nj*(k+1) + nptsv[L][p]};
		  int iv0;
		  for(iv0=0; iv0<nc0; iv0++){
		    int iv = i1[iv0];
		    swap_4((char*)&iv);
		    fwrite(&iv,sizeof(int),1,VTK_FILE);
		  }
		}
	  } // for(p
      } // if(levelIndex==0
      } //if(levelExits
    } // for(L

    fprintf(VTK_FILE,"\nCELL_TYPES %d\n",nc);
    swap_4((char*)&type);
    for(int i=0; i<nc; i++) fwrite(&type,sizeof(int),1,VTK_FILE);
    IntVector c1(nimax,njmax,nkmax);
    dimen = c1;
    return npts;
}

//______________________________________________________________________
//    Notes:
// Now the material index is kind of a hard thing.  There is no way
// to reliably determine a default material.  Materials are defined
// on the patch for each varialbe, so this subset of materials could
// change over patches.  We can guess, that there will be a material
// 0.  This shouldn't cause the program to crash.  It will spit out
// an exception and exit gracefully.


int main(int argc, char** argv)
{

  //__________________________________
  //  Default Values

  unsigned long time_start = 0;
  unsigned long time_end = (unsigned long)-1;
  string input_uda_name;  
  string output_file_name("-");
  IntVector var_start(0,0,0);
  IntVector var_end(1000,1000,1000);
  int levelIndex = 1;
  string variable_name;
  int ifirst = 1;
  int di = 1;

  int material = 0;
  
  //__________________________________
  // Parse arguments

  for(int i=1;i<argc;i++){
    string s=argv[i];
    if(s == "-v" || s == "--variable") {
      variable_name = string(argv[++i]);
    } else if (s == "-m" || s == "--material") {
      material = atoi(argv[++i]);
    } else if (s == "-vv" || s == "--verbose") {
      verbose = true;
    } else if (s == "-q" || s == "--quiet") {
      quiet = true;
    } else if (s == "-tlow" || s == "--timesteplow") {
      time_start = strtoul(argv[++i],(char**)NULL,10);
    } else if (s == "-thigh" || s == "--timestephigh") {
      time_end = strtoul(argv[++i],(char**)NULL,10);
    } else if (s == "-timestep" || s == "--timestep") {
      int val = strtoul(argv[++i],(char**)NULL,10);
      time_start = val;
      time_end = val;
    } else if (s == "-ifirst") {
      ifirst = atoi(argv[++i]);
    } else if (s == "-delta") {
      di = atoi(argv[++i]);
    } else if (s == "-istart" || s == "--indexs") {
      int x = atoi(argv[++i]);
      int y = atoi(argv[++i]);
      int z = atoi(argv[++i]);
      var_start = IntVector(x,y,z);
    } else if (s == "-iend" || s == "--indexe") {
      int x = atoi(argv[++i]);
      int y = atoi(argv[++i]);
      int z = atoi(argv[++i]);
      var_end = IntVector(x,y,z);
    } else if (s == "-l" || s == "--level") {
      levelIndex = atoi(argv[++i]);
    } else if( (s == "-h") || (s == "--help") ) {
      usage( "", argv[0] );
    } else if (s == "-uda") {
      input_uda_name = string(argv[++i]);
    } else if (s == "-o" || s == "--out") {
      output_file_name = string(argv[++i]);
    } else if (s == "--cellCoords" || s == "-cellCoords" ) {
      d_printCell_coords = true;
    }else {
      usage(s, argv[0]);
    }
  }
  
  if(input_uda_name == ""){
    cerr << "No archive file specified\n";
    usage("", argv[0]);
  }

  try {
    int nwant = 4;
    string varwant[4]={"press_CC","vel_CC","rho_CC","temp_CC"};
    DataArchive* archive = scinew DataArchive(input_uda_name);
    
    vector<string> vars;
    vector<const Uintah::TypeDescription*> types;

    archive->queryVariables(vars, types);
    ASSERTEQ(vars.size(), types.size());
    if (verbose) cout << "There are " << vars.size() << " variables:\n";
    unsigned int var_index = 0;

    //__________________________________    
    // read in cell indices from a file
  // query time info from dataarchive
  vector<int> index;
  vector<double> times;

  archive->queryTimesteps(index, times);
  ASSERTEQ(index.size(), times.size());
  if (!quiet){
    cout << "There are " << index.size() << " timesteps\n";
  }
  
  // set default max time value
  if (time_end == (unsigned long)-1) {
    if (verbose) {
      cout <<"Initializing time_step_upper to "<<times.size()-1<<"\n";
    }
    time_end = times.size() - 1;
  }      

  //__________________________________
  // bullet proofing 
  if (time_end >= times.size() || time_end < time_start) {
    cerr << "timestephigh("<<time_end<<") must be greater than " << time_start 
         << " and less than " << times.size()-1 << endl;
    exit(1);
  }
  if (time_start >= times.size() || time_end > times.size()) {
    cerr << "timestep must be between 0 and " << times.size()-1 << endl;
    exit(1);
  }

  FILE* VTK_FILE;
    
  //__________________________________
  // loop over timesteps
  if(ifirst<0){
    time_start = 0;
    di = time_end;
    ifirst = 1;
  }
  for (unsigned long time_step = time_start; time_step <= time_end; time_step += di) {
    var_start = IntVector(0,0,0);
    int i, j, k, nimax, njmax, nkmax;

    bool levelExists = false;
    GridP grid = archive->queryGrid(times[time_step]); 
    int numLevels = grid->numLevels();
   
    vector<int> numpatches(numLevels,0);
    vector<vector<IntVector> > low(numLevels), high(numLevels);
    for (int L = 0;L < numLevels; L++) {
      levelIndex = L;
      const LevelP level = grid->getLevel(L);
      if (level->getIndex() == levelIndex){
        levelExists = true;
      }
      //}
    if (!levelExists){
      cerr<< " Level " << levelIndex << " does not exist at this timestep " << time_step << endl;
    }

    if(levelExists){   // only extract data if the level exists
      const LevelP level = grid->getLevel(levelIndex);
      //__________________________________
      // User input starting and ending indicies    
          
        // find the corresponding patches
        Level::selectType patches;
        level->selectPatches(IntVector(0,0,0), IntVector(1001,1001,1001), patches);
        if( patches.size() == 0){
          cerr << " Could not find any patches on Level " << level->getIndex()
               << " Double check the starting and ending indices "<< endl;
          exit(1);
        }
        numpatches[levelIndex] = patches.size();
        low[levelIndex].resize(patches.size()); high[levelIndex].resize(patches.size());
	for(int p=0; p<patches.size(); p++){
	  low[levelIndex][p] = patches[p]->getLowIndex();// + IntVector(1,1,1);
          high[levelIndex][p] = patches[p]->getHighIndex();// - IntVector(1,1,1);
	}
    }
    }
    int npts;    

     char str[20],str1[20];
     str[0] = '\0';
     strcpy(str,"ice");
     sprintf(str1,"%04d",time_step/di+ifirst);
     strcat(str,str1);
     strcat(str,".vtk");
     //# Open vtk binary file
     if((VTK_FILE = fopen(str, "wb"))==NULL) {
        cerr << "Failed to open " << str << endl;
        return 0;
     }
     
     //# Write out HEADER
     fprintf(VTK_FILE, "# vtk DataFile Version 4.0\n");
     fprintf(VTK_FILE, "Data from uda to vtk\n");
     fprintf(VTK_FILE, "BINARY\n");
     fprintf(VTK_FILE, "DATASET UNSTRUCTURED_GRID\n");

     vector<vector<vector<vector<double> > > > vol_frac;
     vector<vector<vector<double> > > phi, phim;
     vector<vector<vector<Vector> > > phiv, phimv;
     int nummat;

     unsigned int vw;
     IntVector nijk0, dimen;
     for(vw=0; vw<nwant; vw++){
       int npts1 = 0;
       for(levelIndex=0; levelIndex<numLevels; levelIndex++){
       int p, pb, pe, nploop;
       if(levelIndex==0){
	 nploop = 1;
	 pb = 0;
	 pe = numpatches[levelIndex];
       }else{
	 nploop = numpatches[levelIndex];
       }
       for(p=0; p<nploop; p++){
	 if(levelIndex>0){
	   pb = p;
	   pe = p + 1;
	 }
    
	 bool var_found = false;
	 unsigned int var_index = 0;
	 for (;var_index < vars.size(); var_index++) {
	   if (varwant[vw] == vars[var_index]) {
	     variable_name=vars[var_index];
	     var_found = true;
	     break;
	   }
	 }
	 if(!var_found){
	   if(!vw){
	     cout<<"the first variable must be there"<<endl;
	     exit(1);
	   }
	   continue;
	 }
	 //__________________________________
	 // get type and subtype of data
	 const Uintah::TypeDescription* td = types[var_index];
	 const Uintah::TypeDescription* subtype = td->getSubType();

	 if(vw==0){
           switch (subtype->getType()) {
           case Uintah::TypeDescription::double_type:
	     {
	     if(p==0&&levelIndex==0){
	       npts=maxIntVect<double>(archive, variable_name, td, material,
				       levelIndex, nijk0, dimen,
				       time_step, VTK_FILE, ifirst, nummat);
	       nimax = dimen.x(); njmax = dimen.y(); nkmax = dimen.z();
	       vol_frac.resize(nummat);
	       for(material=0; material<nummat; material++){
		 vol_frac[material].resize(nimax);
		 for(i=0; i<nimax; i++){
		   vol_frac[material][i].resize(njmax);
		   for(j=0; j<njmax; j++){
		     vol_frac[material][i][j].resize(nkmax,0.0);
		   }
		 }
	       }
	       phi.resize(nimax); phim.resize(nimax);
	       phiv.resize(nimax); phimv.resize(nimax);
	       for(i=0; i<nimax; i++){
		 phi[i].resize(njmax);
		 phim[i].resize(njmax);
		 phiv[i].resize(njmax);
		 phimv[i].resize(njmax);
		 for(j=0; j<njmax; j++){
		   phi[i][j].resize(nkmax,0.0);
		   phim[i][j].resize(nkmax,0.0);
		   phiv[i][j].resize(nkmax);
		   phimv[i][j].resize(nkmax);
		 }
	       }
	     } // if(p==0&&...
	     material = 0;
	     }
	   break;
           case Uintah::TypeDescription::float_type:
	     npts = maxIntVect<float>(archive, variable_name, td, material,
				      levelIndex, var_start, var_end,
				      time_step, VTK_FILE, ifirst, nummat);
	     break;
           case Uintah::TypeDescription::int_type:
	     npts = maxIntVect<int>(archive, variable_name, td, material,
				    levelIndex, var_start, var_end,
				    time_step, VTK_FILE, ifirst, nummat);
	     break;
           case Uintah::TypeDescription::Vector:
	     npts = maxIntVect<Vector>(archive, variable_name, td, material,
				       levelIndex, var_start, var_end,
				       time_step, VTK_FILE, ifirst, nummat);    
	     break;
           /*case Uintah::TypeDescription::Other:
           if (subtype->getName() == "Stencil7") {
           printData<Stencil7>(archive, variable_name, td, material, use_cellIndex_file,
           levelIndex, var_start, var_end, cells,
           time_start, time_end, *output_stream);    
           break;
           }*/
              // don't break on else - flow to the error statement
           case Uintah::TypeDescription::Matrix3:
           case Uintah::TypeDescription::bool_type:
           case Uintah::TypeDescription::short_int_type:
           case Uintah::TypeDescription::long_type:
           case Uintah::TypeDescription::long64_type:
              cerr << "Subtype is not implemented\n";
              exit(1);
              break;
           default:
	     cerr << "Unknown subtype\n";
	     exit(1);
           }
	   if(p==0&&levelIndex==0) fprintf(VTK_FILE, "\nPOINT_DATA %d", npts);
        }
	if(p==0&&levelIndex==0){
	  var_start = IntVector(0,0,0);
	  var_end = nijk0;
	}
	for(material=0; material<nummat; material++){
	  string volfname = "vol_frac_CC";
	  printData<double>(archive, volfname, td, material,
			    levelIndex, var_start, var_end, pb, pe,
			    time_step, vol_frac[material]);
	}
	int ni = nijk0.x()-1, nj = nijk0.y()-1, nk = nijk0.z()-1, dir;
	//if(nk==2) nk=1;
        //__________________________________
        //  print data
        switch (subtype->getType()) {
        case Uintah::TypeDescription::double_type:
	  {
            if(variable_name=="press_CC"){
	      material = 0;
	      printData<double>(archive, variable_name, td, material,
				levelIndex, var_start, var_end, pb, pe,
				time_step, phi);
	    }else if(variable_name=="rho_CC"){
	      if(levelIndex>0){
		ni = high[levelIndex][p].x() - low[levelIndex][p].x();
		nj = high[levelIndex][p].y() - low[levelIndex][p].y();
		nk = high[levelIndex][p].z() - low[levelIndex][p].z();
		if(nk==3) nk = 1;
	      }
	      for(k=0; k<nk; k++)
		for(j=0; j<nj; j++)
		  for(i=0; i<ni; i++){
		    phi[i][j][k] = 0.0;
		  }
	      for(material=0; material<nummat; material++){
		printData<double>(archive, variable_name, td, material,
				  levelIndex, var_start, var_end, pb, pe,
				  time_step, phim);
		for(k=0; k<nk; k++)
		  for(j=0; j<nj; j++)
		    for(i=0; i<ni; i++){
		      phi[i][j][k] += phim[i][j][k];
		    }
	      }
              material = 0;
	    }else{
	      if(levelIndex>0){
		ni = high[levelIndex][p].x() - low[levelIndex][p].x();
		nj = high[levelIndex][p].y() - low[levelIndex][p].y();
		nk = high[levelIndex][p].z() - low[levelIndex][p].z();
		if(nk==3) nk = 1;
	      }
	      for(k=0; k<nk; k++)
		for(j=0; j<nj; j++)
		  for(i=0; i<ni; i++){
		    phi[i][j][k] = 0.0;
		  }
	      for(material=0; material<nummat; material++){
		printData<double>(archive, variable_name, td, material,
				  levelIndex, var_start, var_end, pb, pe,
				  time_step, phim);
		for(k=0; k<nk; k++)
		  for(j=0; j<nj; j++)
		    for(i=0; i<ni; i++){
		      phi[i][j][k] += vol_frac[material][i][j][k]*phim[i][j][k];
		    }
	      }
              material = 0;
	    }
	    if(p==0&&levelIndex==0) fprintf(VTK_FILE,"\nSCALARS %s float 1\nLOOKUP_TABLE default\n",variable_name.c_str());
	    if(levelIndex>0){
	      ni = high[levelIndex][p].x() - low[levelIndex][p].x();
	      nj = high[levelIndex][p].y() - low[levelIndex][p].y();
	      nk = high[levelIndex][p].z() - low[levelIndex][p].z();
	      if(nk==3) nk = 1;
	    }
	    for(k=0; k<nk; k++)
	      for(j=0; j<nj; j++)
		for(i=0; i<ni; i++){
		  npts1++;
		  float rl = phi[i][j][k];
		  swap_4((char*)&rl);
		  fwrite(&rl,sizeof(float),1,VTK_FILE);
		}
	  }
           break;
        case Uintah::TypeDescription::float_type:
	  {
	    vector<vector<vector<float> > > phi(ni);
	    for(i=0; i<ni; i++){
	      phi[i].resize(nj);
	      for(j=0; j<nj; j++){
		phi[i][j].resize(nk,0.0);
	      }
	    }
           printData<float>(archive, variable_name, td, material,
			    levelIndex, var_start, var_end, pb, pe,
              time_step, phi);
	  }
           break;
        case Uintah::TypeDescription::int_type:
	  {
	    vector<vector<vector<int> > > phi(ni);
	    for(i=0; i<ni; i++){
	      phi[i].resize(nj);
	      for(j=0; j<nj; j++){
		phi[i][j].resize(nk,0.0);
	      }
	    }
	   printData<int>(archive, variable_name, td, material,
			  levelIndex, var_start, var_end, pb, pe,
              time_step, phi);
	  }
           break;
        case Uintah::TypeDescription::Vector:
	  {
	    if(levelIndex>0){
	      ni = high[levelIndex][p].x() - low[levelIndex][p].x();
	      nj = high[levelIndex][p].y() - low[levelIndex][p].y();
	      nk = high[levelIndex][p].z() - low[levelIndex][p].z();
	      if(nk==3) nk = 1;
	    }
	    for(k=0; k<nk; k++)
	      for(j=0; j<nj; j++)
		for(i=0; i<ni; i++){
		  for(dir=0; dir<3; dir++)
		    phiv[i][j][k][dir] = 0.0;
		}
	    for(material=0; material<nummat; material++){
	      printData<Vector>(archive, variable_name, td, material,
				levelIndex, var_start, var_end, pb, pe,
				time_step, phimv);
	      for(k=0; k<nk; k++)
		for(j=0; j<nj; j++)
		  for(i=0; i<ni; i++){
		    for(dir=0; dir<3; dir++)
		    phiv[i][j][k][dir] += vol_frac[material][i][j][k]*phimv[i][j][k][dir];
		  }
	    }
	    material = 0;
	   if(p==0&&levelIndex==0) fprintf(VTK_FILE,"\nVECTORS %s float\n",variable_name.c_str());
	   for(k=0; k<nk; k++)
	     for(j=0; j<nj; j++)
	       for(i=0; i<ni; i++){
		 npts1++;
		 float rl = phiv[i][j][k][0];
		 swap_4((char*)&rl);
		 fwrite(&rl,sizeof(float),1,VTK_FILE);
		 rl = phiv[i][j][k][1];
		 swap_4((char*)&rl);
		 fwrite(&rl,sizeof(float),1,VTK_FILE);
		 rl = phiv[i][j][k][2];
		 swap_4((char*)&rl);
		 fwrite(&rl,sizeof(float),1,VTK_FILE);
	       }
	  }
	  break;
        /*case Uintah::TypeDescription::Other:
        if (subtype->getName() == "Stencil7") {
        printData<Stencil7>(archive, variable_name, td, material, use_cellIndex_file,
        levelIndex, var_start, var_end, cells,
        time_start, time_end, *output_stream);    
        break;
        }*/
           // don't break on else - flow to the error statement
        case Uintah::TypeDescription::Matrix3:
        case Uintah::TypeDescription::bool_type:
        case Uintah::TypeDescription::short_int_type:
        case Uintah::TypeDescription::long_type:
        case Uintah::TypeDescription::long64_type:
           break;
        default:
	  break;
	  }
       } // for(p
       } // for(levelIndex
     } // for(vw
     fclose(VTK_FILE);
  }// for(time
  
  } catch (Exception& e) {
     cerr << "Caught exception: " << e.message() << endl;
     exit(1);
  } catch(...){
     cerr << "Caught unknown exception\n";
     exit(1);
  }
}
