/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */


#include <Core/Parallel/Parallel.h>
#include <Core/Parallel/ProcessorGroup.h>
#include <CCA/Components/Regridder/BNRRegridder.h>
#include <CCA/Components/Regridder/PatchFixer.h>
using namespace Uintah;

#include <iostream>

using namespace Uintah;
using namespace std;
const int X_RES=10, Y_RES=10;
float prob=.98;
int f_array[X_RES][Y_RES]=
{
	{1,1,0,0,0,0,0,0,0,0},
	{1,1,0,0,0,0,0,0,0,0},
	{1,1,1,1,0,0,0,0,0,0},
	{1,1,1,1,0,0,0,0,0,0},
	{0,0,1,1,1,1,0,0,0,0},
	{0,0,1,1,1,1,0,0,0,0},
	{0,0,0,0,1,1,1,1,0,0},
	{0,0,0,0,1,1,1,1,0,0},
	{0,0,0,0,0,0,1,1,1,1},
	{0,0,0,0,0,0,1,1,1,1}
};
/*
int f_array[X_RES][Y_RES]=
{
	{1,1,1,1,1,1,1,1,0,0},
	{1,1,1,1,0,1,1,1,0,0},
	{1,1,1,1,0,0,0,0,1,1},
	{1,1,0,1,1,0,0,0,1,1},
	{1,1,0,1,1,1,0,0,1,1},
	{1,1,0,0,1,1,1,0,1,1},
	{1,1,0,0,0,1,1,1,1,1},
	{1,1,0,0,0,0,1,1,1,1},
	{0,0,0,0,1,1,1,1,1,1},
	{0,0,0,1,1,1,1,1,1,1}
};
*/
int main(int argc, char** argv)
{
	Uintah::Parallel::determineIfRunningUnderMPI( argc, argv );
	Uintah::Parallel::initializeManager( argc, argv );

	ProcessorGroup *d_myworld=Uintah::Parallel::getRootProcessorGroup();
	int rank=d_myworld->myrank();
	int numprocs=d_myworld->size();
	cout << "rank:" << rank <<  ": Starting \n";
	
	srand(rank*rank*rank);
	BNRRegridder bnr(d_myworld);
	PatchFixer fixup(d_myworld);
	vector<Region> patches;
	Region patch;

	vector<IntVector> flags;
	int size=X_RES*Y_RES/numprocs;
	int rem=X_RES*Y_RES%numprocs;
	int mystart=0;
	int mysize=size;
	if(rank<rem)
		mysize++;
	
	for(int p=0;p<numprocs;p++)
	{
		if(p==rank)
			break;
		if(p<rem)
			mystart+=size+1;
		else
			mystart+=size;
	}
	int f=0;	
	for(int x=0;x<X_RES;x++)
	{
		for(int y=0;y<Y_RES;y++,f++)
		{
			if(f>=mystart && f<mystart+mysize)
			{
				if(f_array[x][y]==1)
				//if(r<=prob)
				{
					IntVector flag;
					flag[0]=x;
					flag[1]=y;
					flag[2]=0;
					flags.push_back(flag);
				}
			}
		}
	}
/*	
	cout << rank << ": flags:";
	for(unsigned int f=0;f<flags.size();f++)	
	{
			cout << "{" << flags[f][0] << "," << flags[f][1] << "," << flags[f][2] << "},";
	}
	cout << endl;
*/
	double start1, start2, finish1, finish2;
	if(rank==0)
		cout << "rank:" << rank <<  ": Starting BR\n";
	start1=MPI_Wtime();
	bnr.RunBR(flags,patches);
	finish1=MPI_Wtime();
	
	if(rank==0)
	{
		cout << "rank:" << rank << ": BR done\n";
			cout << "There are: " << patches.size() << " patches.\n";
			/*
			cout << "They are: ";
			for(unsigned int p=0;p<patches.size();p++)
			{
					cout << "{";
					cout << patches[p].low[0] << "-" <<patches[p].high[0] << ", ";
					cout << patches[p].low[1] << "-" <<patches[p].high[1] << ", ";
					cout << patches[p].low[2] << "-" <<patches[p].high[2] << "} ";
			}
			cout << endl;
			cout << "Starting Fixup\n";
			*/
	}	

	//FIXUP
	start2=MPI_Wtime();
	fixup.FixUp(patches);
	finish2=MPI_Wtime();
	
	if(rank==0)
	{
			cout << "Fixup done\n";
			cout << "There are: " << patches.size() << " patches.\n";
		
			cout << "They are: ";
			for(unsigned int p=0;p<patches.size();p++)
			{
					cout << "{";
					cout << patches[p].getLow()[0] << "-" <<patches[p].getHigh()[0] << ", ";
					cout << patches[p].getLow()[1] << "-" <<patches[p].getHigh()[1] << ", ";
					cout << patches[p].getLow()[2] << "-" <<patches[p].getHigh()[2] << "} ";
			}
			cout << endl;
			
	}
		
	MPI_Barrier(MPI_COMM_WORLD);	
	//if(rank==0)
		cout << "Timings: BR=" << finish1-start1 << " FixUp=" << finish2-start2 << " Total=" << finish1-start1+finish2-start2 << endl;

	
}
