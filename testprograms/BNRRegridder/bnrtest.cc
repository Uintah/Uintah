
#include <Packages/Uintah/Core/Parallel/Parallel.h>
#include <Packages/Uintah/CCA/Components/Regridder/BNRRegridder.h>
#include <Packages/Uintah/CCA/Components/Regridder/PatchFixer.h>
#include <Packages/Uintah/Core/Parallel/ProcessorGroup.h>

#include <iostream>
using namespace std;

const int X=700, Y=700;
float prob=.98;
/*
int f_array[X][Y]=
{
	{1,1,0,0,0,0,0,0,0,0},
	{1,1,0,0,0,0,0,0,0,0},
	{1,1,1,1,0,0,0,0,0,0},
	{1,1,1,1,0,0,0,0,0,0},
	{0,0,1,1,0,0,0,0,0,0},
	{0,0,1,1,1,1,1,1,1,1},
	{0,0,1,1,1,1,1,1,1,1},
	{0,0,0,0,0,0,0,0,1,1},
	{0,0,0,0,0,0,0,0,1,1},
	{0,0,0,0,0,0,0,0,1,1}
};
*/
/*
int f_array[X][Y]=
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
	Uintah::Parallel::initializeManager( argc, argv, "" );

	ProcessorGroup *d_myworld=Uintah::Parallel::getRootProcessorGroup();
	int rank=d_myworld->myrank();
	int numprocs=d_myworld->size();
	cout << "rank:" << rank <<  ": Starting \n";
	
	srand(rank*rank*rank);
	BNRRegridder bnr(d_myworld);
	PatchFixer fixup(d_myworld);
	vector<PseudoPatch> patches;
	PseudoPatch patch;

	vector<IntVector> flags;
	int size=X*Y/numprocs;
	int rem=X*Y%numprocs;
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
	for(int x=0;x<X;x++)
	{
		for(int y=0;y<Y;y++,f++)
		{
			if(f>=mystart && f<mystart+mysize)
			{
				float r=drand48();				
				//	if(f_array[x][y]==1)
				if(r<=prob)
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
			*/
	}
		
	MPI_Barrier(MPI_COMM_WORLD);	
	if(rank==0)
		cout << "Timings: BR=" << finish1-start1 << " FixUp=" << finish2-start2 << " Total=" << finish1-start1+finish2-start2 << endl;

	
}
