#include <iostream>
using namespace std;

#define _TIMESFC_
#include <Packages/Uintah/CCA/Ports/SFC.h>
#include <Packages/Uintah/Core/Parallel/Parallel.h>
#include<Core/Thread/Time.h>
using namespace Uintah;

#define depth 1000.0f
#define height 1000.0f
#define width 1000.0f

#define DIM 2
#define LOCS float
int main(int argc, char** argv)
{
	Uintah::Parallel::determineIfRunningUnderMPI( argc, argv);	
	Uintah::Parallel::initializeManager( argc, argv, "");
	ProcessorGroup *d_myworld=Uintah::Parallel::getRootProcessorGroup();
	MPI_Comm Comm=d_myworld->getComm();	
	vector<LOCS> locs;
	vector<unsigned int> orders;
	
	
	int ref=3;
	if(argc>=2)
		ref=atoi(argv[1]);

	int div=(int)pow((float)DIM,ref);
	
	unsigned int P=d_myworld->size();
	unsigned int N=(unsigned int)pow((float)BINS,ref);
	unsigned int n=N/P;
	int rem=N%P;
	int rank=d_myworld->myrank();
	LOCS xx,yy;

#if DIM == 3
	SFC3f mycurve(HILBERT,d_myworld);
#elif DIM == 2
	SFC2f mycurve(HILBERT,d_myworld);
#endif
//	mycurve.Profile();
//	return 0;
	int starti=0;

	if(rank<rem)
	{
		starti=rank*(n+1);
		n++;
	}
	else
	{
		starti=rem*(n+1)+(rank-rem)*n;
	}
#if DIM==3
	LOCS zz;
	LOCS dz=LOCS(depth/div);
#endif	

	LOCS dx=LOCS(width/div), dy=LOCS(height/div);
	int i=0;
	xx=(-width+width/div)/2.0f;
	for(int x=0;x<div;x++)
	{	
		yy=(-height+height/div)/2.0f;
		for(int y=0;y<div;y++)
		{	
#if DIM==3
			zz=(-depth+depth/div)/2.0f;
			for(int z=0;z<div;z++)
			{
#endif
				if(i>=starti && i<(int)(starti+n))
				{
					locs.push_back(xx);
					locs.push_back(yy);
#if DIM==3
					locs.push_back(zz);
#endif	
				}
				i++;
#if DIM==3
				zz+=dz;
			}
#endif
			yy+=dy;
		}
		xx+=dx;
	}

	mycurve.SetLocalSize(n);
	mycurve.SetRefinements(ref);
	mycurve.SetOutputVector(&orders);
	mycurve.SetLocations(&locs); 
	mycurve.SetMergeParameters(3000,2,.15);
#if DIM==3
	mycurve.SetDimensions(height,width,depth);
	mycurve.SetCenter(0,0,0);
#elif DIM==2
	mycurve.SetDimensions(height,width);
	mycurve.SetCenter(0,0);
#endif
	double start, finish;
#ifdef _TIMESFC_
	double ttime;
	start=timer->currentSeconds();
#endif
	mycurve.GenerateCurve();
#ifdef _TIMESFC_
	finish=timer->currentSeconds();
	ttime=finish-start;


	double sum;
	
	MPI_Reduce(&ttime,&sum,1,MPI_DOUBLE,MPI_SUM,0,Comm);
	ttime=sum/P;
	MPI_Reduce(&sertime,&sum,1,MPI_DOUBLE,MPI_SUM,0,Comm);
	sertime=sum/P;
	MPI_Reduce(&ptime,&sum,1,MPI_DOUBLE,MPI_SUM,0,Comm);
	ptime=sum/P;
	MPI_Reduce(&cleantime,&sum,1,MPI_DOUBLE,MPI_SUM,0,Comm);
	cleantime=sum/P;
	MPI_Reduce(&gtime,&sum,1,MPI_DOUBLE,MPI_SUM,0,Comm);
	gtime=sum/P;
	
	if(rank==0)
		cout << N << " " << n << " " << ttime << " " << sertime << " " << ptime << " " << cleantime << " " << gtime <<  endl;
#endif

	Uintah::Parallel::finalizeManager();
	return 0;
}
