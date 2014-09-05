#include <iostream>
using namespace std;

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
	
	vector<LOCS> locs, locss ;
	vector<unsigned int> orders, orderss;
	
	int ref=3;
	if(argc>=2)
		ref=atoi(argv[1]);

	int div=(int)pow((float)DIM,ref);
	
	unsigned int N=(unsigned int)pow((float)BINS,ref);
	unsigned int n=N/d_myworld->size();
	int rem=N%d_myworld->size();
	int rank=d_myworld->myrank();
	LOCS xx,yy;
#if DIM==3
	LOCS zz;
	LOCS dz=LOCS(depth/div);
#endif	

	LOCS dx=LOCS(width/div), dy=LOCS(height/div);

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
				locss.push_back(xx);
				locss.push_back(yy);
#if DIM==3
				locss.push_back(zz);
				
				zz+=dz;
			}
#endif
			yy+=dy;
		}
		xx+=dx;
	}

	int start=0;

	if(rank<rem)
	{
		start=rank*(n+1);
		n++;
	}
	else
	{
		start=rem*(n+1)+(rank-rem)*n;
	}
	
	for(unsigned int i=0;i<n;i++) 
	{
		locs.push_back(locss[(start+i)*DIM]);
		locs.push_back(locss[(start+i)*DIM+1]);
#if DIM==3
		locs.push_back(locss[(start+i)*DIM+2]);
#endif
	}

#if DIM == 3
	SFC3f mycurve(HILBERT,d_myworld);
#elif DIM == 2
	SFC2f mycurve(HILBERT,d_myworld);
#endif
	mycurve.SetLocalSize(n);
	mycurve.SetRefinements(ref);
	mycurve.SetOutputVector(&orders);
	mycurve.SetLocations(&locs); 
	mycurve.SetMergeParameters(4000,3,.15);
#if DIM==3
	mycurve.SetDimensions(height,width,depth);
	mycurve.SetCenter(0,0,0);
#elif DIM==2
	mycurve.SetDimensions(height,width);
	mycurve.SetCenter(0,0);
#endif
	SCIRun::Time *timer;

	double stime=timer->currentSeconds();
	mycurve.GenerateCurve();
	double ftime=timer->currentSeconds();
	
	if(rank==0)
		cout << ftime-stime << endl;
	
//	for(unsigned int i=0; i<n;i++)
//	{
//		cout << orders[i] << " ";
//	}
//	cout << endl;
	
	
	mycurve.SetLocations(&locss); 
	mycurve.SetLocalSize(N);
	mycurve.SetOutputVector(&orderss);

	stime=timer->currentSeconds();
	mycurve.GenerateCurve(true);
	ftime=timer->currentSeconds();
	if(rank==0)
		cout << ftime-stime << endl;
	
	if(rank==0)
	{
		unsigned int j=0,r;
		for(unsigned int i=0;i<n;i++)
		{
			if(orders[i]!=orderss[j])
				cout << j << ":" << orders[i] << "!=" << orderss[j] << " ";
			j++;
		}
	
		n=N/d_myworld->size();
		MPI_Status status;
		for(int p=1;p<d_myworld->size();p++)
		{
			if(p<rem)
				r=n+1;
			else
				r=n;
			
			MPI_Recv(&orders[0],r,MPI_INT,p,0,d_myworld->getComm(), &status);

			for(unsigned int i=0;i<r;i++)
			{
				if(orders[i]!=orderss[j])
					cout << j << ":" << orders[i] << "!=" << orderss[j] << " ";
				j++;
			}
		}
		cout << endl;
	}
	else
	{
		MPI_Send(&orders[0],n,MPI_INT,0,0,d_myworld->getComm());
	}
	Uintah::Parallel::finalizeManager();
	return 0;
}
