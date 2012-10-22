/*
 * The MIT License
 *
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
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

#include <iostream>
using namespace std;

#include <CCA/Ports/SFC.h>
#include <Core/Parallel/Parallel.h>
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
	vector<LOCS> locs, locss;
	vector<DistributedIndex> orders, orderss;
	
	
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

  SFC<LOCS> mycurve(d_myworld);
  


#if DIM == 3
    mycurve.SetNumDimensions(3);
#elif DIM == 2
    mycurve.SetNumDimensions(2);
#endif
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
			  locss.push_back(xx);
				locss.push_back(yy);
#if DIM==3
				locss.push_back(zz);
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
  mycurve.SetMergeMode(1);
  mycurve.SetCleanup(BATCHERS);
  mycurve.SetMergeParameters(3000,500,2,.15);  //Should do this by profiling

  LOCS dim[3]={width,height,depth};
  LOCS center[3]={0,0,0};
  mycurve.SetDimensions(dim);
  mycurve.SetCenter(center);

  if(rank==0)
    cout << " Generating curve in parallel\n";

  MPI_Barrier(Comm);
  
  double start=Time::currentSeconds();
  mycurve.GenerateCurve();
  double finish=Time::currentSeconds();
  
  cout << rank << ": Time to generate curve:" << finish-start << endl;

  MPI_Barrier(Comm);

  orderss.resize(N);
  mycurve.SetLocalSize(N);
  mycurve.SetOutputVector(&orderss);
  mycurve.SetLocations(&locss);
  
  if(rank==0)
    cout << " Generating curve in serial\n";
  MPI_Barrier(Comm);

  start=Time::currentSeconds();
  mycurve.GenerateCurve(true);
  finish=Time::currentSeconds();
  
  cout << rank << ": Time to generate curve:" << finish-start << endl;

  MPI_Barrier(Comm);

  if(rank==0)
  {
    cout << "Verifying curve\n";
	  unsigned int pn=N/P;
    unsigned int j=0,r;
    unsigned int starti;
    for(unsigned int i=0;i<n;i++)
    {
	    if(orders[i].p<rem)
	    {
		    starti=orders[i].p*(pn+1);
	    }
	    else
	    {
		    starti=rem*(pn+1)+(orders[i].p-rem)*pn;
	    }

      int index1=starti  + orders[i].i;
      int index2=orderss[j].i;
      if(index1!=index2)
        cout << j << ": " << index1 << "!=" << index2 << "\n";

      //cout << "index1:" << orders[i].p << ":" << orders[i].i << " index2:" << orderss[j].p << ":" << orderss[j].i << endl;
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

      MPI_Recv(&orders[0],r*sizeof(DistributedIndex),MPI_BYTE,p,0,d_myworld->getComm(), &status);
      for(unsigned int i=0;i<r;i++)
      {
	      if(orders[i].p<rem)
	      {
		      starti=orders[i].p*(pn+1);
  	    } 
	      else
	      {
		      starti=rem*(pn+1)+(orders[i].p-rem)*pn;
  	    }

        int index1=starti  + orders[i].i;
        int index2=orderss[j].i;
        if(index1!=index2)
          cout << j << ": " << index1 << "!=" << index2 <<  "\n";
        //cout << "index1:" << orders[i].p << ":" << orders[i].i << " index2:" << orderss[j].p << ":" << orderss[j].i << endl;
        j++;
      }
    }
  }
  else
  {
    MPI_Send(&orders[0],n*sizeof(DistributedIndex),MPI_BYTE,0,0,d_myworld->getComm());
  }

	Uintah::Parallel::finalizeManager();
	return 0;
}
