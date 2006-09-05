#ifndef _SFC
#define _SFC


#include <vector>
#include <queue>
#include <iostream>
#include <fstream>
#include <math.h>
using namespace std;

#include<Packages/Uintah/Core/Parallel/ProcessorGroup.h>
#include<Core/Thread/Time.h>

#include <Packages/Uintah/CCA/Ports/share.h>
#include <Core/Exceptions/InternalError.h>

namespace Uintah{

#ifdef _TIMESFC_
SCIRun::Time *timer;
double start, finish;
double ptime=0, cleantime=0, sertime=0, gtime=0;
#endif
enum Curve {HILBERT, MORTON, GREY};
enum CleanupType{BATCHERS,LINEAR};

#define SERIAL 1
#define PARALLEL 2
template<class BITS>
struct History
{
	unsigned int i;
	BITS bits;
};

extern SCISHARE int dir3[8][3];
extern SCISHARE int dir2[4][2];
extern SCISHARE int dir1[2][1];

extern SCISHARE int hinv3[][8];
extern SCISHARE int ginv3[][8];
extern SCISHARE int hinv2[][4];
extern SCISHARE int ginv2[][4];

extern SCISHARE int horder3[][8];
extern SCISHARE int gorder3[][8];
extern SCISHARE int morder3[][8];
extern SCISHARE int horder2[][4];
extern SCISHARE int gorder2[][4];
extern SCISHARE int morder2[][4];

extern SCISHARE int horient3[][8];
extern SCISHARE int gorient3[][8];
extern SCISHARE int morient3[][8];
extern SCISHARE int horient2[][4];
extern SCISHARE int gorient2[][4];
extern SCISHARE int morient2[][4];

extern SCISHARE int orient1[][2];
extern SCISHARE int order1[][2];

#define REAL double
#define EPSILON 1e-6

#define BINS (1<<DIM)

template<int DIM,class BITS>
void outputhistory(BITS history)
{
        BITS mask=0;
        BITS val;
        unsigned int bits;

        bits=sizeof(BITS)*8;
        //output bucket history
        for(int i=0;i<DIM;i++)
        {
            mask<<=1;
            mask|=1;
        }
        mask<<=(bits-DIM);
        while(bits>0)
        {
                val=history&mask;
                val>>=(bits-2);
                cout << val;
                mask>>=DIM;
                bits-=DIM;
        }
}
template<int DIM, class LOCS>
class SFC
{
public:
	SFC(int dir[][DIM], const ProcessorGroup *d_myworld) : dir(dir),set(0), locsv(0), locs(0), orders(0), sendbuf(0), recievebuf(0), mergebuf(0), d_myworld(d_myworld), block_size(3000), blocks_in_transit(3), sample_percent(.1), cleanup(BATCHERS) {};
	virtual ~SFC() {};
	void GenerateCurve(int mode=0);
	void SetRefinements(int refinements);
	void SetLocalSize(unsigned int n);
	void SetLocations(vector<LOCS> *locs);
	void SetOutputVector(vector<unsigned int> *orders);
	void SetMergeParameters(unsigned int block_size, unsigned int blocks_in_transit, float sample_percent);
	void SetBlockSize(unsigned int b) {block_size=b;};
	void SetBlocksInTransit(unsigned int b) {blocks_in_transit=b;};
	void SetSamplePercent(float p) {sample_percent=p;};
	void SetCleanup(CleanupType cleanup) {this->cleanup=cleanup;};

protected:
	
	//order and orientation arrays
	int (*order)[BINS];
	int (*orientation)[BINS];
	int (*inverse)[BINS];
	
	//direction array
	int (*dir)[DIM];

	//curve parameters
	REAL dimensions[DIM];
	REAL center[DIM];
	int refinements;
	unsigned int n;
	
	//byte variable used to determine what curve parameters are set
	unsigned char set;

	//XY(Z) locations of points
	vector<LOCS> *locsv;
	LOCS *locs;

	//Output vector
	vector<unsigned int> *orders;
	
	//Buffers
	void* sendbuf;
	void* recievebuf;
	void* mergebuf;
	
	const ProcessorGroup *d_myworld;
	
	//Merge-Exchange Parameters
	unsigned int block_size;
	unsigned int blocks_in_transit;
	float sample_percent;
	
	unsigned int *n_per_proc;
	CleanupType cleanup;
	
	int rank, P;
	MPI_Comm Comm;	
	void Serial();
	void SerialR(unsigned int* orders,vector<unsigned int> *bin, unsigned int n, REAL *center, REAL *dimension, unsigned int o=0);

  template<class BITS> void SerialH();
	template<class BITS> void SerialHR(unsigned int* orders,History<BITS>* corders,vector<unsigned int > *bin, unsigned int n, REAL *center, REAL *dimension,                                     unsigned int o=0, int r=1, BITS history=0);
	template<class BITS> void Parallel();
	template<class BITS> int MergeExchange(int to);	
	template<class BITS> void PrimaryMerge();
	template<class BITS> void Cleanup();
	template<class BITS> void Batchers();
	template<class BITS> void Linear();

	virtual unsigned char Bin(LOCS *point, REAL *center)=0;
};

template<class LOCS>
class SFC1D : public SFC<1,LOCS>
{
  public:
    SFC1D(const ProcessorGroup *d_myworld) : SFC<1,LOCS>(dir1,d_myworld) 
    {
      SFC<1,LOCS>::order=order1;
      SFC<1,LOCS>::orientation=orient1;
      SFC<1,LOCS>::inverse=order1;
    }
                        
		void SetDimensions(REAL wx);
		void SetCenter(REAL x);
		void SetRefinementsByDelta(REAL deltax);
  private:
    inline unsigned char Bin(LOCS *point, REAL *center);
};

template<class LOCS>
class SFC2D : public SFC<2,LOCS>
{

	public:
		SFC2D(Curve curve,const ProcessorGroup *d_myworld) : SFC<2,LOCS>(dir2,d_myworld) {SetCurve(curve);};
		virtual ~SFC2D() {};
		void SetCurve(Curve curve);
		void SetDimensions(REAL wx, REAL wy);
		void SetCenter(REAL x, REAL y);
		void SetRefinementsByDelta(REAL deltax, REAL deltay);

	private:
		 inline unsigned char Bin(LOCS *point, REAL *center);

};

template<class LOCS>
class SFC3D : public SFC<3,LOCS>
{
	public:
		SFC3D(Curve curve,const ProcessorGroup *d_myworld) : SFC<3,LOCS>(dir3, d_myworld) {SetCurve(curve);};
		virtual ~SFC3D() {};
		void SetCurve(Curve curve);
		void SetDimensions(REAL wx, REAL wy, REAL wz);
		void SetCenter(REAL x, REAL y, REAL z);
		void SetRefinementsByDelta(REAL deltax, REAL deltay, REAL deltaz);

private:
		inline unsigned char Bin(LOCS *point, REAL *center);
};

class SFC1f : public SFC1D<float>
{
	public:
		SFC1f(const ProcessorGroup *d_myworld) : SFC1D<float>(d_myworld) {};
};

class SFC1d : public SFC1D<double>
{
	public:
		SFC1d(const ProcessorGroup *d_myworld) : SFC1D<double>(d_myworld) {};
};

class SFC2f : public SFC2D<float>
{
	public:
		SFC2f(Curve curve,const ProcessorGroup *d_myworld) : SFC2D<float>(curve,d_myworld) {};
};

class SFC2d : public SFC2D<double>
{
	public:
		SFC2d(Curve curve,const ProcessorGroup *d_myworld) : SFC2D<double>(curve,d_myworld) {} ;
};

class SFC3f : public SFC3D<float>
{
	public:
		SFC3f(Curve curve,const ProcessorGroup *d_myworld) : SFC3D<float>(curve,d_myworld) {};
};

class SFC3d : public SFC3D<double>
{
	public:
		SFC3d(Curve curve,const ProcessorGroup *d_myworld) : SFC3D<double>(curve,d_myworld) {} ;
};


/***********SFC**************************/
const char errormsg[6][30]={
	"Locations vector not set\n",
	"Output vector not set\n", 
	"Local size not set\n",
	"Dimensions not set\n",
	"Center not set\n", 
	"Refinements not set\n",
					 };

template<int DIM, class LOCS>
void SFC<DIM,LOCS>::GenerateCurve(int mode)
{
	int errors=0;
	unsigned char mask;
	P=d_myworld->size();
	if(P==1 || mode==1)
	{
		errors=5;
		mask=0x0f;
	}
	else 
	{
		errors=6;
		mask=0x1f;
	}	
		
	char res=mask&set;
	if(res!=mask)
	{	
		cerr << "Error(s) forming SFC:\n************************\n";
		mask=1;
		for(int i=0;i<errors;i++)
		{
			res=mask&set;
			if(res!=mask)
			{
				cerr << "  " << errormsg[i];
			}
			mask=mask<<1;
		}
		cerr << "************************\n";	
		return;
	}
	if(P==1 || mode==1 )
	{
		Serial();
	}
	else
	{
		//make new sub group if needed?
		rank=d_myworld->myrank();
		Comm=d_myworld->getComm();
		//Pick which generate to use
    if((int)refinements*DIM<=(int)sizeof(unsigned char)*8)
    {
      Parallel<unsigned char>();
    }
    else if((int)refinements*DIM<=(int)sizeof(unsigned short)*8)
    {
      Parallel<unsigned short>();
    }
    else if((int)refinements*DIM<=(int)sizeof(unsigned int)*8)
		{
			Parallel<unsigned int>();
		}
		else
		{
      if((int)refinements*DIM>(int)sizeof(unsigned long long)*8)
      {
        refinements=sizeof(unsigned long long)*8/DIM;
        cerr << "Warning: Not enough bits to form full SFC lowering refinements to: " << refinements << endl;
      }		 	
			Parallel<unsigned long long>();
		}
	}
}

template<int DIM, class LOCS>
void SFC<DIM,LOCS>::Serial()
{
  if(n!=0)
  {
	  orders->resize(n);

	  unsigned int *o=&(*orders)[0];
	
	  for(unsigned int i=0;i<n;i++)
	  {
		  o[i]=i;
	  }

	  vector<unsigned int> bin[BINS];
	  for(int b=0;b<BINS;b++)
	  {
		  bin[b].reserve(n/BINS);
	  }
	  //Recursive call
	  SerialR(o,bin,n,center,dimensions);
  }
}

template<int DIM, class LOCS> template<class BITS>
void SFC<DIM,LOCS>::SerialH()
{
  if(n!=0)
  {
	  orders->resize(n);

	  History<BITS> *sbuf=(History<BITS>*)sendbuf;
	  unsigned int *o=&(*orders)[0];
	
	  for(unsigned int i=0;i<n;i++)
	  {
		  o[i]=i;
	  }

	  vector<unsigned int> bin[BINS];
	  for(int b=0;b<BINS;b++)
	  {
		  bin[b].reserve(n/BINS);
	  }
	  //Recursive call
	  SerialHR<BITS>(o,sbuf,bin,n,center,dimensions);
  }
}

template<int DIM, class LOCS> 
void SFC<DIM,LOCS>::SerialR(unsigned int* orders,vector<unsigned int> *bin, unsigned int n, REAL *center, REAL *dimension, unsigned int o)
{
	REAL newcenter[BINS][DIM], newdimension[DIM];
	unsigned int size[BINS];

	unsigned char b;
	unsigned int i;
	unsigned int index=0;

	//Empty bins
	for(b=0;b<BINS;b++)
	{
		bin[b].clear();
	}

	//Bin points
	for(i=0;i<n;i++)
	{
		b=Bin(&locs[orders[i]*DIM],center);
		bin[inverse[o][b]].push_back(orders[i]);
	}

	//Reorder points by placing bins together in order
	for(b=0;b<BINS;b++)
	{
		size[b]=(unsigned int)bin[b].size();
		memcpy(&orders[index],&bin[b][0],sizeof(unsigned int)*size[b]);
		index+=size[b];
	}

	//Halve all dimensions
	for(int d=0;d<DIM;d++)
	{
		newdimension[d]=dimension[d]*.5;
	}

	//recursivly call
	for(b=0;b<BINS;b++)
	{
		for(int d=0;d<DIM;d++)
		{
			newcenter[b][d]=center[d]+dimension[d]*.25*dir[order[o][b]][d];		
		}

		if(size[b]==n)
		{
			bool same=true;
			//Check if locations are the same
			REAL l[DIM];
			memcpy(l,&locs[orders[0]*DIM],sizeof(LOCS)*DIM);
			i=1;
			while(same && i<n)
			{
				for(int d=0;d<DIM;d++)
				{
					if(l[d]-locs[orders[i]*DIM+d]>EPSILON || l[d]-locs[orders[i]*DIM+d]<-EPSILON)
					{
						same=false;
						break;
					}
				}
				i++;
			}

			if(!same)
			{
				SerialR(orders,bin,size[b],newcenter[b],newdimension,orientation[o][b]);
			}

		}
		else if(size[b]>1 )
		{
				SerialR(orders,bin,size[b],newcenter[b],newdimension,orientation[o][b]);
		}
		orders+=size[b];
	}
}

template<int DIM, class LOCS> template<class BITS> 
void SFC<DIM,LOCS>::SerialHR(unsigned int* orders, History<BITS>* corders,vector<unsigned int> *bin, unsigned int n, REAL *center, REAL *dimension, unsigned int o, int r, BITS history)
{
	REAL newcenter[BINS][DIM], newdimension[DIM];
	unsigned int size[BINS];

	unsigned char b;
	unsigned int i;
	unsigned int index=0;
	
  //Empty bins
	for(b=0;b<BINS;b++)
	{
		bin[b].clear();
	}

	//Bin points
	for(i=0;i<n;i++)
	{
		b=inverse[o][Bin(&locs[orders[i]*DIM],center)];
		bin[b].push_back(orders[i]);
	}

	//Reorder points by placing bins together in order
	for(b=0;b<BINS;b++)
	{
		size[b]=(unsigned int)bin[b].size();
		memcpy(&orders[index],&bin[b][0],sizeof(unsigned int)*size[b]);
		index+=size[b];
	}

	//Halve all dimensions
	for(int d=0;d<DIM;d++)
	{
		newdimension[d]=dimension[d]*.5;
	}

	BITS NextHistory;
	//recursivly call
	for(b=0;b<BINS;b++)
	{
		for(int d=0;d<DIM;d++)
		{
			newcenter[b][d]=center[d]+dimension[d]*.25*dir[order[o][b]][d];		
		}

		if(r==refinements)
		{
			NextHistory= ((history<<DIM)|b);
			//save history for each point in bucket
			for(unsigned int j=0;j<size[b];j++)
			{
				corders[j].bits=NextHistory;
				corders[j].i=orders[j];
			}
		}
		else if(size[b]>1)
		{
			NextHistory= ((history<<DIM)|b);
			SerialHR<BITS>(orders,corders,bin,size[b],newcenter[b],newdimension,orientation[o][b],r+1,NextHistory);
		}
		else if (size[b]==1)
		{
			NextHistory= ((history<<DIM)|b);

			LOCS *loc=&locs[orders[0]*DIM];
			REAL Clocs[DIM];
			REAL dims[DIM];
			int Co=orientation[o][b];

			for(int d=0;d<DIM;d++)
			{
				Clocs[d]=center[d]+dir[order[o][b]][d]*dimension[d]*.25;

				dims[d]=newdimension[d];
			}
			
			int ref=r;
			int b;
			for(;ref<refinements;ref++)
			{
				b=inverse[Co][Bin(loc,Clocs)];

				NextHistory<<=DIM;
				NextHistory|=b;

				for(int d=0;d<DIM;d++)
				{
					dims[d]*=.5;
					Clocs[d]=Clocs[d]+dir[order[o][b]][d]*dims[d]*.5;
				}
				Co=orientation[Co][b];
			}
			corders[0].bits=NextHistory;
			corders[0].i=orders[0];
		}
		corders+=size[b];
		orders+=size[b];
	}
}

template<int DIM, class LOCS> template<class BITS>
void SFC<DIM,LOCS>::Parallel()
{
	vector<History<BITS> > sbuf, rbuf, mbuf;
	unsigned int i;
	vector<unsigned int> n_per_proc(P);
	this->n_per_proc=&n_per_proc[0];	
  
  if(n!=0)
  {
	  sbuf.resize(n);
	  mbuf.resize(n);
	  sendbuf=(void*)&(sbuf[0]);
	  recievebuf=(void*)&(rbuf[0]);
	  mergebuf=(void*)&(mbuf[0]);
  }
  else
  {
    sendbuf=recievebuf=mergebuf=0;
  }

	//start sending n to every other processor
	//start recieving n from every other processor
	//or
	//preform allgather (threaded preferable)
#ifdef _TIMESFC_
	start=timer->currentSeconds();
#endif
	MPI_Allgather(&n,1,MPI_INT,&n_per_proc[0],1,MPI_INT,Comm);
#ifdef _TIMESFC_
	finish=timer->currentSeconds();
	gtime+=finish-start;
#endif

#ifdef _TIMESFC_
	start=timer->currentSeconds();
#endif
	SerialH<BITS>();	//Saves results in sendbuf
#ifdef _TIMESFC_
	finish=timer->currentSeconds();
	sertime+=finish-start;
#endif
	//find index start & max
	unsigned int max_n=n;
	unsigned int istart=0;

	for(int p=0;p<P;p++)
	{
		if(n_per_proc[p]>max_n)
			max_n=n_per_proc[p];

		if(p<rank)
			istart+=n_per_proc[p];
	}
	
	//make two pointers to internal buffers for a fast copy	
	History<BITS> *c=(History<BITS>*)sendbuf;

  for(i=1;i<n;i++)
  {
    if(c[i].bits<c[i-1].bits)
    {
      char filename[100];
      sprintf(filename,"sfcdebug%d.txt",rank);
      cout << "Error forming local curve: " << c[i].i << ":" << c[i].bits << " is less than " << c[i-1].i << ":" << c[i-1].bits << endl;
      cout << "Please email the file '" << filename << "' to luitjens@cs.utah.edu\n";
      ofstream sfcdebug(filename);
     
      sfcdebug << "Error forming local curve: " << c[i].i << ":" << c[i].bits << " is less than " << c[i-1].i << ":" << c[i-1].bits << endl;
      sfcdebug << "DIM:" << DIM << endl;
      sfcdebug << "Local Size:" << n << endl;
      sfcdebug << "Dimensions:";
      for(int d=0;d<DIM;d++)
              sfcdebug << dimensions[d] << " ";
      sfcdebug << endl;
      sfcdebug << "Center:";
      for(int d=0;d<DIM;d++)
              sfcdebug << center[d] << " ";
      sfcdebug << endl;
      sfcdebug << "Refinements:" << refinements << endl;
      
      sfcdebug << "LOCS:";
      for(unsigned int i=0;i<n;i++)
      {
        sfcdebug << "[";
        for(int d=0;d<DIM-1;d++)
        {
         sfcdebug << locs[i*DIM+d] << ", ";
        }
        sfcdebug << locs[i*DIM+DIM-1] << "] ";
      }
      sfcdebug << endl;
      throw SCIRun::InternalError("Error forming local curve\n",__FILE__,__LINE__);
    }
  }
  
  //increment indexies 
	for(i=0;i<n;i++)
	{
		c[i].i+=istart;	
	}
	

	//resize buffers
//	sbuf.resize(max_n);
	rbuf.resize(max_n);
//	mbuf.resize(max_n);

	sendbuf=(void*)&(sbuf[0]);
	recievebuf=(void*)&(rbuf[0]);
	mergebuf=(void*)&(mbuf[0]);

#ifdef _TIMESFC_
	start=timer->currentSeconds();
#endif
	PrimaryMerge<BITS>();
#ifdef _TIMESFC_
	finish=timer->currentSeconds();
	ptime+=finish-start;
#endif
	
	History<BITS> *ssbuf=(History<BITS>*)sendbuf;

#ifdef _TIMESFC_
	start=timer->currentSeconds();
#endif
	Cleanup<BITS>();
#ifdef _TIMESFC_
	finish=timer->currentSeconds();
	cleantime+=finish-start;
#endif
  
	ssbuf=(History<BITS>*)sendbuf;
	
	orders->resize(n);

	//make pointer to internal buffers for a fast copy	
	unsigned int* o=&(*orders)[0];
  c=(History<BITS>*)sendbuf;
		
	//copy permutation to orders
	for(i=0;i<n;i++)
	{
		//COPY to orders
		o[i]=c[i].i;	
	}
}

#define ASCENDING 0
#define DESCENDING 1
template<int DIM, class LOCS> template<class BITS>
int SFC<DIM,LOCS>::MergeExchange(int to)
{
	float inv_denom=1.0/sizeof(History<BITS>);
//	cout << rank <<  ": Merge Exchange started with " << to << endl;
	int direction= (int) (rank>to);
	BITS emax, emin;
	queue<MPI_Request> squeue, rqueue;
	unsigned int tag=0;
	unsigned int n2=n_per_proc[to];
  
  //temperary fix to prevent processors with no elements from crashing
  if(n==0 || n2==0)
     return 0;

	MPI_Request srequest, rrequest;
	MPI_Status status;
	
	History<BITS> *sbuf=(History<BITS>*)sendbuf, *rbuf=(History<BITS>*)recievebuf, *mbuf=(History<BITS>*)mergebuf;
	History<BITS> *msbuf=sbuf, *mrbuf=rbuf;
	
	//min_max exchange
	if(direction==ASCENDING)
	{
		emax=sbuf[n-1].bits;
		MPI_Isend(&emax,sizeof(BITS),MPI_BYTE,to,0,Comm,&srequest);
		MPI_Irecv(&emin,sizeof(BITS),MPI_BYTE,to,0,Comm,&rrequest);
	}
	else
	{
		emin=sbuf[0].bits;
		MPI_Isend(&emin,sizeof(BITS),MPI_BYTE,to,0,Comm,&srequest);
		MPI_Irecv(&emax,sizeof(BITS),MPI_BYTE,to,0,Comm,&rrequest);
	}
	MPI_Wait(&rrequest,&status);
	MPI_Wait(&srequest,&status);
	
	if(emax<emin)	//if exchange not needed 
	{
		return 0;
	}
//	cout << rank << ": Max-min done\n";
	
	unsigned int nsend=n_per_proc[rank];
	unsigned int nrecv=n_per_proc[to];
	//sample exchange
	unsigned int minn=min(n,n2);
	unsigned int sample_size=(int)(minn*sample_percent);

	if(sample_size>=5)
	{
//		cout << rank << " creating samples\n";
		BITS *highsample=(BITS*)mbuf, *lowsample=(BITS*)rbuf, *mysample, *theirsample;
		float stridelow,stridehigh,mystride;
		unsigned int index=0, ihigh=0,ilow=0,count=0;
		if(direction==ASCENDING)
		{
			mysample=lowsample;
			theirsample=highsample;
			mystride=stridelow=n/(float)sample_size;
			stridehigh=n2/(float)sample_size;
		}
		else
		{
			mysample=highsample;
			theirsample=lowsample;
			stridelow=n2/(float)sample_size;
			mystride=stridehigh=n/(float)sample_size;
		}
	
		//create sample
		for(unsigned int i=0;i<sample_size;i++)
		{
			index=int(mystride*i);
			mysample[i]=sbuf[index].bits;
		}
//		cout << "exchanging samples\n";
		//exchange samples
		MPI_Isend(mysample,sample_size*sizeof(BITS),MPI_BYTE,to,1,Comm,&srequest);
		MPI_Irecv(theirsample,sample_size*sizeof(BITS),MPI_BYTE,to,1,Comm,&rrequest);
	
		MPI_Wait(&rrequest,&status);
		MPI_Wait(&srequest,&status);
		
//		cout << "done exchanging samples\n";
		//merge samples
	
		while(count<minn)
		{
			if(lowsample[ilow]<=highsample[ihigh])
			{
				ilow++;
			}
			else
			{
				ihigh++;
			}
			count=int(ilow*stridelow)+int(ihigh*stridehigh);
		}
		
		if(ilow>sample_size) //handle case where ilow goes to far
		{
			ihigh+=(ilow-sample_size);
		}
		nrecv=nsend=int((ihigh+2)*stridehigh);

		if(nsend>n)
		{
			nsend=n;
		}
		if(nrecv>n2)
		{
			nrecv=n2;
		}
	}	
	//final exchange
//	cout << rank << ": sample done\n";
	
	int b;
	unsigned int block_count=0;
	int sremaining=nsend;
	int rremaining=nrecv;
//	cout << sremaining << " " << rremaining << endl;
	unsigned int sent=0, recvd=0, merged=0;
//	cout << rank << " Block size: " << block_size << endl;	
	if(direction==ASCENDING)
	{
		//Merge Ascending
		//Send Descending
		//Recieve Ascending
		
		//position buffers
		sbuf+=n;
		
		while(block_count<blocks_in_transit)
		{
			//send
			if(sremaining>=(int)block_size)
			{
				sbuf-=block_size;
				MPI_Isend(sbuf,block_size*sizeof(History<BITS>),MPI_BYTE,to,tag,Comm,&srequest);
				squeue.push(srequest);
				sent+=block_size;
//				cout << rank << ": Sending block of size: " << block_size << endl;
				sremaining-=block_size;
			}
			else if(sremaining>0)
			{
				sbuf-=sremaining;
				MPI_Isend(sbuf,sremaining*sizeof(History<BITS>),MPI_BYTE,to,tag,Comm,&srequest);
				squeue.push(srequest);
				sent+=sremaining;
//				cout << rank << ": Sending block of size: " << sremaining << endl;
				sremaining=0;
			}
			
			//recieve
			if(rremaining>=(int)block_size)
			{
				MPI_Irecv(rbuf+recvd,block_size*sizeof(History<BITS>),MPI_BYTE,to,tag,Comm,&rrequest);
				rqueue.push(rrequest);
				recvd+=block_size;
//				cout << rank << ": Recieving block of size: " << block_size << endl;
				rremaining-=block_size;
			}
			else if(rremaining>0)
			{
				MPI_Irecv(rbuf+recvd,rremaining*sizeof(History<BITS>),MPI_BYTE,to,tag,Comm,&rrequest);
				rqueue.push(rrequest);
				recvd+=rremaining;

//				cout << rank << ": Recieving block of size: " << rremaining << endl;
				rremaining=0;
			}
		
			block_count++;
			tag++;
		}
		while(!rqueue.empty())
		{
			MPI_Wait(&(rqueue.front()),&status);
			rqueue.pop();
			
			MPI_Get_count(&status,MPI_BYTE,&b);
			b=int(b*inv_denom);
//			cout << rank << " recieved block of size\n";
			while(b>0 && merged<n)
			{
				if(mrbuf[0].bits<msbuf[0].bits)
				{
					//pull from recieve buffer
					mbuf[merged]=mrbuf[0];
					mrbuf++;
					b--;
				}
				else
				{
					//pull from send buffer
					mbuf[merged]=msbuf[0];
					msbuf++;
				}
				merged++;
			}

			//send next block
			if(sremaining>=(int)block_size)
			{
				sbuf-=block_size;
				MPI_Isend(sbuf,block_size*sizeof(History<BITS>),MPI_BYTE,to,tag,Comm,&srequest);
				squeue.push(srequest);
				sent+=block_size;
				sremaining-=block_size;
			}
			else if(sremaining>0)
			{
				sbuf-=sremaining;
				MPI_Isend(sbuf,sremaining*sizeof(History<BITS>),MPI_BYTE,to,tag,Comm,&srequest);
				squeue.push(srequest);
				sent+=sremaining;
				sremaining=0;
			}
			
			//recieve
			if(rremaining>=(int)block_size)
			{
				MPI_Irecv(rbuf+recvd,block_size*sizeof(History<BITS>),MPI_BYTE,to,tag,Comm,&rrequest);
				rqueue.push(rrequest);
				recvd+=block_size;
				rremaining-=block_size;
			}
			else if(rremaining>0)
			{
				MPI_Irecv(rbuf+recvd,rremaining*sizeof(History<BITS>),MPI_BYTE,to,tag,Comm,&rrequest);
				rqueue.push(rrequest);
				recvd+=rremaining;
				rremaining=0;
			}
		
			block_count++;
			tag++;
		
			//wait for a send, it should be done now
			MPI_Wait(&(squeue.front()),&status);
			squeue.pop();
		}

		if(merged<n)	//merge additional elements from send buff
		{
			memcpy(mbuf+merged,msbuf,(n-merged)*sizeof(History<BITS>));
//			cout << rank << " (ASC) merging more from " << to << endl;
		}
	}
	else
	{
		//Merge Descending
		//Send Ascending
		//Recieve Descending

		//position buffers
		mbuf+=n;
		rbuf+=n2;

		msbuf+=n-1;
		mrbuf+=n2-1;

			
		while(block_count<blocks_in_transit)
		{
			//send
			if(sremaining>=(int)block_size)
			{
//				cout << rank << " sending block of size " << block_size << endl;
				MPI_Isend(sbuf+sent,block_size*sizeof(History<BITS>),MPI_BYTE,to,tag,Comm,&srequest);
				squeue.push(srequest);
				sent+=block_size;
				sremaining-=block_size;
			}
			else if(sremaining>0)
			{
//				cout << rank << " sending block of size " << sremaining << endl;
				MPI_Isend(sbuf+sent,sremaining*sizeof(History<BITS>),MPI_BYTE,to,tag,Comm,&srequest);
				squeue.push(srequest);
				sent+=sremaining;
				sremaining=0;
			}

			if(rremaining>=(int)block_size)
			{
//				cout << rank << " recieving block of size " << block_size << endl;
				rbuf-=block_size;
				MPI_Irecv(rbuf,block_size*sizeof(History<BITS>),MPI_BYTE,to,tag,Comm,&rrequest);
				rqueue.push(rrequest);
				recvd+=block_size;
				rremaining-=block_size;
			}
			else if(rremaining>0)
			{
//				cout << rank << " recieving block of size " << rremaining << endl;
				rbuf-=rremaining;
				MPI_Irecv(rbuf,rremaining*sizeof(History<BITS>),MPI_BYTE,to,tag,Comm,&rrequest);
				rqueue.push(rrequest);
				recvd+=rremaining;
				rremaining=0;
			}
			block_count++;
			tag++;
		}
		while(!rqueue.empty())
		{
			MPI_Wait(&(rqueue.front()),&status);
			rqueue.pop();

			MPI_Get_count(&status,MPI_BYTE,&b);
			b=int(b*inv_denom);
//			cout << rank << " recieved block of size\n";
			while(b>0 && merged<n)
			{
				if(mrbuf[0].bits>msbuf[0].bits) //merge from recieve buff
				{
					mbuf--;
					mbuf[0]=mrbuf[0];
					mrbuf--;
					b--;
				}
				else	//merge from send buff
				{
					mbuf--;
					mbuf[0]=msbuf[0];
					msbuf--;
				}
				merged++;
			}
			
			//send more if needed
			if(sremaining>=(int)block_size)
			{
//				cout << rank << " sending block of size " << block_size << endl;
				MPI_Isend(sbuf+sent,block_size*sizeof(History<BITS>),MPI_BYTE,to,tag,Comm,&srequest);
				squeue.push(srequest);
				sent+=block_size;
				sremaining-=block_size;
			}
			else if(sremaining>0)
			{
//				cout << rank << " sending block of size " << sremaining << endl;
				MPI_Isend(sbuf+sent,sremaining*sizeof(History<BITS>),MPI_BYTE,to,tag,Comm,&srequest);
				squeue.push(srequest);
				sent+=sremaining;
				sremaining=0;
			}
			if(rremaining>=(int)block_size)
			{
//				cout << rank << " recieving block of size " << block_size << endl;
				rbuf-=block_size;
				MPI_Irecv(rbuf,block_size*sizeof(History<BITS>),MPI_BYTE,to,tag,Comm,&rrequest);
				rqueue.push(rrequest);
				recvd+=block_size;
				rremaining-=block_size;
			}
			else if(rremaining>0)
			{
//				cout << rank << " recieving block of size " << rremaining << endl;
				rbuf-=rremaining;
				MPI_Irecv(rbuf,rremaining*sizeof(History<BITS>),MPI_BYTE,to,tag,Comm,&rrequest);
				rqueue.push(rrequest);
				recvd+=rremaining;
				rremaining=0;
			}
			tag++;

			MPI_Wait(&(squeue.front()),&status);
			squeue.pop();
		}
		if(merged<n) //merge additional elements off of msbuf
		{
			int rem=n-merged;
			memcpy(mbuf-rem,msbuf-rem+1,(rem)*sizeof(History<BITS>));
//			cout << rank << " (DSC) merging more from " << to << endl;
		
			/*
			(while(merged<n)
			{
				mbuf--;
				mbuf[0]=msbuf[0];
				msbuf--;
				merged++;
			}
			*/
		}
	}
	while(!rqueue.empty())
	{
		MPI_Wait(&(rqueue.front()),&status);
		rqueue.pop();
//		cout << rank << " recieved left over block\n";
	}
	while(!squeue.empty())
	{
		MPI_Wait(&(squeue.front()),&status);
		squeue.pop();
//		cout << rank << " sent left over block\n";
	}
	
	swap(mergebuf,sendbuf);
	
	sbuf=(History<BITS>*)sendbuf;
	return 1;
}

struct HC_MERGE
{
	unsigned int base;
	unsigned int P;
};

template<int DIM, class LOCS> template<class BITS>
void SFC<DIM,LOCS>::PrimaryMerge()
{
	queue<HC_MERGE> q;
	HC_MERGE cur;
	bool send;
	int to=-1;
	cur.base=0;
	cur.P=P;
	q.push(cur);
	while(!q.empty())
	{
		int base, P;
		cur=q.front();
		q.pop();

		base=cur.base;
		P=cur.P;
		send=false;
		if(rank>=base && rank<base+(P>>1))
		{
			send=true;
			to=rank+((P+1)>>1);
		}
		else if(rank-((P+1)>>1)>=base && rank-((P+1)>>1)<base+(P>>1))
		{
			send=true;
			to=rank-((P+1)>>1);
		}

		if(send)
		{
			MergeExchange<BITS>(to);
		}

		//make next stages

		cur.P=((P+1)>>1);
		if(cur.P>1)
		{
			cur.base=base+(P>>1);
			q.push(cur);
		}

		cur.P=P-((P+1)>>1);
		if(cur.P>1)
		{
			cur.base=base;
			q.push(cur);
		}
	}

}

template<int DIM, class LOCS> template<class BITS>
void SFC<DIM,LOCS>::Cleanup()
{
	switch(cleanup)
	{
		case BATCHERS:
			Batchers<BITS>();
			break;
		case LINEAR:
			Linear<BITS>();
			break;
	};
}
template<int DIM, class LOCS> template <class BITS>
void SFC<DIM,LOCS>::Batchers()
{
	int p, r, t, q, d;

	t=1;

	while(t<P)
		t<<=1;

	p=t>>1;
	for(;p>=1;p>>=1)
	{
		q=t>>1;
		r=0;
		d=p;

		bool more;
		do
		{
			more=false;

			if(rank<P-d && (rank&p)==r)
			{
				MergeExchange<BITS>(rank+d);
			}
			else if(rank-d>=0 && ((rank-d)&p)==r)
			{
				MergeExchange<BITS>(rank-d);
			}
			if(q!=p)
			{
				more=true;
				d=q-p;
				q>>=1;
				r=p;
			}
		}while(more);
	}
}

template<int DIM, class LOCS> template <class BITS>
void SFC<DIM,LOCS>::Linear()
{
	unsigned int i=1, c=1, val=0;
	int mod=(int)ceil(log((float)P)/log(3.0f));

	while(c!=0)
	{
		val=0;
		if(rank%2==0)	//exchange right then left
		{
			if(rank!=P-1)
			{
				val+=MergeExchange<BITS>(rank+1);	
			}

			if(rank!=0)
			{
				val+=MergeExchange<BITS>(rank-1);
			}
			
		}
		else	//exchange left then right
		{
			if(rank!=0)
			{
				val+=MergeExchange<BITS>(rank-1);
			}
			
			if(rank!=P-1)
			{
				val+=MergeExchange<BITS>(rank+1);	
			}
		}
		i++;

		if(i%mod==0)
		{
			MPI_Allreduce(&val,&c,1,MPI_INT,MPI_MAX,Comm);
		}
	}
	

}
template<int DIM, class LOCS>
void SFC<DIM,LOCS>::SetMergeParameters(unsigned int block_size, unsigned int blocks_in_transit, float sample_percent)
{
	this->block_size=block_size;
	this->blocks_in_transit=blocks_in_transit;
	this->sample_percent=sample_percent;
}
template<int DIM, class LOCS>
void SFC<DIM,LOCS>::SetRefinements(int refinements)
{
	this->refinements=refinements;
	set=set|32;
}

template<int DIM, class LOCS>
void SFC<DIM,LOCS>::SetLocalSize(unsigned int n)
{
	this->n=n;
	set=set|4;
}


template<int DIM, class LOCS>
void SFC<DIM,LOCS>::SetLocations(vector<LOCS> *locsv)
{
	if(locsv!=0)
	{
		this->locsv=locsv;
		this->locs=&(*locsv)[0];
		set=set|1;
	}
}

template<int DIM, class LOCS>
void SFC<DIM,LOCS>::SetOutputVector(vector<unsigned int> *orders)
{
	if(orders!=0)
	{
		this->orders=orders;
		set=set|2;
	}
}

template<class LOCS>
void SFC1D<LOCS>::SetDimensions(REAL wx)
{
  SFC<1,LOCS>::dimensions[0]=wx;
  SFC<1,LOCS>::set|=8;
}

template<class LOCS>
void SFC1D<LOCS>::SetCenter(REAL x)
{
  SFC<1,LOCS>::center[0]=x;
  SFC<1,LOCS>::set|=16;
}

template<class LOCS>
void SFC1D<LOCS>::SetRefinementsByDelta(REAL deltax)
{
	char mask=8;
	if( (mask&SFC<1,LOCS>::set) != mask)
	{
		cout << "SFC Error: Cannot set refinements by delta until dimensions have been set\n";
	}
	SFC<1,LOCS>::refinements=(int)ceil(log(SFC<1,LOCS>::dimensions[0]/deltax)/log(2.0));
	SFC<1,LOCS>::set|=32;
}
template<class LOCS>
void SFC2D<LOCS>::SetCurve(Curve curve)
{
	switch(curve)
	{
		case HILBERT:
			SFC<2,LOCS>::order=horder2;
			SFC<2,LOCS>::orientation=horient2;
			SFC<2,LOCS>::inverse=hinv2;
		  break;
		case MORTON:
			SFC<2,LOCS>::order=morder2;
		  SFC<2,LOCS>::orientation=morient2;
			SFC<2,LOCS>::inverse=morder2;
			break;
		case GREY:
			SFC<2,LOCS>::order=gorder2;
		  SFC<2,LOCS>::orientation=gorient2;
			SFC<2,LOCS>::inverse=ginv2;
		  break;
	}
}
		 
template<class LOCS>
void SFC2D<LOCS>::SetDimensions(REAL wx, REAL wy)
{
	SFC<2,LOCS>::dimensions[0]=wx;
	SFC<2,LOCS>::dimensions[1]=wy;
	SFC<2,LOCS>::set|=8;
}

template<class LOCS>
void SFC2D<LOCS>::SetCenter(REAL x, REAL y)
{
	SFC<2,LOCS>::center[0]=x;
	SFC<2,LOCS>::center[1]=y;
	SFC<2,LOCS>::set|=16;
}

template<class LOCS>
void SFC2D<LOCS>::SetRefinementsByDelta(REAL deltax, REAL deltay)
{
	char mask=8;
	if( (mask&SFC<2,LOCS>::set) != mask)
	{
		cout << "SFC Error: Cannot set refinements by delta until dimensions have been set\n";
	}
	SFC<2,LOCS>::refinements=(int)ceil(log(SFC<2,LOCS>::dimensions[0]/deltax)/log(2.0));
	SFC<2,LOCS>::refinements=max(SFC<2,LOCS>::refinements,(int)ceil(log(SFC<2,LOCS>::dimensions[1]/deltay)/log(2.0)));
	SFC<2,LOCS>::set=SFC<2,LOCS>::set|32;

}

template<class LOCS>
unsigned char SFC1D<LOCS>::Bin(LOCS *point, REAL *center)
{
  return point[0]<center[0];
} 
	
template<class LOCS>
unsigned char  SFC2D<LOCS>::Bin(LOCS *point, REAL *center)
{

	unsigned char bin=0;
	
	if(point[0]<center[0])
		bin|=1;

	if(point[1]<center[1])
		bin|=2;

	return bin;

}
/*****************SFC3D*********************/
template<class LOCS>
void SFC3D<LOCS>::SetCurve(Curve curve)
{
	switch(curve)
	{
		case HILBERT:
			SFC<3,LOCS>::order=horder3;
		        SFC<3,LOCS>::orientation=horient3;
			SFC<3,LOCS>::inverse=hinv3;
		      	break;
		case MORTON:
			SFC<3,LOCS>::order=morder3;
		        SFC<3,LOCS>::orientation=morient3;
			SFC<3,LOCS>::inverse=morder3;
			break;
		case GREY:
			SFC<3,LOCS>::order=gorder3;
		        SFC<3,LOCS>::orientation=gorient3;
			SFC<3,LOCS>::inverse=ginv3;
		        break;
	}
}
template<class LOCS>
void SFC3D<LOCS>::SetDimensions(REAL wx, REAL wy, REAL wz)
{
	SFC<3,LOCS>::dimensions[0]=wx;
	SFC<3,LOCS>::dimensions[1]=wy;
	SFC<3,LOCS>::dimensions[2]=wz;
	SFC<3,LOCS>::set=SFC<3,LOCS>::set|8;
}
template<class LOCS>
void SFC3D<LOCS>::SetCenter(REAL x, REAL y, REAL z)
{
	SFC<3,LOCS>::center[0]=x;
	SFC<3,LOCS>::center[1]=y;
	SFC<3,LOCS>::center[2]=z;

	SFC<3,LOCS>::set=SFC<3,LOCS>::set|16;
}
template<class LOCS>
void SFC3D<LOCS>::SetRefinementsByDelta(REAL deltax, REAL deltay, REAL deltaz)
{
	char mask=8;
	if( (mask&SFC<3,LOCS>::set) != mask)
	{
		cout << "SFC Error: Cannot set refinements by delta until dimensions have been set\n";
	}

	SFC<3,LOCS>::refinements=(int)ceil(log(SFC<3,LOCS>::dimensions[0]/deltax)/log(2.0));
	SFC<3,LOCS>::refinements=max(SFC<3,LOCS>::refinements,(int)ceil(log(SFC<3,LOCS>::dimensions[1]/deltay)/log(2.0)));
	SFC<3,LOCS>::refinements=max(SFC<3,LOCS>::refinements,(int)ceil(log(SFC<3,LOCS>::dimensions[2]/deltaz)/log(2.0)));
	SFC<3,LOCS>::set=SFC<3,LOCS>::set|32;
}
template<class LOCS>
unsigned char  SFC3D<LOCS>::Bin(LOCS *point, REAL *center)
{
	unsigned char bin=0;
	
	if(point[0]>=center[0])
		bin|=4;

	if(point[1]<center[1])
		bin|=2;

	if(point[2]<center[2])
		bin|=1;

	return bin;
}

} //End Namespace Uintah
#endif
