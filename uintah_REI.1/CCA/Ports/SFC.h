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
const int TIMERS=6;
double timers[TIMERS]={0};
#endif
enum Curve {HILBERT, MORTON, GREY};
enum CleanupType{BATCHERS,LINEAR};

#define SERIAL 1
#define PARALLEL 2
struct DistributedIndex
{
  unsigned int i;
  unsigned int p;
  DistributedIndex(int index, int processor)
  {
     i=index;
     p=processor;
  }
  DistributedIndex() {}
};
template<class BITS>
struct History
{
	DistributedIndex index;
	BITS bits;
};
struct Group
{
  int start_rank;
  int size;
  int partner_group;
  int parent_group;
  
  Group(int start, int s, int partner,int parent)
  {
    start_rank=start;
    size=s;
    partner_group=partner;
    parent_group=parent;
  }
  Group() {}
};
template <class BITS>
inline bool operator<=(const History<BITS> &a, const History<BITS> &b)
{
  return a.bits<=b.bits;
}
template <class BITS>
inline bool operator<(const History<BITS> &a, const History<BITS> &b)
{
  return a.bits<b.bits;
}
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
	SFC(int dir[][DIM], const ProcessorGroup *d_myworld) : dir(dir),set(0), locsv(0), locs(0), orders(0), d_myworld(d_myworld), block_size(3000), blocks_in_transit(3), sample_percent(.1), cleanup(BATCHERS), mergemode(0) {};
	virtual ~SFC() {};
	void GenerateCurve(int mode=0);
	void SetRefinements(int refinements);
	void SetLocalSize(unsigned int n);
	void SetLocations(vector<LOCS> *locs);
	void SetOutputVector(vector<DistributedIndex> *orders);
	void SetMergeParameters(unsigned int block_size, unsigned int blocks_in_transit, float sample_percent);
	void SetBlockSize(unsigned int b) {block_size=b;};
	void SetBlocksInTransit(unsigned int b) {blocks_in_transit=b;};
	void SetSamplePercent(float p) {sample_percent=p;};
	void SetCleanup(CleanupType cleanup) {this->cleanup=cleanup;};
  void SetMergeMode(int mode) {this->mergemode=mode;};

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
	vector<DistributedIndex> *orders;
	
	const ProcessorGroup *d_myworld;
	
	//Merge-Exchange Parameters
	unsigned int block_size;
	unsigned int blocks_in_transit;
	float sample_percent;
	
	CleanupType cleanup;
	
	int rank, P;
	MPI_Comm Comm;	

  //Parllel2 Parameters
  unsigned int buckets;
  int b;
  vector<int> histograms;
  vector<int> cuts;
  int mergemode;
  
	void Serial();
	void SerialR(DistributedIndex* orders,vector<DistributedIndex> *bin, unsigned int n, REAL *center, REAL *dimension, unsigned int o=0);

  template<class BITS> void SerialH(History<BITS> *histories);
	template<class BITS> void SerialHR(DistributedIndex* orders,History<BITS>* corders,vector<DistributedIndex> *bin, unsigned int n, REAL *center, REAL *dimension,                                     unsigned int o=0, int r=1, BITS history=0);
	template<class BITS> void Parallel();
	template<class BITS> void Parallel0();
	template<class BITS> void Parallel1();
	template<class BITS> void Parallel2();
  
  template<class BITS> void CalculateHistogramsAndCuts(vector<History<BITS> > &histories);
	template<class BITS> void ComputeLocalHistogram(int *histogram,vector<History<BITS> > &histories);
	
  template<class BITS> int MergeExchange(int to,vector<History<BITS> > &histories, vector<History<BITS> >&rbuf, vector<History<BITS> > &mbuf);	
	template<class BITS> void PrimaryMerge(vector<History<BITS> > &histories, vector<History<BITS> >&rbuf, vector<History<BITS> > &mbuf);
	template<class BITS> void Cleanup(vector<History<BITS> > &histories, vector<History<BITS> >&rbuf, vector<History<BITS> > &mbuf);
	template<class BITS> void Batchers(vector<History<BITS> > &histories, vector<History<BITS> >&rbuf, vector<History<BITS> > &mbuf);
	template<class BITS> void Linear(vector<History<BITS> > &histories, vector<History<BITS> >&rbuf, vector<History<BITS> > &mbuf);

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
        if(rank==0)
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

	  DistributedIndex *o=&(*orders)[0];
	
	  for(unsigned int i=0;i<n;i++)
	  {
		  o[i]=DistributedIndex(i,0);
	  }

	  vector<DistributedIndex> bin[BINS];
	  for(int b=0;b<BINS;b++)
	  {
		  bin[b].reserve(n/BINS);
	  }
	  //Recursive call
	  SerialR(o,bin,n,center,dimensions);
  }
}

template<int DIM, class LOCS> template<class BITS>
void SFC<DIM,LOCS>::SerialH(History<BITS> *histories)
{
  if(n!=0)
  {
	  orders->resize(n);

	  DistributedIndex *o=&(*orders)[0];
	
	  for(unsigned int i=0;i<n;i++)
	  {
		  o[i]=DistributedIndex(i,rank);
	  }

	  vector<DistributedIndex> bin[BINS];
	  for(int b=0;b<BINS;b++)
	  {
		  bin[b].reserve(n/BINS);
	  }
	  //Recursive call
	  SerialHR<BITS>(o,histories,bin,n,center,dimensions);
  }
}

template<int DIM, class LOCS> 
void SFC<DIM,LOCS>::SerialR(DistributedIndex* orders,vector<DistributedIndex> *bin, unsigned int n, REAL *center, REAL *dimension, unsigned int o)
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
		b=Bin(&locs[orders[i].i*DIM],center);
		bin[inverse[o][b]].push_back(orders[i]);
	}

	//Reorder points by placing bins together in order
	for(b=0;b<BINS;b++)
	{
		size[b]=(unsigned int)bin[b].size();
		memcpy(&orders[index],&bin[b][0],sizeof(DistributedIndex)*size[b]);
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
			memcpy(l,&locs[orders[0].i*DIM],sizeof(LOCS)*DIM);
			i=1;
			while(same && i<n)
			{
				for(int d=0;d<DIM;d++)
				{
					if(l[d]-locs[orders[i].i*DIM+d]>EPSILON || l[d]-locs[orders[i].i*DIM+d]<-EPSILON)
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
void SFC<DIM,LOCS>::SerialHR(DistributedIndex* orders, History<BITS>* corders,vector<DistributedIndex> *bin, unsigned int n, REAL *center, REAL *dimension, unsigned int o, int r, BITS history)
{
	REAL newcenter[BINS][DIM], newdimension[DIM];
	unsigned int size[BINS];

	unsigned char b;
	unsigned int i;
	unsigned int index=0;

  if(n==1)
  {
    REAL newcenter[DIM];
    //initialize newdimension and newcenter
    for(int d=0;d<DIM;d++)
    {
      newdimension[d]=dimension[d];
      newcenter[d]=center[d];
    }
    for(;r<=refinements;r++)
    {
      //calculate history
      b=inverse[o][Bin(&locs[orders[0].i*DIM],newcenter)];
      
      //halve all dimensions and calculate newcenter
      for(int d=0;d<DIM;d++)
      {
        newcenter[d]=newcenter[d]+newdimension[d]*.25*dir[order[o][b]][d];
        newdimension[d]*=.5;
      }
      //update orientation
      o=orientation[o][b];

      //set history
      history=((history<<DIM)|b);
    }
    corders[0].bits=history;
    corders[0].index=orders[0];
    return;
  }  

  //Empty bins
	for(b=0;b<BINS;b++)
	{
		bin[b].clear();
	}

	//Bin points
	for(i=0;i<n;i++)
	{
		b=inverse[o][Bin(&locs[orders[i].i*DIM],center)];
		bin[b].push_back(orders[i]);
	}

	//Reorder points by placing bins together in order
	for(b=0;b<BINS;b++)
	{
		size[b]=(unsigned int)bin[b].size();
		memcpy(&orders[index],&bin[b][0],sizeof(DistributedIndex)*size[b]);
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
				corders[j].index=orders[j];
			}
		}
		else if(size[b]>=1)
		{
			NextHistory= ((history<<DIM)|b);
			SerialHR<BITS>(orders,corders,bin,size[b],newcenter[b],newdimension,orientation[o][b],r+1,NextHistory);
		}
		corders+=size[b];
		orders+=size[b];
	}
}

template<int DIM, class LOCS> template<class BITS>
void SFC<DIM,LOCS>::ComputeLocalHistogram(int *histogram,vector<History<BITS> > &histories)
{
  //initialize to zero
  for(unsigned int i=0;i<buckets;i++)
  {
    histogram[i]=0;
  }
  //compute bit mask
  int mask=0;
  for(int i=0;i<b;i++)
  {
    mask<<=1;
    mask|=1;
  } 
  int shift=refinements*DIM-b;
  mask<<=shift;
  
  for(unsigned int i=0;i<n;i++)
  {
    int bucket=(histories[i].bits&mask) >> shift;
    
    histogram[bucket]++;
  }

}
template<int DIM, class LOCS> template<class BITS>
void SFC<DIM,LOCS>::CalculateHistogramsAndCuts(vector<History<BITS> > &histories)
{
  float max_imbalance;
  //calculate b
  b=(int)ceil(log(2.0*P)/log(2.0));
  if(b>refinements*DIM)
      b=refinements*DIM;
 
  cuts.resize(P+1); 
  do
  {
    //calcualte buckets
    buckets=1<<(b);
    
    //resize histograms
    histograms.resize( (P+1)*buckets );

    //compute histograms
    ComputeLocalHistogram<BITS>(&histograms[ rank*(buckets) ],histories);
    
    //all gather histograms
    MPI_Allgather(&histograms[rank*(buckets)],buckets,MPI_INT,&histograms[0],buckets,MPI_INT,Comm);        

    //sum histogram
    int *sum=&histograms[P*buckets];
    int N=0;
    for(unsigned int i=0;i<buckets;i++) 
    {
      sum[i]=0;
      for(int p=0;p<P;p++)
      {
        sum[i]+=histograms[p*buckets+i];
      }
      N+=sum[i];
    } 
    
    //calculate cut points 
    float mean=float(N)/P;
    float target=mean;
    float remaining=N;
    
    for(int p=0;p<P;p++)
      cuts[p]=0;
    int current=0;
    int p=1;
    max_imbalance=0; 
    for(unsigned int bucket=0;bucket<buckets;bucket++)
    {
      if(current+sum[bucket]>1.2*target && current>.8*target)
      {
        //move to the next proc
        float imbalance=fabs(current/mean-1);
        if(imbalance>max_imbalance)
          max_imbalance=imbalance;
        
        cuts[p]=bucket;
        remaining-=current;
        target=remaining/(P-p);
        p++;
       
        current=sum[bucket];
      }
      else
      {
        current+=sum[bucket];
      }
    }
    cuts[p]=buckets;
   
    //increase b
    b+=DIM;
    /*
    if(max_imbalance>.15 && b<refinements*DIM)
    {
	cout << "repeat: " << max_imbalance << "P:" << P << " b:" << b << " buckets:" << buckets << endl;
    }
    */
  }while(max_imbalance>.15 && b<refinements*DIM);
}
template<int DIM, class LOCS> template<class BITS>
void SFC<DIM,LOCS>::Parallel()
{
  switch (mergemode)
  {
    case 0:
            Parallel0<BITS>();
            break;
    case 1: case 2: case 3:
            Parallel1<BITS>();
            break;
    case 4: 
            Parallel2<BITS>();
            break;
    default:
            if(rank==0)
            {
              cout << "Error invalid merge mode\n";
            }
            exit(0);
            break;
  }
}
template<int DIM, class LOCS> template<class BITS>
void SFC<DIM,LOCS>::Parallel2()
{
  int total_recvs=0;
  int num_recvs=0;
#ifdef _TIMESFC_
  start=timer->currentSeconds();
#endif
  vector<History<BITS> > myhistories(n),recv_histories(n),merge_histories(n),temp_histories(n);

  //calculate local curves 
  SerialH<BITS>(&myhistories[0]);	//Saves results in sendbuf
  
  /*
  cout << rank << ": histories:";
  for(unsigned int i=0;i<n;i++)
  {
    cout << (int)myhistories[i].bits << " ";
  } 
  cout << endl;
  */
  
  //calculate b
  b=(int)ceil(log(4.0*P)/log(2.0));
  if(b>refinements*DIM)
      b=refinements*DIM;

  //calcualte buckets
  buckets=1<<b;
  //cout << rank << ": bits for histogram:" << b << " buckets:" << buckets << endl;
  
  //create local histogram and cuts
  vector <int> histogram(buckets+P+1);
  vector <int> recv_histogram(buckets+P+1);
  vector <int> sum_histogram(buckets+P+1);
  vector <int> next_recv_histogram(buckets+P+1);

  histogram[buckets]=0;
  histogram[buckets+1]=buckets;
 
  //cout << rank << ": creating histogram\n";
  ComputeLocalHistogram<BITS>(&histogram[0], myhistories);
  //cout << rank << ": done creating histogram\n";

  /*
  cout << rank << ": local histogram: ";
  for(unsigned int i=0;i<buckets;i++)
          cout << histogram[i] << " ";
  cout << endl;
  */
  
#ifdef _TIMESFC_
  finish=timer->currentSeconds();
  timers[0]+=finish-start;
  start=timer->currentSeconds();
#endif 

 
  //merging phase
  int stages=0;
  for(int p=1;p<P;p<<=1,stages++);
  
  //create groups
  vector<vector<Group> > groups(stages+1);
  
  groups[0].push_back(Group(0,P,-1,-1));

  //cout << rank << ": creating merging groups\n";
  for(int stage=0;stage<stages;stage++)
  {
    for(unsigned int g=0;g<groups[stage].size();g++)
    {
        Group current=groups[stage][g];
        Group left,right;
       
        left.parent_group=right.parent_group=g;
        
        right.size=current.size/2;
        left.size=current.size-right.size;
        
        right.partner_group=groups[stage+1].size();
        left.partner_group=groups[stage+1].size()+1;
        
         
        left.start_rank=current.start_rank;
        right.start_rank=left.start_rank+left.size;
  
        if(right.size!=0)
        {
          groups[stage+1].push_back(left);
          groups[stage+1].push_back(right);
        }
        else
        {
          left.partner_group=-1;
          groups[stage+1].push_back(left);
        }
    }
  } 
  /*
  if(rank==0)
  {
    cout << "groups:" << endl;
    for(int stage=stages;stage>=0;stage--)
    {
      cout << "Stage Begin\n";
      for(unsigned int g=0;g<groups[stage].size();g++)
      {
         cout << "groups[" << g <<  "]: size:" << groups[stage][g].size << " start_rank:" << groups[stage][g].start_rank << " partner group:" << groups[stage][g].partner_group << " parent group:" << groups[stage][g].parent_group << endl;
      }
      cout << "Stage End\n";
    }
    cout << endl;
  }
  MPI_Barrier(Comm);
  //*/
  
  
  vector<MPI_Request> hsreqs;
  MPI_Request rreq;
 
  //initialize next groups 
  Group next_group=groups[stages][rank];
  Group next_partner_group, next_parent_group;
  int next_local_rank=-1;
  int next_partner_rank=-1;
  
  //place histgram into sum_histogram
  histogram.swap(sum_histogram);
     
#ifdef _TIMESFC_
  finish=timer->currentSeconds();
  timers[1]+=finish-start;
  start=timer->currentSeconds();
#endif

  //start first histogram send
  if(next_group.partner_group!=-1)
  {
    //set next groups
    next_parent_group=groups[stages-1][next_group.parent_group];
    next_local_rank=rank-next_group.start_rank;
    
    if(next_group.partner_group!=-1)
    {
    //set next status
      next_partner_group=groups[stages][next_group.partner_group];
      next_partner_rank=next_partner_group.start_rank+next_local_rank;

    //start sending histogram
      if(next_local_rank<next_partner_group.size)
      {
        //cout << rank << ": sending and recieving from: " << next_partner_rank << endl;
        MPI_Request request;
          
        //start send
        MPI_Isend(&sum_histogram[0],buckets+next_group.size+1,MPI_INT,next_partner_rank,stages,Comm,&request);
        hsreqs.push_back(request);
        //start recv
        //start send
        MPI_Irecv(&next_recv_histogram[0],buckets+next_partner_group.size+1,MPI_INT,next_partner_rank,stages,Comm,&rreq);
      }
      else
      {
        //partner doesn't exist
        //no send needed

        //recieve from rank 0 in partner group
        //cout << rank << ": recieving from: " << next_partner_group.start_rank << endl;
        //start send
        MPI_Irecv(&next_recv_histogram[0],buckets+next_partner_group.size+1,MPI_INT,next_partner_group.start_rank,stages,Comm,&rreq);
      }       
        
      if(next_group.size<next_partner_group.size && next_local_rank==0)
      {
        MPI_Request request;
        //send to last one in partner group
        //cout << rank << ": sending additional to: " << next_partner_group.start_rank+next_partner_group.size-1 << endl;
        //start send
        MPI_Isend(&sum_histogram[0],buckets+next_group.size+1,MPI_INT,next_partner_group.start_rank+next_partner_group.size-1,stages,Comm,&request);
        hsreqs.push_back(request);
      }
    }
  }
#ifdef _TIMESFC_
  finish=timer->currentSeconds();
  timers[3]+=finish-start;
  start=timer->currentSeconds();
#endif 
  Group group;
  //cout << rank << ": merging phase start\n";
  for(int stage=stages;stage>0;stage--)
  { 
    MPI_Status status;
    //update current group state
    group=next_group;
    Group partner_group=next_partner_group;
    Group parent_group=next_parent_group;
    int local_rank=next_local_rank;

    //update next group state
    next_group=groups[stage-1][group.parent_group];
    next_local_rank=rank-next_group.start_rank;
    
    if(next_group.partner_group!=-1)
    {
      next_partner_group=groups[stage-1][next_group.partner_group];
      next_partner_rank=next_partner_group.start_rank+next_local_rank;
    }
    
    if(stage-2>=0)
    {
      next_parent_group=groups[stage-2][next_group.parent_group];
    }
    
    //cout << rank << ": next group:  start_rank:" << next_group.start_rank << " size:" << next_group.size << " partner_group:" << next_group.partner_group << " next_local_rank:" << next_local_rank << " next_partner_rank:" << next_partner_rank <<endl;
    
   
    if(group.partner_group!=-1)
    {
#ifdef _TIMESFC_
      finish=timer->currentSeconds();
      timers[1]+=finish-start;
      start=timer->currentSeconds();
#endif
      //wait for histogram communiation to complete
      MPI_Wait(&rreq,&status);
      MPI_Waitall(hsreqs.size(),&hsreqs[0],MPI_STATUSES_IGNORE);
      hsreqs.clear();
      
#ifdef _TIMESFC_
      finish=timer->currentSeconds();
      timers[3]+=finish-start;
      start=timer->currentSeconds();
#endif

      //swap histograms, histogram is now the current histogram
      histogram.swap(sum_histogram);
      recv_histogram.swap(next_recv_histogram);
      /*   
      cout << rank << ": recieved histogram: ";
      for(unsigned int i=0;i<buckets;i++)
      {
        cout << recv_histogram[i] << " ";
      }
      cout << endl;
      cout << rank << ": cuts: ";
      for(unsigned int i=0;i<partner_group.size;i++)
      {
	      cout << recv_histogram[buckets+i] << " ";
      }
      cout << endl;
      */
      int total=0;
      //sum histograms
      for(unsigned int i=0;i<buckets;i++)
      {
        sum_histogram[i]=recv_histogram[i]+histogram[i];
        total+=sum_histogram[i];
      }

      //calcualte new cuts
      float mean=float(total)/parent_group.size;
      float target=ceil(mean);
      int remaining=total;
        
      //cout << rank << ": mean:" << mean << " target: " << target << " total:" << total << endl;
      //initiliaze cuts to 0
      for(int p=0;p<P+1;p++)
        sum_histogram[buckets+p]=0;
      
      int current=0;
      int p=1;
      for(unsigned int bucket=0;bucket<buckets;bucket++)
      {
        float takeimb=fabs(current+sum_histogram[bucket]-target); //amount away if p-1 takes this work
        float notakeimb=fabs(current-target);                     //amount away if p takes this work
        if(takeimb>notakeimb) //minimize imbalance
        {
                
          //move to the next proc
          sum_histogram[buckets+p]=bucket;
          remaining-=current;
          target=ceil((float)remaining/(parent_group.size-p));
          p++;
 
          current=sum_histogram[bucket];
        }
        else
        {
          current+=sum_histogram[bucket];
        }
      }
      sum_histogram[buckets+p]=buckets;
    
      /*
      //if(local_rank==0)
      {
        cout << rank << ": buckets:" << buckets << " parent_group.size:" << parent_group.size << endl;
        cout << rank << ": new histogram:";
        for(unsigned int i=0;i<buckets;i++)
        {
          cout << sum_histogram[i] << " ";
        }
        cout << endl;
        cout << rank << ": cuts:";
        for(int i=0;i<parent_group.size+1;i++)
        {
          cout << sum_histogram[buckets+i] << " ";
        }
        cout << endl;
      }
      //*/
    }

#ifdef _TIMESFC_
    finish=timer->currentSeconds();
    timers[2]+=finish-start;
    start=timer->currentSeconds();
#endif
    if(next_group.partner_group!=-1)
    {
       //start sending histogram
      if(next_local_rank<next_partner_group.size)
      {
        //cout << rank << ": sending and recieving from: " << next_partner_rank << endl;
        MPI_Request request;
          
        //start send
        MPI_Isend(&sum_histogram[0],buckets+next_group.size+1,MPI_INT,next_partner_rank,stage-1,Comm,&request);
        hsreqs.push_back(request);
        //start recv
        MPI_Irecv(&next_recv_histogram[0],buckets+next_partner_group.size+1,MPI_INT,next_partner_rank,stage-1,Comm,&rreq);
      }
      else
      {
        //partner doesn't exist
        //no send needed

        //recieve from rank 0 in partner group
        //cout << rank << ": recieving from: " << next_partner_group.start_rank << endl;
        //start send
        MPI_Irecv(&next_recv_histogram[0],buckets+next_partner_group.size+1,MPI_INT,next_partner_group.start_rank,stage-1,Comm,&rreq);
      }       
        
      if(next_group.size<next_partner_group.size && next_local_rank==0)
      {
        MPI_Request request;
        //send to last one in partner group
        //cout << rank << ": sending additional to: " << next_partner_group.start_rank+next_partner_group.size-1 << endl;
        //start send
        MPI_Isend(&sum_histogram[0],buckets+next_group.size+1,MPI_INT,next_partner_group.start_rank+next_partner_group.size-1,stage-1,Comm,&request);
        hsreqs.push_back(request);
      }
    }
#ifdef _TIMESFC_
    finish=timer->currentSeconds();
    timers[3]+=finish-start;
    start=timer->currentSeconds();
#endif
    //redistribute keys
    if(group.partner_group!=-1)
    {
/*
      if(parent_group.size<0 || parent_group.size>2048)
      {
	cout << rank << ": error invalid parent group!!\n";
      }
*/
      vector<int> sendcounts(parent_group.size,0), recvcounts(parent_group.size,0), senddisp(parent_group.size,0), recvdisp(parent_group.size,0);
        
      int oldstart=histogram[buckets+local_rank],oldend=histogram[buckets+local_rank+1];
      //cout << rank << ": oldstart:" << oldstart << " oldend:" << oldend << endl;
        
      //calculate send count
      for(int p=0;p<parent_group.size;p++)
      {
        //i own old histogram from buckets oldstart to oldend
        //any elements between oldstart and oldend that do not belong on me according to the new cuts must be sent
        //cout << rank << ": sum_histogram[buckets+p]:" << sum_histogram[buckets+p] << " sum_histogram[buckets+p+1]:" << sum_histogram[buckets+p+1] << endl; 
        int start=max(oldstart,sum_histogram[buckets+p]),end=min(oldend,sum_histogram[buckets+p+1]);
	for(int bucket=start;bucket<end;bucket++)
        {
           sendcounts[p]+=histogram[bucket];
        }
      }
        
      //calculate recv count
      //i will recieve from every processor that owns a bucket assigned to me
      //ownership is determined by that processors old histogram and old cuts
       
      int newstart=sum_histogram[buckets+next_local_rank],newend=sum_histogram[buckets+next_local_rank+1];
      //cout << rank << ": newstart: " << newstart << " newend:" << newend << endl;

      int *lefthistogram,*righthistogram;
      int leftsize,rightsize;

      if(group.start_rank<partner_group.start_rank)
      {
        lefthistogram=&histogram[0];
        leftsize=group.size;
        righthistogram=&recv_histogram[0];
        rightsize=partner_group.size;
      }
      else
      {
        righthistogram=&histogram[0];
        rightsize=group.size;
        lefthistogram=&recv_histogram[0];
        leftsize=partner_group.size;
      }
        
      //old histogram and cuts is histogram
      for(int p=0;p<leftsize;p++)
      {
        //cout << rank << ": lefthistogram[buckets+p]:" << lefthistogram[buckets+p] << " lefthistogram[buckets+p+1]:" << lefthistogram[buckets+p+1] << endl;
	int start=max(newstart,lefthistogram[buckets+p]), end=min(newend,lefthistogram[buckets+p+1]);
        for(int bucket=start;bucket<end;bucket++)
        {
          recvcounts[p]+=lefthistogram[bucket];
        }
      } 
      //old histogram and cuts is recv_histogram
      for(int p=0;p<rightsize;p++)
      {
	int start=max(newstart,righthistogram[buckets+p]),end=min(newend,righthistogram[buckets+p+1]);
        for(int bucket=start;bucket<end;bucket++)
        {
          recvcounts[p+leftsize]+=righthistogram[bucket];
        }
      } 
      unsigned int newn=0;
      for(int p=0;p<parent_group.size;p++)
      {
        newn+=recvcounts[p];
      }
      if(newn!=n)
      {
	  cout << rank << ": Warning newn may be invalid it is:" <<  newn << " n is:" << n << endl;
	  cout << rank << ": stages;" << stages << " stage:" << stage << endl;
	  cout << rank << ": partner rank:" << partner_group.start_rank+local_rank;
	  cout << rank << ": histogram sizes:" << histogram.size() << " " << recv_histogram.size() << " " << sum_histogram.size() << endl;
	  
          cout << rank << ": recvcounts:";
          for(int i=0;i<parent_group.size;i++)
                  cout << recvcounts[i] << " ";
          cout << endl;
          cout << rank << ": sendcounts:";
          for(int i=0;i<parent_group.size;i++)
                  cout << sendcounts[i] << " ";
          cout << endl;
     	
	  cout << rank << ": histogram:";
	  for(unsigned int i=0;i<buckets;i++)
	  {
	     cout << histogram[i] << " ";
	  }
	  cout << endl;
	  cout << rank << ": histogram cuts:";
	  for(int i=0;i<leftsize+1;i++)
	  {
	     cout << histogram[i+buckets] << " ";
	  }
	  cout << endl;
	  cout << rank << ": recv histogram:";
	  for(unsigned int i=0;i<buckets;i++)
	  {
	     cout << recv_histogram[i] << " ";
	  }
	  cout << endl;
	  cout << rank << ": recv cuts:";
	  for(int i=0;i<rightsize+1;i++)
	  {
	     cout << righthistogram[i+buckets] << " ";
	  }
	  cout << endl;
	  cout << rank << ": sum histogram:";
	  for(unsigned int i=0;i<buckets;i++)
	  {
	     cout << sum_histogram[i] << " ";
	  }
	  cout << endl;
	  cout << rank << ": sum cuts:";
	  for(int i=0;i<rightsize+1;i++)
	  {
	     cout << sum_histogram[i+buckets] << " ";
	  }
	  cout << endl;
	  
      }
      //cout << rank << " resizing histories to:" << newn << endl;
      
      recv_histories.resize(newn);
      merge_histories.resize(newn);
      
      /*
      cout << rank << ": sendcounts: ";
      for(int p=0;p<parent_group.size;p++)
      {
        cout << sendcounts[p] << " ";
      }
      cout << endl;
      cout << rank << ": recvcounts: ";
      for(int p=0;p<parent_group.size;p++)
      {
        cout << recvcounts[p] << " ";
      }
      cout << endl;
      //*/
      //calculate displacements
      for(int p=1;p<parent_group.size;p++)
      {
        senddisp[p]+=senddisp[p-1]+sendcounts[p-1];
        recvdisp[p]+=recvdisp[p-1]+recvcounts[p-1];
      }
       
      //redistribute keys 
      vector<MPI_Request> rreqs,sreqs;
      
#ifdef _TIMESFC_
     finish=timer->currentSeconds();
     timers[2]+=finish-start;
     start=timer->currentSeconds();
#endif        
      for(int p=0;p<parent_group.size;p++)
      {
          
        if(p==next_local_rank)
          continue;
          
        MPI_Request request;
          
        //start send
        if(sendcounts[p]!=0)
        {
          //cout << rank << ": sending to " << parent_group.start_rank+p << endl;
	  if((int)myhistories.size()<senddisp[p]+sendcounts[p])
	  {
		cout << rank << ": error sending, send size is bigger than buffer\n";
 	  }
          MPI_Isend(&myhistories[senddisp[p]],sendcounts[p]*sizeof(History<BITS>),MPI_BYTE,parent_group.start_rank+p,2*stages+stage,Comm,&request);
          sreqs.push_back(request); 
        }
          
        //start recv
        if(recvcounts[p]!=0)
        {
          //cout << rank << ": recieving from " << parent_group.start_rank+p << endl;
	  if((int)recv_histories.size()<recvdisp[p]+recvcounts[p])
	  {
		cout << rank << ": error reciving, recieve size is bigger than buffer\n";
 	  }
          MPI_Irecv(&recv_histories[recvdisp[p]],recvcounts[p]*sizeof(History<BITS>),MPI_BYTE,parent_group.start_rank+p,2*stages+stage,Comm,&request);
          rreqs.push_back(request);
        }
          
      }
#ifdef _TIMESFC_
     finish=timer->currentSeconds();
     timers[5]+=finish-start;
     start=timer->currentSeconds();
#endif      
     total_recvs+=rreqs.size();
     num_recvs++;
#if 0 
      unsigned int stages=1;
      unsigned int l=1;
      unsigned int rsize=rreqs.size();
      
      
      if(recvcounts[next_local_rank]!=0)
        rsize++;
     
      //compute merging stages
      for(;l<rsize;stages++,l<<=1);
     
      //cout << rank << ": merging stages:" << stages << endl;
      //create buffers
      vector<vector<vector<History<BITS> > > > done(stages);
      if(recvcounts[next_local_rank]!=0)
      {
        //copy my list to buffers 
		
        done[0].push_back(vector<History<BITS > >(myhistories.begin()+senddisp[next_local_rank],myhistories.begin()+senddisp[next_local_rank]+sendcounts[next_local_rank]));
      }
      //wait for recvs
      for(unsigned int i=0;i<rreqs.size();i++)
      { 
        MPI_Status status;
        int index;
    	
	finish=timer->currentSeconds();
    	timers[1]+=finish-start;
    	start=timer->currentSeconds();
     
        //wait any
        MPI_Waitany(rreqs.size(),&rreqs[0],&index,&status);
        
        finish=timer->currentSeconds();
        timers[2]+=finish-start;
        start=timer->currentSeconds();
        int mstage=0;
        int p=status.MPI_SOURCE-parent_group.start_rank; 
        //add list to done
        done[0].push_back(vector<History<BITS> >(recv_histories.begin()+recvdisp[p],recv_histories.begin()+recvdisp[p]+recvcounts[p])); 
        //process done requests
        while(done[mstage].size()==2)
        {
          //create mergeto buffer
          done[mstage+1].push_back(vector<History<BITS> >(done[mstage][0].size()+done[mstage][1].size()) ); 
          done[mstage+1].back().resize(0); 
          //cout << rank << ": merging:  mstage:" << mstage << " list sizes are:" << done[mstage][0].size() << " and " << done[mstage][1].size() << endl;
          //merge lists into new buffer
          merge(done[mstage][0].begin(),done[mstage][0].end(),done[mstage][1].begin(),done[mstage][1].end(),back_inserter(done[mstage+1].back()));
          /*
          cout << rank << ": done merging, list is"; 
          for(unsigned int i=0;i<done[mstage+1][0].size();i++)
          {
            cout << (int)done[mstage+1][0][i].bits << " ";
          }
          cout << endl;
          */
          //clear buffers we merged from
          done[mstage].resize(0);
          //next merging stage
          mstage++;   
        } 
      }
      //finish remaining merges
      for(unsigned int mstage=0;mstage<stages-1;mstage++)
      {
        if(done[mstage].size()==1)
        {
          //create next level and assign this vector to it
          done[mstage+1].push_back(vector<History<BITS> >(done[mstage][0].begin(),done[mstage][0].end()));
        }
        else if(done[mstage].size()==2)
        {
          //create mergeto buffer
          done[mstage+1].push_back(vector<History<BITS> >(done[mstage][0].size()+done[mstage][1].size()) ); 
          done[mstage+1].back().resize(0); 
        
          //merge lists into new buffer
          merge(done[mstage][0].begin(),done[mstage][0].end(),done[mstage][1].begin(),done[mstage][1].end(),back_inserter(done[mstage+1].back()));
        }
      }

      if(done.back().size()>0)
      {
        //cout << rank << ": resizing mergefrom to size:" << done.back().back().size() << endl;
        merge_histories.resize(done.back().back().size());
        //cout << rank << ": copying to mergefrom\n"; 
        merge_histories.assign(done.back().back().begin(),done.back().back().end());
      }
      else
      {
        merge_histories.resize(0);
      }
#elif 1
      temp_histories.reserve(newn);
      temp_histories.resize(0);
      merge_histories.resize(0);
      
      if(recvcounts[next_local_rank]!=0)
      {
        //copy my list to merge buffer
        merge_histories.assign(myhistories.begin()+senddisp[next_local_rank],myhistories.begin()+senddisp[next_local_rank]+sendcounts[next_local_rank]);
      }
      
      for(unsigned int i=0;i<rreqs.size();i++)
      {
        MPI_Status status;
        int index;
        
#ifdef _TIMESFC_
      	finish=timer->currentSeconds();
      	timers[4]+=finish-start;
      	start=timer->currentSeconds();
#endif
        //wait any
        MPI_Waitany(rreqs.size(),&rreqs[0],&index,&status);
#ifdef _TIMESFC_
      	finish=timer->currentSeconds();
     	  timers[5]+=finish-start;
      	start=timer->currentSeconds();
#endif
        
        int p=status.MPI_SOURCE-parent_group.start_rank; 
        if(merge_histories.size()==0)
        {
         
          temp_histories.assign(recv_histories.begin()+recvdisp[p],recv_histories.begin()+recvdisp[p]+recvcounts[p]);
        }
        else
        {
          merge(merge_histories.begin(),merge_histories.end(),recv_histories.begin()+recvdisp[p],recv_histories.begin()+recvdisp[p]+recvcounts[p],back_inserter(temp_histories));
        }
       
         merge_histories.swap(temp_histories);
      }

#endif
#ifdef _TIMESFC_
      finish=timer->currentSeconds();
      timers[4]+=finish-start;
      start=timer->currentSeconds();
#endif
      
      //wait for sends
      if(sreqs.size()>0)
     	 MPI_Waitall(sreqs.size(), &sreqs[0], MPI_STATUSES_IGNORE);
      
#ifdef _TIMESFC_
      finish=timer->currentSeconds();
      timers[5]+=finish-start;
      start=timer->currentSeconds();
#endif
      
      myhistories.swap(merge_histories);
    } //end no partner group
  } //end merging stages
  //Copy permutation to orders
  orders->resize(myhistories.size());
  for(unsigned int i=0;i<myhistories.size();i++)
  {
    (*orders)[i]=myhistories[i].index;
  } 
  /* 
  cout << rank << ": final list: ";
  for(unsigned int i=0;i<myhistories.size();i++)
  {
     cout << (int)myhistories[i].bits << " ";
  }
  cout << endl;
  */
#ifdef _TIMESFC_
  finish=timer->currentSeconds();
  timers[1]+=finish-start;
  start=timer->currentSeconds();
#endif
#if 0
  double avg_recvs=double(total_recvs)/num_recvs;
  double sum,max;
  MPI_Reduce(&avg_recvs,&sum,1,MPI_DOUBLE,MPI_SUM,0,Comm);
  MPI_Reduce(&avg_recvs,&max,1,MPI_DOUBLE,MPI_MAX,0,Comm);
  avg_recvs=sum/P;

  if(rank==0)
  {
    cout << "averge recvs:" << avg_recvs << " max recvs:" << max << endl;
  }
#endif
}
        
template<int DIM, class LOCS> template<class BITS>
void SFC<DIM,LOCS>::Parallel1()
{
 
#ifdef _TIMESFC_
  start=timer->currentSeconds();
#endif  
  vector<History<BITS> > myhistories(n), mergefrom(n), mergeto(n);
 
  
  //calculate local curves 
  SerialH<BITS>(&myhistories[0]);
  
#ifdef _TIMESFC_
  finish=timer->currentSeconds();
  timers[0]+=finish-start;
  start=timer->currentSeconds();
#endif
  CalculateHistogramsAndCuts<BITS>(myhistories);
  
  //build send counts and displacements
  vector<int> sendcounts(P,0);
  vector<int> recvcounts(P,0);
  vector<int> senddisp(P,0);
  vector<int> recvdisp(P,0);
  
  for(int p=0;p<P;p++)
  {
    //calculate send count
      //my row of the histogram summed up across buckets assigned to p
    for(int bucket=cuts[p];bucket<cuts[p+1];bucket++)
    {
       sendcounts[p]+=histograms[rank*buckets+bucket];     
    }
    
    //calculate recv count
      //my bucket colums of the histogram summed up for each processor
    for(int bucket=cuts[rank];bucket<cuts[rank+1];bucket++)
    {
      recvcounts[p]+=histograms[p*buckets+bucket];            
    }

  }
  //calculate displacements
  for(int p=1;p<P;p++)
  {
    senddisp[p]+=senddisp[p-1]+sendcounts[p-1];
    recvdisp[p]+=recvdisp[p-1]+recvcounts[p-1];
    
  }
  int newn=0;
  for(int p=0;p<P;p++)
  {
    newn+=recvcounts[p];
  }
  //delete histograms
  histograms.resize(0);

#ifdef _TIMESFC_
  finish=timer->currentSeconds();
  timers[1]+=finish-start;
  start=timer->currentSeconds();
#endif
  /*
  if(rank==0)   
  {
     for(int p=0;p<P+1;p++)
     {
        for(unsigned int i=0;i<buckets;i++)
          cout << histograms[p*buckets+i] << " ";
        cout << endl;
     }
     cout << "send:senddisp:recv counts:recev disp:\n";
     for(int p=0;p<P;p++)
     {
       cout << sendcounts[p] << ":" << senddisp[p] << ":" <<  recvcounts[p] << ":" << recvdisp[p] << endl;
     }
  }
  */
  //cout << rank << ": newn:" << newn << endl;
  
  myhistories.reserve(newn), mergefrom.reserve(newn); mergeto.reserve(newn);
  
  //Recieve Keys
  
  if(mergemode==1)
  {
    //scale counts by size of history
    for(int p=0;p<P;p++)
    {
      sendcounts[p]*=sizeof(History<BITS>);
      recvcounts[p]*=sizeof(History<BITS>);
      senddisp[p]*=sizeof(History<BITS>);
      recvdisp[p]*=sizeof(History<BITS>);
    }
    //cout << rank << ": all to all\n";
    MPI_Alltoallv(&myhistories[0],&sendcounts[0],&senddisp[0],MPI_BYTE,
                 &mergefrom[0],&recvcounts[0],&recvdisp[0],MPI_BYTE,Comm);
#ifdef _TIMESFC_
    finish=timer->currentSeconds();
    timers[2]+=finish-start;
    start=timer->currentSeconds();
#endif
    /* 
    if(rank==31)
    {
      cout << rank << ":" << "lists to merge: ";
      for(int p=0;p<P;p++)
      {
        cout << "list:" << p << ": ";
        for(unsigned int i=recvdisp[p]/sizeof(History<BITS>);i<(recvcounts[p]+recvdisp[p])/sizeof(History<BITS>);i++)
        {
          cout <<  (int)newhistories[i].bits << " ";
        }
      }
      cout << endl;
    }
    */

  
    vector<int> mergesizes(P),mergepointers(P);
    for(int p=0;p<P;p++)
    {
      mergesizes[p]=recvcounts[p]/sizeof(History<BITS>);
      mergepointers[p]=recvdisp[p]/sizeof(History<BITS>);
    }
    //Merge Keys using hypercube design
    int lists=P;
 
    while(lists>1)
    {
      /*
      cout << rank << ": lists to merge: ";
      for(int l=0;l<lists;l++)
      {
        cout << "list:" << l << ": ";
        for(int i=mergepointers[l];i<mergepointers[l]+mergesizes[l];i++)
        {
          cout <<  (int)mergefrom[i].bits << " ";
        }
      } 
      cout << endl << endl;   
      */
      int l=0;
      mergeto.resize(0);
      for(int i=0;i<lists;i+=2)
      {
        int mln=mergesizes[i]+mergesizes[i+1];
        if(mln!=0)
        {
          int mlp=mergeto.size();
          typename vector<History<BITS> >::iterator l1begin=mergefrom.begin()+mergepointers[i];
          typename vector<History<BITS> >::iterator l2begin=mergefrom.begin()+mergepointers[i+1];
          typename vector<History<BITS> >::iterator l1end=mergefrom.begin()+mergepointers[i]+mergesizes[i];
          typename vector<History<BITS> >::iterator l2end=mergefrom.begin()+mergepointers[i+1]+mergesizes[i+1];
          /*
          cout << rank << ": merging lists: ";
          for(vector<History<BITS> >::iterator iter=l1begin;iter<l1end;iter++)
          {
            cout << (int)iter->bits << " ";
          }
          cout << " and ";
          for(vector<History<BITS> >::iterator iter=l2begin;iter<l2end;iter++)
          {
            cout << (int)iter->bits << " ";
          }
          */
          merge(l1begin,l1end,l2begin,l2end,std::back_inserter(mergeto));
          mergesizes[l]=mln;
          mergepointers[l]=mlp;
          /*
          cout << " merged list:"; 
          for(vector<History<BITS> >::iterator iter=mergeto.begin()+mlp;iter<mergeto.begin()+mlp+mln;iter++)
          {
            cout << (int)iter->bits << " ";
          }
          cout << endl;
          */
          l++;
        }
      }          
      lists=l;
      //swap(mergeto,mergefrom);
      mergeto.swap(mergefrom);
    } 
  }
  else if(mergemode==2 || mergemode==3)
  {
    vector<MPI_Request> sreqs;
    vector<MPI_Request> rreqs;
  
    vector<History<BITS> > recvbuf(newn);
    MPI_Request empty;
    //start sends
    for(int p=0;p<P;p++)
    {
      if(sendcounts[p]!=0 && p!=rank)
      {
        //cout << rank << " sending " << sendcounts[p] << " to rank " << p << endl;
        sreqs.push_back(empty);
        MPI_Isend(&myhistories[senddisp[p]],sendcounts[p]*sizeof(History<BITS>),MPI_BYTE,p,0,Comm,&sreqs.back());
      }
    } 
    //start recieves
    for(int p=0;p<P;p++)
    {
      if(recvcounts[p]!=0 && p!=rank)
      {
        //cout << rank << " recving " << recvcounts[p] << " from rank " << p << endl;
        rreqs.push_back(empty);
        MPI_Irecv(&recvbuf[recvdisp[p]],recvcounts[p]*sizeof(History<BITS>),MPI_BYTE,p,0,Comm,&rreqs.back());
      }
    } 
  
    if(mergemode==2)
    { 
      //move my list into merge from.
      mergefrom.assign(myhistories.begin()+senddisp[rank],myhistories.begin()+senddisp[rank]+sendcounts[rank]);
  
      //wait for recvs
      for(unsigned int i=0;i<rreqs.size();i++)
      { 
        MPI_Status status;
        int index;
        //cout << "doing waitany\n";
        //wait any
#ifdef _TIMESFC_
	finish=timer->currentSeconds();
	timers[3]+=finish-start;
	start=timer->currentSeconds();
#endif
        MPI_Waitany(rreqs.size(),&rreqs[0],&index,&status);
#ifdef _TIMESFC_
	finish=timer->currentSeconds();
	timers[2]+=finish-start;
	start=timer->currentSeconds();
#endif
    
        mergeto.resize(0);
        //merge
        int p=status.MPI_SOURCE; 
        //cout << "Recieved list from " << p << endl;
        merge(mergefrom.begin(),mergefrom.end(),recvbuf.begin()+recvdisp[p],recvbuf.begin()+recvdisp[p]+recvcounts[p],std::back_inserter(mergeto));   
        //cout << "done merging\n";
        mergeto.swap(mergefrom);
      }
    }
    else if (mergemode==3)
    {
       unsigned int stages=1;
       unsigned int l=1;
       unsigned int rsize=rreqs.size();
       if(recvcounts[rank]!=0)
               rsize++;
       
       //compute merging stages
       for(;l<rsize;stages++,l<<=1);
       
       //create buffers
       vector<vector<vector<History<BITS> > > > done(stages);
        
       //cout << rank << ": stages: " << stages << endl; 

       if(recvcounts[rank]!=0)
       {
        //copy my list to buffers 
        done[0].push_back(vector<History<BITS > >(myhistories.begin()+senddisp[rank],myhistories.begin()+senddisp[rank]+sendcounts[rank]));
       }
      //wait for recvs
      for(unsigned int i=0;i<rreqs.size();i++)
      { 
        MPI_Status status;
        int index;
       
#ifdef _TIMESFC_
	finish=timer->currentSeconds();
	timers[3]+=finish-start;
	start=timer->currentSeconds();
#endif
        //wait any
        MPI_Waitany(rreqs.size(),&rreqs[0],&index,&status);
#ifdef _TIMESFC_
	finish=timer->currentSeconds();
	timers[2]+=finish-start;
	start=timer->currentSeconds();
#endif
       
        int mstage=0;
        int p=status.MPI_SOURCE; 
        //add list to done
        done[0].push_back(vector<History<BITS> >(recvbuf.begin()+recvdisp[p],recvbuf.begin()+recvdisp[p]+recvcounts[p])); 

        //process done requests
        while(done[mstage].size()==2)
        {
          //create mergeto buffer
          done[mstage+1].push_back(vector<History<BITS> >(done[mstage][0].size()+done[mstage][1].size()) ); 
          done[mstage+1].back().resize(0); 
          //cout << rank << ": merging:  mstage:" << mstage << " list sizes are:" << done[mstage][0].size() << " and " << done[mstage][1].size() << endl;
          //merge lists into new buffer
          merge(done[mstage][0].begin(),done[mstage][0].end(),done[mstage][1].begin(),done[mstage][1].end(),back_inserter(done[mstage+1].back()));
          /*
          cout << rank << ": done merging, list is"; 
          for(unsigned int i=0;i<done[mstage+1][0].size();i++)
          {
             cout << (int)done[mstage+1][0][i].bits << " ";
          }
          cout << endl;
          */
          //clear buffers we merged from
          done[mstage].resize(0);
          //next merging stage
          mstage++;   
        } 
      }
      //finish remaining merges
      for(unsigned int mstage=0;mstage<stages-1;mstage++)
      {
         if(done[mstage].size()==1)
         {
          //create next level and assign this vector to it
          done[mstage+1].push_back(vector<History<BITS> >(done[mstage][0].begin(),done[mstage][0].end()));
         }
         else if(done[mstage].size()==2)
         {
          //create mergeto buffer
          done[mstage+1].push_back(vector<History<BITS> >(done[mstage][0].size()+done[mstage][1].size()) ); 
          done[mstage+1].back().resize(0); 
          
          //merge lists into new buffer
          merge(done[mstage][0].begin(),done[mstage][0].end(),done[mstage][1].begin(),done[mstage][1].end(),back_inserter(done[mstage+1].back()));
         }
      }
     

        
      //cout << rank << ": resizing mergefrom to size:" << done.back().back().size() << endl;
      mergefrom.resize(done.back().back().size());
      //cout << rank << ": copying to mergefrom\n"; 
      mergefrom.assign(done.back().back().begin(),done.back().back().end());

    }
    
    //wait for sends
    for(unsigned int i=0;i<sreqs.size();i++)
    {
     MPI_Status status;
     MPI_Wait(&sreqs[i],&status);
    }
  }
  //Copy permutation to orders
  orders->resize(newn);
  for(int i=0;i<newn;i++)
  {
    (*orders)[i]=mergefrom[i].index;
  } 
 /* 
  cout << rank << ": final list: ";
  for(unsigned int i=0;i<mergefrom.size();i++)
  {
     cout << (int)mergefrom[i].bits << " ";
  }
   cout << endl;
 */
#ifdef _TIMESFC_
  finish=timer->currentSeconds();
  timers[3]+=finish-start;
#endif

}
template<int DIM, class LOCS> template<class BITS>
void SFC<DIM,LOCS>::Parallel0()
{
	vector<History<BITS> > histories(n);
	unsigned int i;
  
#ifdef _TIMESFC_
	start=timer->currentSeconds();
#endif
	SerialH<BITS>(&histories[0]);	//Saves results in sendbuf
#ifdef _TIMESFC_
	finish=timer->currentSeconds();
	timers[0]+=finish-start;
#endif
  vector<History<BITS> > rbuf, mbuf;
  
  
#ifdef _TIMESFC_
	start=timer->currentSeconds();
#endif
	PrimaryMerge<BITS>(histories,rbuf,mbuf);
#ifdef _TIMESFC_
	finish=timer->currentSeconds();
	timers[1]+=finish-start;
#endif
	
#ifdef _TIMESFC_
	start=timer->currentSeconds();
#endif
	Cleanup<BITS>(histories,rbuf,mbuf);
#ifdef _TIMESFC_
	finish=timer->currentSeconds();
	timers[2]+=finish-start;
#endif
  
	orders->resize(n);
	
	//copy permutation to orders
	for(i=0;i<n;i++)
	{
    (*orders)[i]=histories[i].index;
	}
  /*
  cout << rank << ": final list: ";
  for(unsigned int i=0;i<histories.size();i++)
  {
     cout << (int)histories[i].bits << ":" << histories[i].index.p << ":" << histories[i].index.i << " ";
  }
  cout << endl;
  */
}

#define ASCENDING 0
#define DESCENDING 1
template<int DIM, class LOCS> template<class BITS>
int SFC<DIM,LOCS>::MergeExchange(int to,vector<History<BITS> > &sendbuf, vector<History<BITS> >&recievebuf, vector<History<BITS> > &mergebuf)
{
	float inv_denom=1.0/sizeof(History<BITS>);
//	cout << rank <<  ": Merge Exchange started with " << to << endl;
	int direction= (int) (rank>to);
	BITS emax, emin;
	queue<MPI_Request> squeue, rqueue;
	unsigned int tag=0;
  unsigned int n2;
	
  MPI_Request srequest, rrequest;
	MPI_Status status;
  
  MPI_Isend(&n,1,MPI_INT,to,0,Comm,&srequest);
	MPI_Irecv(&n2,1,MPI_INT,to,0,Comm,&rrequest);
	
  MPI_Wait(&rrequest,&status);
	MPI_Wait(&srequest,&status);
  
  //temperary fix to prevent processors with no elements from crashing
  if(n==0 || n2==0)
     return 0;
  
	
	
	//min_max exchange
	if(direction==ASCENDING)
	{
		emax=sendbuf[n-1].bits;
		MPI_Isend(&emax,sizeof(BITS),MPI_BYTE,to,0,Comm,&srequest);
		MPI_Irecv(&emin,sizeof(BITS),MPI_BYTE,to,0,Comm,&rrequest);
	}
	else
	{
		emin=sendbuf[0].bits;
		MPI_Isend(&emin,sizeof(BITS),MPI_BYTE,to,0,Comm,&srequest);
		MPI_Irecv(&emax,sizeof(BITS),MPI_BYTE,to,0,Comm,&rrequest);
	}
  
  MPI_Wait(&rrequest,&status);
	MPI_Wait(&srequest,&status);
	
	if(emax<emin)	//if exchange not needed 
	{
		return 0;
	}
  
	//cout << rank << ": Max-min done\n";
  
  recievebuf.resize(n2);
  mergebuf.resize(n);
	
  History<BITS> *sbuf=&sendbuf[0], *rbuf=&recievebuf[0], *mbuf=&mergebuf[0];
	History<BITS> *msbuf=sbuf, *mrbuf=rbuf;
  
	unsigned int nsend=n;
	unsigned int nrecv=n2;
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
	//cout << rank << ": sample done\n";
	
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
	
  sendbuf.swap(mergebuf);
  //cout << rank << ": done ME\n";
	return 1;
}

struct HC_MERGE
{
	unsigned int base;
	unsigned int P;
};

template<int DIM, class LOCS> template<class BITS>
void SFC<DIM,LOCS>::PrimaryMerge(vector<History<BITS> > &histories, vector<History<BITS> >&rbuf, vector<History<BITS> > &mbuf)
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
			MergeExchange<BITS>(to,histories,rbuf,mbuf);
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
void SFC<DIM,LOCS>::Cleanup(vector<History<BITS> > &histories, vector<History<BITS> >&rbuf, vector<History<BITS> > &mbuf)
{
	switch(cleanup)
	{
		case BATCHERS:
			Batchers<BITS>(histories, rbuf, mbuf);
			break;
		case LINEAR:
			Linear<BITS>(histories, rbuf, mbuf);
			break;
	};
}
template<int DIM, class LOCS> template <class BITS>
void SFC<DIM,LOCS>::Batchers(vector<History<BITS> > &histories, vector<History<BITS> >&rbuf, vector<History<BITS> > &mbuf)
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
				MergeExchange<BITS>(rank+d,histories,rbuf,mbuf);
			}
			else if(rank-d>=0 && ((rank-d)&p)==r)
			{
				MergeExchange<BITS>(rank-d,histories,rbuf,mbuf);
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
void SFC<DIM,LOCS>::Linear(vector<History<BITS> > &histories, vector<History<BITS> >&rbuf, vector<History<BITS> > &mbuf)
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
				val+=MergeExchange<BITS>(rank+1,histories,rbuf,mbuf);	
			}

			if(rank!=0)
			{
				val+=MergeExchange<BITS>(rank-1,histories,rbuf,mbuf);
			}
			
		}
		else	//exchange left then right
		{
			if(rank!=0)
			{
				val+=MergeExchange<BITS>(rank-1,histories,rbuf,mbuf);
			}
			
			if(rank!=P-1)
			{
				val+=MergeExchange<BITS>(rank+1,histories,rbuf,mbuf);	
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
void SFC<DIM,LOCS>::SetOutputVector(vector<DistributedIndex> *orders)
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
