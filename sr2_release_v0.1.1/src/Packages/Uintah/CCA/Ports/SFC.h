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
double start, finish;
const int TIMERS=7;
double timers[TIMERS]={0};
#endif
enum Curve {HILBERT, MORTON, GREY};
enum CleanupType{BATCHERS,LINEAR};

#define SERIAL 1
#define PARALLEL 2

struct DistributedIndex
{
  unsigned int i;
  unsigned short p;
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
inline bool operator>=(const History<BITS> &a, const History<BITS> &b)
{
  return a.bits>=b.bits;
}
template <class BITS>
inline bool operator<(const History<BITS> &a, const History<BITS> &b)
{
  return a.bits<b.bits;
}
template <class BITS>
inline bool operator>(const History<BITS> &a, const History<BITS> &b)
{
  return a.bits>b.bits;
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
  SFC(int dir[][DIM], const ProcessorGroup *d_myworld) : dir(dir),set(0), locsv(0), locs(0), orders(0), d_myworld(d_myworld), comm_block_size(3000), blocks_in_transit(3), merge_block_size(100), sample_percent(.1), cleanup(BATCHERS), mergemode(1) {};
  virtual ~SFC() {};
  void GenerateCurve(int mode=0);
  void SetRefinements(int refinements);
  void SetLocalSize(unsigned int n);
  void SetLocations(vector<LOCS> *locs);
  void SetOutputVector(vector<DistributedIndex> *orders);
  void SetMergeParameters(unsigned int comm_block_size,unsigned int merge_block_size, unsigned int blocks_in_transit, float sample_percent);
  void SetCommBlockSize(unsigned int b) {comm_block_size=b;};
  void SetMergeBlockSize(unsigned int b) {merge_block_size=b;};
  void SetBlocksInTransit(unsigned int b) {blocks_in_transit=b;};
  void SetSamplePercent(float p) {sample_percent=p;};
  void SetCleanup(CleanupType cleanup) {this->cleanup=cleanup;};
  void SetMergeMode(int mode) {this->mergemode=mode;};
  void MergeTest(unsigned int N,int repeat);
  void ProfileMergeParameters(int repeat=21);
protected:

  SCIRun::Time *timer;
  
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
  unsigned int comm_block_size;
  unsigned int blocks_in_transit;
  unsigned int merge_block_size;
  float sample_percent;
  
  CleanupType cleanup;
  
  int rank, P;
  MPI_Comm Comm;  

  //Parllel2 Parameters
  unsigned int buckets;
  int b;
  int mergemode;
  
  void Serial();
  void SerialR(DistributedIndex* orders,vector<DistributedIndex> *bin, unsigned int n, REAL *center, REAL *dimension, unsigned int o=0);

  template<class BITS> void SerialH(History<BITS> *histories);
  template<class BITS> void SerialHR(DistributedIndex* orders,History<BITS>* corders,vector<DistributedIndex> *bin, unsigned int n, REAL *center, REAL *dimension,                                     unsigned int o=0, int r=1, BITS history=0);
  template<class BITS> void Parallel();
  template<class BITS> void Parallel0();
  template<class BITS> void Parallel1();
  template<class BITS> void Parallel2();
  template<class BITS> void Parallel3();
  template<class BITS> void ProfileMergeParametersT(int repeat);

  template<class BITS> void BlockedMerge(History<BITS>* start1, History<BITS>* end1, History<BITS>*start2,History<BITS>* end2,History<BITS>* out);

  
  template<class BITS> void CalculateHistogramsAndCuts(vector<BITS> &histograms, vector<BITS> &cuts, vector<History<BITS> > &histories);
  template<class BITS> void ComputeLocalHistogram(BITS *histogram,vector<History<BITS> > &histories);
  
  template<class BITS> int MergeExchange(int to,vector<History<BITS> > &histories, vector<History<BITS> >&rbuf, vector<History<BITS> > &mbuf);  
  template<class BITS> void PrimaryMerge(vector<History<BITS> > &histories, vector<History<BITS> >&rbuf, vector<History<BITS> > &mbuf);
  template<class BITS> void PrimaryMerge2(vector<History<BITS> > &histories, vector<History<BITS> >&rbuf, vector<History<BITS> > &mbuf,int  procs, int base=0);
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

/*
int myrand(int N)
{
  int retval;
  
  do
  {
    retval=int(rand()/(double)RAND_MAX*N);
  }
  while(retval==N);
  return retval;
}
template<int DIM, class LOCS>
void SFC<DIM,LOCS>::MergeTest(unsigned int N,int repeat)
{
  srand(time(0));
   
  
  P=d_myworld->size();
  Comm=d_myworld->getComm();
  rank=d_myworld->myrank();

  SetCleanup(LINEAR);
  
  //create list
  vector<History<unsigned long long> > list;
  vector<History<unsigned long long> > mylist,rbuf,mbuf;

  vector<int> sendcounts(P,0), senddisp(P,0);
  int recv_count;
  
  //create sendcounts & send disp
  int n=N/P;
  int rem=N%P;

  if(rank<rem)
     recv_count=n+1;
  else
     recv_count=n;

  this->n=recv_count; 
  mylist.resize(recv_count);
  mbuf.resize(recv_count);
  if(rank==0)
  {
    for(int p=0;p<P;p++)
    {
      if(p<rem)
      {
        sendcounts[p]=n+1;      
      }
      else
      {
        sendcounts[p]=n;
      }
    }
    for(int p=1;p<P;p++)
    {
      senddisp[p]=senddisp[p-1]+sendcounts[p-1];
    }
    for(int p=0;p<P;p++)
    {
      
      sendcounts[p]*=sizeof(History<unsigned long long>);
      senddisp[p]*=sizeof(History<unsigned long long>);
    }
    
    list.resize(N);
    for(unsigned int i=0;i<N;i++)
    {
      list[i].bits=i;
    }
    
  } 
  int linear_max=0;
  int total=0;
  for(int i=0;i<repeat;i++)
  {
    linear_total=0;
    if(rank==0)
    {
      //shuffle list
      random_shuffle(list.begin(),list.end(),myrand);
      
      //scatter list      
      MPI_Scatterv(&list[0],&sendcounts[0],&senddisp[0],MPI_BYTE,&mylist[0],recv_count*sizeof(History<unsigned long long>),MPI_BYTE,0,Comm);
    }
    else
    {
      //scatter list
      MPI_Scatterv(0,0,0,MPI_BYTE,&mylist[0],recv_count*sizeof(History<unsigned long long>),MPI_BYTE,0,Comm);
    }

    //sort mylist
    sort(mylist.begin(),mylist.end());
   
    //call primary merge
    //PrimaryMerge2<unsigned long long>(mylist,rbuf,mbuf,P); 
    PrimaryMerge<unsigned long long>(mylist,rbuf,mbuf); 
    
    //call cleanup
    Cleanup<unsigned long long>(mylist,rbuf,mbuf);
    
    total+=linear_total;
    if(linear_total>linear_max)
            linear_max=linear_total;
  }
  if(rank==0)
    cout << P << " " << N << " " << (float)total/repeat << " " << linear_max << endl;
  //output linear_total/repeat;
}
*/
template<int DIM, class LOCS>
void SFC<DIM,LOCS>::ProfileMergeParameters(int repeat)
{
  int errors=0;
  unsigned char mask;
  P=d_myworld->size();
  if(P==1)
  {
    return;
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

  rank=d_myworld->myrank();
  Comm=d_myworld->getComm();
        
  if((int)refinements*DIM<=(int)sizeof(unsigned char)*8)
  {
    ProfileMergeParametersT<unsigned char>(repeat);
  }
  else if((int)refinements*DIM<=(int)sizeof(unsigned short)*8)
  {
    ProfileMergeParametersT<unsigned short>(repeat);
  }
  else if((int)refinements*DIM<=(int)sizeof(unsigned int)*8)
  {
    ProfileMergeParametersT<unsigned int>(repeat);
  }
  else if((int)refinements*DIM<=(int)sizeof(unsigned long)*8)
  {
    ProfileMergeParametersT<unsigned long>(repeat);
  }
  else
  {
    if((int)refinements*DIM>(int)sizeof(unsigned long long)*8)
    {
      refinements=sizeof(unsigned long long)*8/DIM;
    }
    ProfileMergeParametersT<unsigned long long>(repeat);
  }
}  

#define sample(median,block_size, merge_size, blocks, sample_size)                                  \
{                                                                                                   \
   float start,finish;                                                                              \
   float median_sum;                                                                                \
   SetMergeParameters(block_size,merge_size,blocks,sample_size);                                    \
   for(int r=0;r<repeat;r++)                                                                        \
   {                                                                                                \
    histories.assign(histories_original.begin(),histories_original.end());                          \
    start=timer->currentSeconds();                                                                  \
    PrimaryMerge<BITS>(histories,rbuf,mbuf);                                                        \
    finish=timer->currentSeconds();                                                                 \
    times[r]=finish-start;                                                                          \
   }                                                                                                \
   sort(times.begin(),times.end());                                                                 \
   int mid=times.size()/2;                                                                          \
   if(times.size()%2==0)                                                                            \
    median=(times[mid-1]+times[mid])/2;                                                             \
   else                                                                                             \
    median=times[mid];                                                                              \
    MPI_Allreduce(&median,&median_sum,1,MPI_FLOAT,MPI_SUM,Comm);                                    \
    median=median_sum/P;                                                                            \
}


template<int DIM, class LOCS> template <class BITS>
void SFC<DIM,LOCS>::ProfileMergeParametersT(int repeat)
{
  float sig=.01;
  int max_iter=20;
  
  int block_size=100;
  int delta_block_size=50;
  int min_delta_block_size=25;
  int min_block_size=10;
  int max_block_size;
  int blocks=1;
  float sample_size=.15;
  
  MPI_Allreduce(&n,&max_block_size,1,MPI_UNSIGNED,MPI_MAX,Comm);
  vector<History<BITS> > rbuf(max_block_size), mbuf(max_block_size);
  vector<History<BITS> > histories(max_block_size);
  vector<History<BITS> > histories_original(max_block_size);
  
  //sorting input and using it as our test case
  SerialH<BITS>(&histories_original[0]);  //Saves results in sendbuf

  //create merging search parameters...
  vector<float> times(repeat);
 
  //determine block size
  float last,cur;
  int last_dir=0;

  /*
  //run once to avoid some strange problem where the first time running is way off
  sample(cur,block_size,block_size,blocks,sample_size);
  sample(last,block_size+delta_block_size,block_size,blocks,sample_size);
  if(rank==0)
  {
    cout << "cur:" << cur << " last:" << last << endl;
  }
  */

  sample(cur,block_size,block_size,blocks,sample_size);
  sample(last,block_size+delta_block_size,block_size,blocks,sample_size);
  
//  if(rank==0)
//  {
//    cout << "t1: " << block_size << " " << cur << endl;
//    cout << "t2: " << block_size+delta_block_size << " " << last << endl;
//    cout << "-----------------------\n";
//  }
  
  if(cur<last)
  {
    last=cur;
    //block_size-=delta_block_size;
    last_dir=-1;
  }
  else
  {
    block_size+=delta_block_size;
    last_dir=1;
  }
  int last_count=1;
  
  int iter=0;
  while(iter<max_iter)
  {
    iter++;
    if(delta_block_size<min_delta_block_size)
    {
       break;
    }
   // if(rank==0)
   //   cout << "checking with block_size: " << block_size+delta_block_size*last_dir << endl;
    //measure new points
    sample(cur,block_size+delta_block_size*last_dir,block_size+delta_block_size*last_dir,blocks,sample_size);
   // if(rank==0)
   // {
   //   cout << "last: " << block_size << " " << last << endl;
   //   cout << "cur: " << block_size+delta_block_size*last_dir << " " << cur << endl;
   //   cout << "-----------------------\n";
   // }
    
    if(fabs(1-cur/last)<sig)  //if t1 & t2 are not signifigantly different
    {
      //if(rank==0)
      //  cout << "not signifigantly different\n"; 
      //halve delta and switch directions
      delta_block_size/=2;
      last_dir*=-1;
      last_count=0;
      continue;  
    }
    
    if(cur<last)
    {
      
      //continue in the same direction

      block_size+=delta_block_size*last_dir;
      if(last_count>1)
        delta_block_size*=2;    //double the delta
      
      while(block_size+delta_block_size*last_dir>max_block_size)
        delta_block_size/=2;
      while(block_size+delta_block_size*last_dir<min_block_size)
        delta_block_size/=2;
      
      swap(cur,last);
      last_count++;
    }
    else
    {
      //switch direction
      if(last_count==1)
        delta_block_size/=2; //halve the delta
      else if(last_count>1)
        delta_block_size/=4;

      last_dir*=-1;
      last_count=1;
    }
    

  }
//  if(rank==0)
//    cout << "Using block_size=" << block_size <<endl;

  blocks++;
  //determine number of blocks to send at a time
  sample(cur,block_size,block_size,blocks,sample_size);
  
//  if(rank==0)
//    cout << "blocks:" << blocks << " " << cur << endl;
  //last=999999999;

  while(cur<last)
  {
    swap(cur,last);
    blocks++;
    sample(cur,block_size,block_size,blocks,sample_size);
  //  if(rank==0)
  //    cout << "blocks:" << blocks << " " << cur << endl;
  //  if( fabs(1-cur/last)<sig )
  //    break;
  }
  blocks--;
//  if(rank==0)
//    cout << "Using blocks:" << blocks << endl;
  
  
  //determine merge block size
  vector<int> merge_block_sizes;

  if(block_size>=2)
    merge_block_sizes.push_back(int(block_size*.5));
  if(block_size>=3)
    merge_block_sizes.push_back(int(block_size*.3334));
  if(block_size>=4)
    merge_block_sizes.push_back(int(block_size*.25));
  if(block_size>=10)
    merge_block_sizes.push_back(int(block_size*.1));
  if(block_size>=20)
    merge_block_sizes.push_back(int(block_size*.05));

//  last=999999999;
//  sample(last,block_size,merge_block_sizes[0],blocks,sample_size);

  //if(rank==0)
  //  cout << "merge_block_size:" << block_size << " " << last << endl;
  int merge_size=block_size;
  
  for(unsigned int i=0;i<merge_block_sizes.size();i++)
  {
    sample(cur,block_size,merge_block_sizes[i],blocks,sample_size);
    //if(rank==0)
    //  cout << "merge_block_size:" << merge_block_sizes[i] << " " << cur << endl;
    if(fabs(1-cur/last)<sig)
    {
      //if(rank==0)
      //  cout << "not signigantly different\n";
    //  break;
    }
    if(cur<last)
    {
      last=cur;
      merge_size=merge_block_sizes[i];
    }
  }
  //if(rank==0)
  //  cout << "Using merge_block_size:" << merge_size << endl;
  
  SetMergeParameters(block_size,merge_size,blocks,sample_size);
 // if(rank==0)
 // {
 //   cout << "Merge Parmeters:" << block_size << " " << merge_size << " " << blocks << " " << sample_size << " " << last << endl;
 // }
  //determine number of procs?
  //determine merge mode?

}

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
    else if((int)refinements*DIM<=(int)sizeof(unsigned long)*8)
    {
      Parallel<unsigned long>();
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
      LOCS l[DIM];
      memcpy(l,&locs[orders[0].i*DIM],sizeof(LOCS)*DIM);
      i=1;
      while(same && i<n)
      {
        for(int d=0;d<DIM;d++)
        {
          if(fabs(l[d]-locs[orders[i].i*DIM+d])>EPSILON)
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
void SFC<DIM,LOCS>::ComputeLocalHistogram(BITS *histogram,vector<History<BITS> > &histories)
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
void SFC<DIM,LOCS>::CalculateHistogramsAndCuts(vector<BITS> &histograms, vector<BITS> &cuts, vector<History<BITS> > &histories)
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
    MPI_Allgather(&histograms[rank*(buckets)],buckets*sizeof(BITS),MPI_BYTE,&histograms[0],buckets*sizeof(BITS),MPI_BYTE,Comm);        

    //sum histogram
    BITS *sum=&histograms[P*buckets];
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
    for(BITS bucket=0;bucket<buckets;bucket++)
    {
      double imb1=fabs(current-target);
      double imb2=fabs(current+sum[bucket]-target);
      if(imb1<imb2)
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
void SFC<DIM,LOCS>::BlockedMerge(History<BITS>* start1, History<BITS>* end1, History<BITS>*start2,History<BITS>* end2,History<BITS>* out)
{
  History<BITS> *outend=out+(end1-start1)+(end2-start2);
  //merge
  while(out<outend)
  {
    if(end1==start1)  //copy everyting from start2
    {
      memcpy(out,start2,(end2-start2)*sizeof(History<BITS>));
      break;
    }
    else if (end2==start2) //copy everything from start1
    {
      memcpy(out,start1,(end1-start1)*sizeof(History<BITS>));
      break;
    }

    int m1b=min(int(end1-start1),(int)merge_block_size);
    int m2b=min(int(end2-start2),(int)merge_block_size);

    BITS mmin=start1->bits, mmax=(start1+m1b-1)->bits, bmin=start2->bits,bmax=(start2+m2b-1)->bits;

    if(mmax<=bmin) //take everything from start1 block
    {
       memcpy(out,start1,m1b*sizeof(History<BITS>));
       start1+=m1b;
       out+=m1b;
    }
    else if(bmax<mmin) //take everything from start2 block
    {
       memcpy(out,start2,m2b*sizeof(History<BITS>));
       start2+=m2b;
       out+=m2b;
    }
    else      //blocks overlap merge blocks
    {
      History<BITS>* tmpend1=start1+m1b, *tmpend2=start2+m2b;
      for(; start1<tmpend1 && start2 < tmpend2 ; out++)
      {
        if(*start2 < *start1)
          *out=*start2++;
        else
          *out=*start1++;
      }
    }
  }
}
template<int DIM, class LOCS> template<class BITS>
void SFC<DIM,LOCS>::Parallel()
{
  switch (mergemode)
  {
    case 0:case 1:
            Parallel0<BITS>();
            break;
    case 2: case 3: case 4:
            Parallel1<BITS>();
            break;
    case 5: 
            Parallel2<BITS>();
            break;
    case 6: 
            Parallel3<BITS>();
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

template<class BITS>
struct Comm_Msg
{
  History<BITS> *buffer;
  unsigned int size;
  MPI_Request request;
};
template<class BITS>
struct Comm_Partner
{
  History<BITS> *buffer;
  unsigned int remaining;
  unsigned int rank;
  queue<Comm_Msg<BITS> > in_transit;
};
template<int DIM, class LOCS> template<class BITS>
void SFC<DIM,LOCS>::Parallel3()
{
  vector<Comm_Partner<BITS> > spartners(4), rpartners(4);
  vector<MPI_Request> rreqs(rpartners.size()), sreqs(spartners.size());
  vector<int> rindices(rreqs.size()),sindices(sreqs.size());
#ifdef _TIMESFC_
  start=timer->currentSeconds();
#endif
  vector<History<BITS> > myhistories(n);//,recv_histories(n),merge_histories(n),temp_histories(n);

  //calculate local curves 
  SerialH<BITS>(&myhistories[0]);  //Saves results in sendbuf
  
#ifdef _TIMESFC_
  finish=timer->currentSeconds();
  timers[0]+=finish-start;
  start=timer->currentSeconds();
#endif 
  /*
  cout << rank << ": histories:";
  for(unsigned int i=0;i<n;i++)
  {
    cout << (int)myhistories[i].bits << " ";
  } 
  cout << endl;
  */
  
  //calculate b
  b=(int)ceil(log(2.0*P)/log(2.0));
  if(b>refinements*DIM)
      b=refinements*DIM;

  //calcualte buckets
  buckets=1<<b;
  //cout << rank << ": bits for histogram:" << b << " buckets:" << buckets << endl;
  
  //create local histogram and cuts
  vector <BITS> histogram(buckets+P+1);
  vector <BITS> recv_histogram(buckets+P+1);
  vector <BITS> sum_histogram(buckets+P+1);
  vector <BITS> next_recv_histogram(buckets+P+1);

  histogram[buckets]=0;
  histogram[buckets+1]=buckets;
 
  //cout << rank << ": creating histogram\n";
  ComputeLocalHistogram<BITS>(&histogram[0], myhistories);
  //cout << rank << ": done creating histogram\n";

#ifdef _TIMESFC_
  finish=timer->currentSeconds();
  timers[6]+=finish-start;
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
        MPI_Isend(&sum_histogram[0],(buckets+next_group.size+1)*sizeof(BITS),MPI_BYTE,next_partner_rank,0,Comm,&request);
        hsreqs.push_back(request);
        //start recv
        //start send
        MPI_Irecv(&next_recv_histogram[0],(buckets+next_partner_group.size+1)*sizeof(BITS),MPI_BYTE,next_partner_rank,0,Comm,&rreq);
      }
      else
      {
        //partner doesn't exist
        //no send needed

        //recieve from rank 0 in partner group
        //cout << rank << ": recieving from: " << next_partner_group.start_rank << endl;
        //start send
        MPI_Irecv(&next_recv_histogram[0],(buckets+next_partner_group.size+1)*sizeof(BITS),MPI_BYTE,next_partner_group.start_rank,0,Comm,&rreq);
      }       
        
      if(next_group.size<next_partner_group.size && next_local_rank==0)
      {
        MPI_Request request;
        //send to last one in partner group
        //cout << rank << ": sending additional to: " << next_partner_group.start_rank+next_partner_group.size-1 << endl;
        //start send
        MPI_Isend(&sum_histogram[0],(buckets+next_group.size+1)*sizeof(BITS),MPI_BYTE,next_partner_group.start_rank+next_partner_group.size-1,0,Comm,&request);
        hsreqs.push_back(request);
      }
    }
  }
#ifdef _TIMESFC_
  finish=timer->currentSeconds();
  timers[3]+=finish-start;
  start=timer->currentSeconds();
#endif 
  
  vector<History<BITS> > recv_histories(n),merge_histories,temp_histories;
  
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
        MPI_Isend(&sum_histogram[0],(buckets+next_group.size+1)*sizeof(BITS),MPI_BYTE,next_partner_rank,0,Comm,&request);
        hsreqs.push_back(request);
        //start recv
        MPI_Irecv(&next_recv_histogram[0],(buckets+next_partner_group.size+1)*sizeof(BITS),MPI_BYTE,next_partner_rank,0,Comm,&rreq);
      }
      else
      {
        //partner doesn't exist
        //no send needed

        //recieve from rank 0 in partner group
        //cout << rank << ": recieving from: " << next_partner_group.start_rank << endl;
        //start send
        MPI_Irecv(&next_recv_histogram[0],(buckets+next_partner_group.size+1)*sizeof(BITS),MPI_BYTE,next_partner_group.start_rank,0,Comm,&rreq);
      }       
        
      if(next_group.size<next_partner_group.size && next_local_rank==0)
      {
        MPI_Request request;
        //send to last one in partner group
        //cout << rank << ": sending additional to: " << next_partner_group.start_rank+next_partner_group.size-1 << endl;
        //start send
        MPI_Isend(&sum_histogram[0],(buckets+next_group.size+1)*sizeof(BITS),MPI_BYTE,next_partner_group.start_rank+next_partner_group.size-1,0,Comm,&request);
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
        
      BITS oldstart=histogram[buckets+local_rank],oldend=histogram[buckets+local_rank+1];
      //cout << rank << ": oldstart:" << oldstart << " oldend:" << oldend << endl;
        
      //calculate send count
      for(int p=0;p<parent_group.size;p++)
      {
        //i own old histogram from buckets oldstart to oldend
        //any elements between oldstart and oldend that do not belong on me according to the new cuts must be sent
        //cout << rank << ": sum_histogram[buckets+p]:" << sum_histogram[buckets+p] << " sum_histogram[buckets+p+1]:" << sum_histogram[buckets+p+1] << endl; 
        BITS start=max(oldstart,sum_histogram[buckets+p]),end=min(oldend,sum_histogram[buckets+p+1]);
        for(unsigned int bucket=start;bucket<end;bucket++)
        {
           sendcounts[p]+=histogram[bucket];
        }
      }
        
      //calculate recv count
      //i will recieve from every processor that owns a bucket assigned to me
      //ownership is determined by that processors old histogram and old cuts
       
      BITS newstart=sum_histogram[buckets+next_local_rank],newend=sum_histogram[buckets+next_local_rank+1];
      //cout << rank << ": newstart: " << newstart << " newend:" << newend << endl;

      BITS *lefthistogram,*righthistogram;
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
        BITS start=max(newstart,lefthistogram[buckets+p]), end=min(newend,lefthistogram[buckets+p+1]);
        for(unsigned int bucket=start;bucket<end;bucket++)
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
      //cout << rank << " resizing histories to:" << newn << endl;
      
      recv_histories.resize(newn);
      
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
     spartners.resize(0);
     rpartners.resize(0);
     //create merging partners
      for(int p=0;p<parent_group.size;p++)
      {
          
        if(p==next_local_rank)
          continue;
          
        //start send
        if(sendcounts[p]!=0)
        {
          Comm_Partner<BITS> partner;
          partner.buffer=&myhistories[senddisp[p]];
          partner.remaining=sendcounts[p];
          partner.rank=parent_group.start_rank+p;
          spartners.push_back(partner);
        }
          
        //start recv
        if(recvcounts[p]!=0)
        {
          Comm_Partner<BITS> partner;
          partner.buffer=&recv_histories[recvdisp[p]];
          partner.remaining=recvcounts[p];
          partner.rank=parent_group.start_rank+p;
          rpartners.push_back(partner);
        }
      }
#ifdef _TIMESFC_
     finish=timer->currentSeconds();
     timers[2]+=finish-start;
     start=timer->currentSeconds();
#endif
     //cout << rank << ": spartners:" << spartners.size() << " rpartners:" << rpartners.size() << endl;
     //begin sends
     for(unsigned int i=0;i<spartners.size();i++)
     {
        for(unsigned int b=0;b<blocks_in_transit && spartners[i].remaining>0;b++)
        {
          //create message
          Comm_Msg<BITS> msg;
          msg.size=min(comm_block_size,spartners[i].remaining);
          msg.buffer=spartners[i].buffer;
          
          //adjust partner
          spartners[i].buffer+=msg.size;
          spartners[i].remaining-=msg.size;

          //start send
          MPI_Isend(msg.buffer,msg.size*sizeof(History<BITS>),MPI_BYTE,spartners[i].rank,1,Comm,&msg.request);
      
          //add msg to in transit queue
          spartners[i].in_transit.push(msg);    

          //cout << rank << ": started initial sending msg of size: " << msg.size << " to " << spartners[i].rank << endl;
        }
     }
     //begin recvs 
     for(unsigned int i=0;i<rpartners.size();i++)
     {
        for(unsigned int b=0;b<blocks_in_transit && rpartners[i].remaining>0;b++)
        {
          //create message
          Comm_Msg<BITS> msg;
          msg.size=min(comm_block_size,rpartners[i].remaining);
          msg.buffer=rpartners[i].buffer;
          
          //adjust partner
          rpartners[i].buffer+=msg.size;
          rpartners[i].remaining-=msg.size;

          //start send
          MPI_Irecv(msg.buffer,msg.size*sizeof(History<BITS>),MPI_BYTE,rpartners[i].rank,1,Comm,&msg.request);
      
          //add msg to in transit queue
          rpartners[i].in_transit.push(msg);    
          
          //cout << rank << ": started inital recieving msg of size: " << msg.size << " from " << rpartners[i].rank << endl;
        }
      }
#ifdef _TIMESFC_
      finish=timer->currentSeconds();
      timers[5]+=finish-start;
      start=timer->currentSeconds();
#endif      
      
      temp_histories.reserve(newn);
      merge_histories.reserve(newn);
      merge_histories.resize(0);
     
      rreqs.resize(rpartners.size());
      sreqs.resize(spartners.size());
      //fill initial requests
      for(unsigned int i=0;i<rreqs.size();i++)
      {
        rreqs[i]=rpartners[i].in_transit.front().request;
      }
     
      for(unsigned int i=0;i<sreqs.size();i++)
      {
        sreqs[i]=spartners[i].in_transit.front().request;
      }
         
     
      if(recvcounts[next_local_rank]!=0)
      {
        //copy my list to merge buffer
        merge_histories.assign(myhistories.begin()+senddisp[next_local_rank],myhistories.begin()+senddisp[next_local_rank]+sendcounts[next_local_rank]);
      }
      
      //create status and index arrays for waitsome
      rindices.resize(rreqs.size());
      sindices.resize(sreqs.size());
      unsigned int sdone=0,rdone=0;
      //cout << rank << ": sreqs:" << sreqs.size() << " rreqs:" << rreqs.size() << endl;
      while(sdone<sreqs.size() || rdone<rreqs.size())
      {
        if(sdone<sreqs.size())
        {
          int completed;

#ifdef _TIMESFC_
          finish=timer->currentSeconds();
          timers[4]+=finish-start;
          start=timer->currentSeconds();
#endif
          //testsome on sends
          MPI_Testsome(sreqs.size(),&sreqs[0],&completed,&sindices[0],MPI_STATUSES_IGNORE);
#ifdef _TIMESFC_
          finish=timer->currentSeconds();
          timers[5]+=finish-start;
          start=timer->currentSeconds();
#endif

          for(int i=0;i<completed;i++)
          {
            Comm_Partner<BITS> &partner=spartners[sindices[i]];
            //cout << rank << ": completed send to " << partner.rank << endl; 
            partner.in_transit.pop();
            
            //start next send
            if(partner.remaining>0)
            {
              //create message
              Comm_Msg<BITS> new_msg;
              new_msg.size=min(comm_block_size,partner.remaining);
              new_msg.buffer=partner.buffer;

              //adjust partner
              partner.buffer+=new_msg.size;
              partner.remaining-=new_msg.size;

              //start send
              MPI_Isend(new_msg.buffer,new_msg.size*sizeof(History<BITS>),MPI_BYTE,partner.rank,1,Comm,&new_msg.request);

              //add msg to in transit queue
              partner.in_transit.push(new_msg);
              //cout << rank << ": started sending msg of size: " << new_msg.size << " to " << partner.rank << endl;
            }
            
            //reset sreqs
            if(!partner.in_transit.empty())
            {
              sreqs[sindices[i]]=partner.in_transit.front().request;
            }
            else
            {
              //cout << rank << ": done sending to " << partner.rank << endl;
              sdone++;
            }
          }
        }
    
        if(rdone<rreqs.size())
        {
          int completed;
#ifdef _TIMESFC_
          finish=timer->currentSeconds();
          timers[4]+=finish-start;
          start=timer->currentSeconds();
#endif
          //testsome on recvs
          MPI_Testsome(rreqs.size(),&rreqs[0],&completed,&rindices[0],MPI_STATUSES_IGNORE);
#ifdef _TIMESFC_
          finish=timer->currentSeconds();
          timers[5]+=finish-start;
          start=timer->currentSeconds();
#endif

          for(int i=0;i<completed;i++)
          {
            Comm_Partner<BITS> &partner=rpartners[rindices[i]];
            //cout << rank << ": completed recieve from " << partner.rank << endl; 

            Comm_Msg<BITS> msg=partner.in_transit.front();
            partner.in_transit.pop();
            
            //start next recv
            if(partner.remaining>0)
            {
              //create message
              Comm_Msg<BITS> new_msg;
              new_msg.size=min(comm_block_size,partner.remaining);
              new_msg.buffer=partner.buffer;

              //adjust partner
              partner.buffer+=new_msg.size;
              partner.remaining-=new_msg.size;

              //start recv
              MPI_Irecv(new_msg.buffer,new_msg.size*sizeof(History<BITS>),MPI_BYTE,partner.rank,1,Comm,&new_msg.request);

              //add msg to in transit queue
              partner.in_transit.push(new_msg);
              
              //cout << rank << ": started recieving msg of size: " << new_msg.size << " from " << partner.rank << endl;
            } 
            //reset rreqs
            if(!partner.in_transit.empty())
            {
              rreqs[rindices[i]]=partner.in_transit.front().request;
            }
            else
            {
              //cout << rank << ": done recieving from " << partner.rank << endl;
              rdone++;
            }
           
            if(merge_histories.size()==0)
            {
               merge_histories.resize(msg.size);
               memcpy(&merge_histories[0],msg.buffer,msg.size*sizeof(History<BITS>));
            }
            else
            {
              //merge in transit message
              History<BITS> *start1=&merge_histories[0], *end1=start1+merge_histories.size(), *start2=msg.buffer,*end2=start2+msg.size;
              temp_histories.resize( (end1-start1)+(end2-start2));
              History<BITS> *out=&temp_histories[0]; 
            
              //copy whole list if possible
              if(*start1>=*(end2-1))
              {
                memcpy(out,start2,(end2-start2)*sizeof(History<BITS>));
                memcpy(out+(end2-start2),start1,(end1-start1)*sizeof(History<BITS>));
              }
              else if(*start2>=*(end1-1))
              {
                memcpy(out,start1,(end1-start1)*sizeof(History<BITS>));
                memcpy(out+(end1-start1),start2,(end2-start2)*sizeof(History<BITS>));
              }
              else
              {
                for(;start1<end1 && start2<end2; out++)
                {
                  if(*start2<*start1)
                    *out=*start2++;
                  else
                    *out=*start1++;    
                }
                if(start1!=end1)
                  memcpy(out,start1,(end1-start1)*sizeof(History<BITS>));
                else if(start2!=end2)
                  memcpy(out,start2,(end2-start2)*sizeof(History<BITS>));
              }     
              merge_histories.swap(temp_histories); 
            }
            
          } //end for completed
#ifdef _TIMESFC_
          finish=timer->currentSeconds();
          timers[4]+=finish-start;
          start=timer->currentSeconds();
#endif
        } //end if rdone!=rsize
      } //end while rdone!=rsize && sdone!=ssize

      
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
  //cout << rank << ": all done!\n"; 
}
template<int DIM, class LOCS> template<class BITS>
void SFC<DIM,LOCS>::Parallel2()
{
  int total_recvs=0;
  int num_recvs=0;
#ifdef _TIMESFC_
  start=timer->currentSeconds();
#endif
  vector<History<BITS> > myhistories(n);//,recv_histories(n),merge_histories(n),temp_histories(n);

  //calculate local curves 
  SerialH<BITS>(&myhistories[0]);  //Saves results in sendbuf
  
#ifdef _TIMESFC_
  finish=timer->currentSeconds();
  timers[0]+=finish-start;
  start=timer->currentSeconds();
#endif 
  /*
  cout << rank << ": histories:";
  for(unsigned int i=0;i<n;i++)
  {
    cout << (int)myhistories[i].bits << " ";
  } 
  cout << endl;
  */
  
  //calculate b
  b=(int)ceil(log(2.0*P)/log(2.0));
  if(b>refinements*DIM)
      b=refinements*DIM;

  //calcualte buckets
  buckets=1<<b;
  //cout << rank << ": bits for histogram:" << b << " buckets:" << buckets << endl;
  
  //create local histogram and cuts
  vector <BITS> histogram(buckets+P+1);
  vector <BITS> recv_histogram(buckets+P+1);
  vector <BITS> sum_histogram(buckets+P+1);
  vector <BITS> next_recv_histogram(buckets+P+1);

  histogram[buckets]=0;
  histogram[buckets+1]=buckets;
 
  //cout << rank << ": creating histogram\n";
  ComputeLocalHistogram<BITS>(&histogram[0], myhistories);
  //cout << rank << ": done creating histogram\n";

#ifdef _TIMESFC_
  finish=timer->currentSeconds();
  timers[6]+=finish-start;
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
        MPI_Isend(&sum_histogram[0],(buckets+next_group.size+1)*sizeof(BITS),MPI_BYTE,next_partner_rank,stages,Comm,&request);
        hsreqs.push_back(request);
        //start recv
        //start send
        MPI_Irecv(&next_recv_histogram[0],(buckets+next_partner_group.size+1)*sizeof(BITS),MPI_BYTE,next_partner_rank,stages,Comm,&rreq);
      }
      else
      {
        //partner doesn't exist
        //no send needed

        //recieve from rank 0 in partner group
        //cout << rank << ": recieving from: " << next_partner_group.start_rank << endl;
        //start send
        MPI_Irecv(&next_recv_histogram[0],(buckets+next_partner_group.size+1)*sizeof(BITS),MPI_BYTE,next_partner_group.start_rank,stages,Comm,&rreq);
      }       
        
      if(next_group.size<next_partner_group.size && next_local_rank==0)
      {
        MPI_Request request;
        //send to last one in partner group
        //cout << rank << ": sending additional to: " << next_partner_group.start_rank+next_partner_group.size-1 << endl;
        //start send
        MPI_Isend(&sum_histogram[0],(buckets+next_group.size+1)*sizeof(BITS),MPI_BYTE,next_partner_group.start_rank+next_partner_group.size-1,stages,Comm,&request);
        hsreqs.push_back(request);
      }
    }
  }
#ifdef _TIMESFC_
  finish=timer->currentSeconds();
  timers[3]+=finish-start;
  start=timer->currentSeconds();
#endif 
  vector<History<BITS> > recv_histories(n),merge_histories,temp_histories;
  
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
        MPI_Isend(&sum_histogram[0],(buckets+next_group.size+1)*sizeof(BITS),MPI_BYTE,next_partner_rank,stage-1,Comm,&request);
        hsreqs.push_back(request);
        //start recv
        MPI_Irecv(&next_recv_histogram[0],(buckets+next_partner_group.size+1)*sizeof(BITS),MPI_BYTE,next_partner_rank,stage-1,Comm,&rreq);
      }
      else
      {
        //partner doesn't exist
        //no send needed

        //recieve from rank 0 in partner group
        //cout << rank << ": recieving from: " << next_partner_group.start_rank << endl;
        //start send
        MPI_Irecv(&next_recv_histogram[0],(buckets+next_partner_group.size+1)*sizeof(BITS),MPI_BYTE,next_partner_group.start_rank,stage-1,Comm,&rreq);
      }       
        
      if(next_group.size<next_partner_group.size && next_local_rank==0)
      {
        MPI_Request request;
        //send to last one in partner group
        //cout << rank << ": sending additional to: " << next_partner_group.start_rank+next_partner_group.size-1 << endl;
        //start send
        MPI_Isend(&sum_histogram[0],(buckets+next_group.size+1)*sizeof(BITS),MPI_BYTE,next_partner_group.start_rank+next_partner_group.size-1,stage-1,Comm,&request);
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
        
      BITS oldstart=histogram[buckets+local_rank],oldend=histogram[buckets+local_rank+1];
      //cout << rank << ": oldstart:" << oldstart << " oldend:" << oldend << endl;
        
      //calculate send count
      for(int p=0;p<parent_group.size;p++)
      {
        //i own old histogram from buckets oldstart to oldend
        //any elements between oldstart and oldend that do not belong on me according to the new cuts must be sent
        //cout << rank << ": sum_histogram[buckets+p]:" << sum_histogram[buckets+p] << " sum_histogram[buckets+p+1]:" << sum_histogram[buckets+p+1] << endl; 
        BITS start=max(oldstart,sum_histogram[buckets+p]),end=min(oldend,sum_histogram[buckets+p+1]);
        for(unsigned int bucket=start;bucket<end;bucket++)
        {
           sendcounts[p]+=histogram[bucket];
        }
      }
        
      //calculate recv count
      //i will recieve from every processor that owns a bucket assigned to me
      //ownership is determined by that processors old histogram and old cuts
       
      BITS newstart=sum_histogram[buckets+next_local_rank],newend=sum_histogram[buckets+next_local_rank+1];
      //cout << rank << ": newstart: " << newstart << " newend:" << newend << endl;

      BITS *lefthistogram,*righthistogram;
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
        BITS start=max(newstart,lefthistogram[buckets+p]), end=min(newend,lefthistogram[buckets+p+1]);
        for(unsigned int bucket=start;bucket<end;bucket++)
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
      //cout << rank << " resizing histories to:" << newn << endl;
      
      recv_histories.resize(newn);
      
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
        if(sendcounts[p]>0)
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
        if(recvcounts[p]>0)
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
     
     temp_histories.reserve(newn);
     merge_histories.reserve(newn);
     merge_histories.resize(0);
  
      
     unsigned int i=0;
     //populate initial merge buffer
     if(recvcounts[next_local_rank]!=0)
     {
       //copy my list to merge buffer
       merge_histories.assign(myhistories.begin()+senddisp[next_local_rank],myhistories.begin()+senddisp[next_local_rank]+sendcounts[next_local_rank]);
     }
     else if(!rreqs.empty()) //wait for first receive
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
       merge_histories.resize(recvcounts[p]);
       //copy list to merge buffer
       memcpy(&merge_histories[0],&recv_histories[recvdisp[p]],recvcounts[p]*sizeof(History<BITS>));
       //merge_histories.assign(recv_histories.begin()+recvdisp[p],recv_histories.begin()+recvdisp[p]+recvcounts[p]);
       i++;
     }
     
      vector<MPI_Status> statuses(rreqs.size());
      vector<int> indices(rreqs.size());
      for(;i<rreqs.size();i++)
      {
        int completed;
        //MPI_Status status;
        //int index;
       
       
#ifdef _TIMESFC_
        finish=timer->currentSeconds();
        timers[4]+=finish-start;
        start=timer->currentSeconds();
#endif
        //wait any
        // MPI_Waitany(rreqs.size(),&rreqs[0],&index,&status);
        MPI_Waitsome(rreqs.size(),&rreqs[0],&completed,&indices[0],&statuses[0]);
#ifdef _TIMESFC_
        finish=timer->currentSeconds();
        timers[5]+=finish-start;
        start=timer->currentSeconds();
#endif
        for(int j=0;j<completed;j++)
        {
          int p=statuses[j].MPI_SOURCE-parent_group.start_rank; 
        
          History<BITS> *start1=&merge_histories[0], *end1=start1+merge_histories.size(), *start2=&recv_histories[recvdisp[p]],*end2=start2+recvcounts[p], *out=&temp_histories[0];

          temp_histories.resize( (end1-start1)+(end2-start2));
        
          //copy whole list if possible
          if(*start1>=*(end2-1))
          {
            memcpy(out,start2,(end2-start2)*sizeof(History<BITS>));
            memcpy(out+(end2-start2),start1,(end1-start1)*sizeof(History<BITS>));
          }
          else if(*start2>=*(end1-1))
          {
            memcpy(out,start1,(end1-start1)*sizeof(History<BITS>));
            memcpy(out+(end1-start1),start2,(end2-start2)*sizeof(History<BITS>));
          }
          else
          {
            for(;start1<end1 && start2<end2; out++)
            {
              if(*start2<*start1)
                *out=*start2++;
              else
                *out=*start1++;    
            }
            if(start1!=end1)
              memcpy(out,start1,(end1-start1)*sizeof(History<BITS>));
            else if(start2!=end2)
              memcpy(out,start2,(end2-start2)*sizeof(History<BITS>));
          }
            
          merge_histories.swap(temp_histories); 
        }
      }

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
  
  vector<BITS> histograms,cuts;
  CalculateHistogramsAndCuts<BITS>(histograms,cuts,myhistories);
  
  //build send counts and displacements
  vector<int> sendcounts(P,0);
  vector<int> recvcounts(P,0);
  vector<int> senddisp(P,0);
  vector<int> recvdisp(P,0);
  
  for(int p=0;p<P;p++)
  {
    //calculate send count
      //my row of the histogram summed up across buckets assigned to p
    for(BITS bucket=cuts[p];bucket<cuts[p+1];bucket++)
    {
       sendcounts[p]+=histograms[rank*buckets+bucket];     
    }
    
    //calculate recv count
      //my bucket colums of the histogram summed up for each processor
    for(BITS bucket=cuts[rank];bucket<cuts[rank+1];bucket++)
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
  
  if(mergemode==2)
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
  else if(mergemode==3 || mergemode==4)
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
#ifdef _TIMESFC_
    finish=timer->currentSeconds();
    timers[3]+=finish-start;
    start=timer->currentSeconds();
#endif
    if(mergemode==3)
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
  timers[2]+=finish-start;
  start=timer->currentSeconds();
#endif
        MPI_Waitany(rreqs.size(),&rreqs[0],&index,&status);
#ifdef _TIMESFC_
  finish=timer->currentSeconds();
  timers[3]+=finish-start;
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
    else if (mergemode==4)
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
  timers[2]+=finish-start;
  start=timer->currentSeconds();
#endif
        //wait any
        MPI_Waitany(rreqs.size(),&rreqs[0],&index,&status);
#ifdef _TIMESFC_
  finish=timer->currentSeconds();
  timers[3]+=finish-start;
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
    
#ifdef _TIMESFC_
  finish=timer->currentSeconds();
  timers[3]+=finish-start;
  start=timer->currentSeconds();
#endif
    //wait for sends
    for(unsigned int i=0;i<sreqs.size();i++)
    {
     MPI_Status status;
     MPI_Wait(&sreqs[i],&status);
    }
  }
#ifdef _TIMESFC_
  finish=timer->currentSeconds();
  timers[2]+=finish-start;
  start=timer->currentSeconds();
#endif
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
#ifdef _TIMESFC_
  start=timer->currentSeconds();
#endif
  vector<History<BITS> > histories(n);
  unsigned int i;
 
  SerialH<BITS>(&histories[0]);  //Saves results in sendbuf
#ifdef _TIMESFC_
  finish=timer->currentSeconds();
  timers[0]+=finish-start;
#endif
  vector<History<BITS> > rbuf, mbuf(n);
  
 if(mergemode==0)
 {
#ifdef _TIMESFC_
    start=timer->currentSeconds();
#endif
    PrimaryMerge<BITS>(histories,rbuf,mbuf);
#ifdef _TIMESFC_
    finish=timer->currentSeconds();
    timers[1]+=finish-start;
#endif
 }
  
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
  /*
  for(unsigned int i=1;i<histories.size();i++)
  {
     if(histories[i].bits-1!=histories[i-1].bits)
        cout << rank << ": sfc error\n ";
  }
  */
}

template <class BITS>
struct MergeInfo
{
  BITS min;
  BITS max;
  unsigned int n;
};

#define ASCENDING 0
#define DESCENDING 1
template<int DIM, class LOCS> template<class BITS>
int SFC<DIM,LOCS>::MergeExchange(int to,vector<History<BITS> > &sendbuf, vector<History<BITS> >&recievebuf, vector<History<BITS> > &mergebuf)
{
  float inv_denom=1.0/sizeof(History<BITS>);
  //cout << rank <<  ": Merge Exchange started with " << to << endl;
  int direction= (int) (rank>to);
  //BITS emax, emin;
  queue<MPI_Request> squeue, rqueue;
  unsigned int n2;

  MergeInfo<BITS> myinfo,theirinfo;

  MPI_Request srequest, rrequest;
  MPI_Status status;
  
  myinfo.n=n;
  if(n!=0)
  {
    myinfo.min=sendbuf[0].bits;
    myinfo.max=sendbuf[n-1].bits;
  }
  //cout << rank << " n:" << n << " min:" << (int)myinfo.min << "max:" << (int)myinfo.max << endl;

  MPI_Isend(&myinfo,sizeof(myinfo),MPI_BYTE,to,0,Comm,&srequest);
  MPI_Irecv(&theirinfo,sizeof(theirinfo),MPI_BYTE,to,0,Comm,&rrequest);
  MPI_Wait(&rrequest,&status);
  MPI_Wait(&srequest,&status);

  if(myinfo.n==0 || theirinfo.n==0)
  {
     return 0;
  }
  
  //min_max exchange
  if(direction==ASCENDING)
  {
          
    if(myinfo.max<=theirinfo.min) //no exchange needed
        return 0;
    else if(myinfo.min>=theirinfo.max) //full exchange needed
    {
       mergebuf.resize(theirinfo.n);
       MPI_Isend(&sendbuf[0],n*sizeof(History<BITS>),MPI_BYTE,to,0,Comm,&srequest);
       MPI_Irecv(&mergebuf[0],theirinfo.n*sizeof(History<BITS>),MPI_BYTE,to,0,Comm,&rrequest);
       MPI_Wait(&rrequest,&status);
       MPI_Wait(&srequest,&status);
       sendbuf.swap(mergebuf);
       mergebuf.resize(theirinfo.n);
       n=theirinfo.n;
       return 1;
    }
  }
  else
  {
    if(theirinfo.max<=myinfo.min) //no exchange needed
        return 0;
    else if(myinfo.max<=theirinfo.min) //full exchange needed
    {
       mergebuf.resize(theirinfo.n);
       MPI_Isend(&sendbuf[0],n*sizeof(History<BITS>),MPI_BYTE,to,0,Comm,&srequest);
       MPI_Irecv(&mergebuf[0],theirinfo.n*sizeof(History<BITS>),MPI_BYTE,to,0,Comm,&rrequest);
       MPI_Wait(&rrequest,&status);
       MPI_Wait(&srequest,&status);
       sendbuf.swap(mergebuf);
       mergebuf.resize(theirinfo.n);
       n=theirinfo.n;
       return 1;
    }
  }
  n2=theirinfo.n;
  //cout << rank << ": Max-min done\n";
  
  recievebuf.resize(n2);
  
  History<BITS> *sbuf=&sendbuf[0], *rbuf=&recievebuf[0], *mbuf=&mergebuf[0];
  History<BITS> *msbuf=sbuf, *mrbuf=rbuf;
  
  unsigned int nsend=n;
  unsigned int nrecv=n2;
  //sample exchange
  unsigned int minn=min(n,n2);
  unsigned int sample_size=(int)(minn*sample_percent);

  if(sample_size>=5)
  {
//    cout << rank << " creating samples\n";
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
//    cout << "exchanging samples\n";
    //exchange samples
    MPI_Isend(mysample,sample_size*sizeof(BITS),MPI_BYTE,to,1,Comm,&srequest);
    MPI_Irecv(theirsample,sample_size*sizeof(BITS),MPI_BYTE,to,1,Comm,&rrequest);
  
    MPI_Wait(&rrequest,&status);
    MPI_Wait(&srequest,&status);
    
//    cout << "done exchanging samples\n";
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
//  cout << sremaining << " " << rremaining << endl;
  unsigned int sent=0, recvd=0;
  //unsigned int merged=0;
//  cout << rank << " Block size: " << comm_block_size << endl;  
  
  
  if(direction==ASCENDING)
  {
    //Merge Ascending
    //Send Descending
    //Recieve Ascending
      
    History<BITS> *start1=msbuf,*start2=mrbuf,*end1=start1+n,*end2=mrbuf,*out=mbuf, *outend=mbuf+n;
    //position buffers
    sbuf+=n;
    
    while(block_count<blocks_in_transit)
    {
      //send
      if(sremaining>0)
      {
        int send=min(sremaining,(int)comm_block_size);
        sbuf-=send;
        MPI_Isend(sbuf,send*sizeof(History<BITS>),MPI_BYTE,to,1,Comm,&srequest);
        squeue.push(srequest);
        sent+=send;
        sremaining-=send;
      }
      
      //recieve
      if(rremaining>0)
      {
        int recv=min(rremaining,(int)comm_block_size);
        MPI_Irecv(rbuf+recvd,recv*sizeof(History<BITS>),MPI_BYTE,to,1,Comm,&rrequest);
        rqueue.push(rrequest);
        recvd+=recv;
        rremaining-=recv;
      }
    
      block_count++;
    }
    while(!rqueue.empty())
    {
      MPI_Wait(&(rqueue.front()),&status);

      //start next communication
      //send next block
      if(sremaining>0)
      {
        int send=min(sremaining,(int)comm_block_size);
        sbuf-=send;
        MPI_Isend(sbuf,send*sizeof(History<BITS>),MPI_BYTE,to,1,Comm,&srequest);
        squeue.push(srequest);
        sent+=send;
        sremaining-=send;
      }
      
      if(rremaining>0)
      {
        int recv=min(rremaining,(int)comm_block_size);
        MPI_Irecv(rbuf+recvd,recv*sizeof(History<BITS>),MPI_BYTE,to,1,Comm,&rrequest);
        rqueue.push(rrequest);
        recvd+=recv;
        rremaining-=recv;
      }
      
      rqueue.pop();
      
      MPI_Get_count(&status,MPI_BYTE,&b);
      b=int(b*inv_denom);
      end2+=b;
     
      //while there is more merging needed and I have recieved histories 
      while(start2<end2 && out<outend)
      {
        //size of merge blocks
        int mb=min(int(end1-start1),(int)merge_block_size);
        int rb=min(int(end2-start2),(int)merge_block_size);
        
        BITS mmin=start1->bits, mmax=(start1+mb-1)->bits, bmin=start2->bits,bmax=(start2+rb-1)->bits; 

        if(mmax<=bmin) //take everything from mine
        {
          int s=min(mb,int(outend-out));
          //cout << rank << ": take " << s << " from mine\n";
          memcpy(out,start1,s*sizeof(History<BITS>));
          start1+=s;
          out+=s;
        }
        else if (bmax<mmin) //take everything from theirs
        {
          int s=min(rb,int(outend-out));
          //cout << rank << ": take " << s << " from theirs\n";
          memcpy(out,start2,s*sizeof(History<BITS>));
          start2+=s;
          out+=s;
        }
        else  //lists overlap, merge them
        {
          History<BITS>* tmpend2=start2+rb,*tmpend1=start1+mb;
          for(; start2 < tmpend2 && start1<tmpend1 && out<outend ; out++)
          {
            if(*start2 < *start1)
              *out=*start2++;
            else
              *out=*start1++;
          }
        }
      }
  
    
      //wait for a send, it should be done now
      MPI_Wait(&(squeue.front()),&status);
      squeue.pop();
    }
    int rem=outend-out;
    if(rem>0)
    {
      memcpy(out,start1,rem*sizeof(History<BITS>));    
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

    History<BITS> *start1=msbuf,*start2=mrbuf,*end1=start1-n,*end2=mrbuf,*out=mbuf-1, *outend=out-n;
      
    while(block_count<blocks_in_transit)
    {
      //send
      if(sremaining>0)
      {
        int send=min(sremaining,(int)comm_block_size);
        MPI_Isend(sbuf+sent,send*sizeof(History<BITS>),MPI_BYTE,to,1,Comm,&srequest);
        squeue.push(srequest);
        sent+=send;
        sremaining-=send;
      }

      if(rremaining>0)
      {
        int recv=min(rremaining,(int)comm_block_size);
        rbuf-=recv;
        MPI_Irecv(rbuf,recv*sizeof(History<BITS>),MPI_BYTE,to,1,Comm,&rrequest);
        rqueue.push(rrequest);
        recvd+=recv;
        rremaining-=recv;
      }
      block_count++;
    }
    while(!rqueue.empty())
    {
      MPI_Wait(&(rqueue.front()),&status);
    
      //start next communication
      if(sremaining>0)
      {
        int send=min(sremaining,(int)comm_block_size);
        MPI_Isend(sbuf+sent,send*sizeof(History<BITS>),MPI_BYTE,to,1,Comm,&srequest);
        squeue.push(srequest);
        sent+=send;
        sremaining-=send;
      }
      if(rremaining>0)
      {
        int recv=min(rremaining,(int)comm_block_size);
        rbuf-=recv;
        MPI_Irecv(rbuf,recv*sizeof(History<BITS>),MPI_BYTE,to,1,Comm,&rrequest);
        rqueue.push(rrequest);
        recvd+=recv;
        rremaining-=recv;
      }
      
      rqueue.pop();

      MPI_Get_count(&status,MPI_BYTE,&b);
      b=int(b*inv_denom);

      end2-=b;
      
      while(start2>end2 && out>outend)
      {
        //size of merge blocks
        int mb=min(int(start1-end1),(int)merge_block_size);
        int rb=min(int(start2-end2),(int)merge_block_size);
      
        BITS mmin=(start1-mb+1)->bits, mmax=start1->bits, bmin=(start2-rb+1)->bits,bmax=start2->bits; 
        
        if(mmin>bmax) //take everything from mine
        {
          int s=min(mb,int(out-outend));
          memcpy(out-s+1,start1-s+1,s*sizeof(History<BITS>));
          start1-=s;
          out-=s;
        }
        else if (bmin>=mmax) //take everything from theirs
        {
          int s=min(rb,int(out-outend));
          memcpy(out-s+1,start2-s+1,s*sizeof(History<BITS>));
          start2-=s;
          out-=s;
        }
        else  //lists overlap, merge them
        {
          History<BITS> *tmpend2=start2-rb, *tmpend1=start1-mb;
          for(; start2 > tmpend2 && start1 > tmpend1 && out>outend  ; out--)
          {
            if(*start2 > *start1)
              *out=*start2--;
            else
              *out=*start1--;
          }
        }
      }


      MPI_Wait(&(squeue.front()),&status);
      squeue.pop();
    }

    int rem=out-outend;
    if(rem>0)
    {
      memcpy(out-rem+1,start1-rem+1,rem*sizeof(History<BITS>));
    }

    /*
    if(merged<n) //merge additional elements off of msbuf
    {
      int rem=n-merged;
      memcpy(mbuf-rem,msbuf-rem+1,(rem)*sizeof(History<BITS>));
//      cout << rank << " (DSC) merging more from " << to << endl;
    }
    */
  }
  while(!rqueue.empty())
  {
    MPI_Wait(&(rqueue.front()),&status);
    rqueue.pop();
//    cout << rank << " recieved left over block\n";
  }
  while(!squeue.empty())
  {
    MPI_Wait(&(squeue.front()),&status);
    squeue.pop();
//    cout << rank << " sent left over block\n";
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

    if(rank>=base && rank-base<P)
    {
      send=true;
      if(rank-base<P/2)
      {
        to=rank+(P+1)/2;
      }
      else
      {
        to=rank-(P+1)/2;
      }
    }

    /*
    //if(rank>=base && rank<base+(P>>1))
    if(rank>=base && rank-base<P/2)
    {
      send=true;
      to=rank+((P+1)>>1);
    }
    else if(rank-base>=(P+1)/2 && rank-base<P)
    //else if(rank-((P+1)>>1)>=base && rank-((P+1)>>1)<base+(P>>1))
    {
      send=true;
      to=rank-((P+1)>>1);
    }
    */

    if(send)
    {
      MergeExchange<BITS>(to,histories,rbuf,mbuf);
    }

    //make next stages

    cur.P=((P+1)/2);
    if(cur.P>1)
    {
      cur.base=base+(P/2);
      q.push(cur);
    }

    cur.P=P-((P+1)/2);
    if(cur.P>1)
    {
      cur.base=base;
      q.push(cur);
    }
  }

}
template<int DIM, class LOCS> template<class BITS>
void SFC<DIM,LOCS>::PrimaryMerge2(vector<History<BITS> > &histories, vector<History<BITS> >&rbuf, vector<History<BITS> > &mbuf, int P, int base)
{
  if(P==1)
    return;

  PrimaryMerge2(histories,rbuf,mbuf,(P+1)/2,base+P/2);
  PrimaryMerge2(histories,rbuf,mbuf,P-(P+1)/2,base);
  int to=-1;  
 
  if(rank>=base && rank<base+(P>>1))
  {
    to=rank+((P+1)>>1);
  }
  else if(rank-((P+1)>>1)>=base && rank-((P+1)>>1)<base+(P>>1))
  {
    to=rank-((P+1)>>1);
  }

  if(to!=-1)
  {
    MergeExchange<BITS>(to,histories,rbuf,mbuf);
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
  unsigned int i=1, c=1, val=0, iter=0;
  int mod=(int)ceil(log((float)P)/log(3.0f));
  while(c!=0)
  {
    val=0;
    if(rank%2==0)  //exchange right then left
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
    else  //exchange left then right
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
    iter+=2;
  }

}
template<int DIM, class LOCS>
void SFC<DIM,LOCS>::SetMergeParameters(unsigned int comm_block_size,unsigned int merge_block_size, unsigned int blocks_in_transit, float sample_percent)
{
  this->comm_block_size=comm_block_size;
  this->blocks_in_transit=blocks_in_transit;
  this->sample_percent=sample_percent;
  this->merge_block_size=merge_block_size;
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
