/*

The MIT License

Copyright (c) 1997-2011 Center for the Simulation of Accidental Fires and 
Explosions (CSAFE), and  Scientific Computing and Imaging Institute (SCI), 
University of Utah.

License for the specific language governing rights and limitations under
Permission is hereby granted, free of charge, to any person obtaining a 
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation 
the rights to use, copy, modify, merge, publish, distribute, sublicense, 
and/or sell copies of the Software, and to permit persons to whom the 
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included 
in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS 
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
DEALINGS IN THE SOFTWARE.

*/


#ifndef _SFC
#define _SFC

#include <vector>
#include <queue>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>
#include <climits>
#include <cstring>

#include <Core/Parallel/ProcessorGroup.h>
#include <Core/Thread/Time.h>
#include <CCA/Ports/uintahshare.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Util/Assert.h>
using namespace SCIRun;
using namespace Uintah;
namespace Uintah{

#ifdef _TIMESFC_
double start, finish;
const int TIMERS=7;
double timers[TIMERS]={0};
#endif
enum Curve {HILBERT=0, MORTON=1, GREY=2};
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
  Group()
  {
    size=-1;
    start_rank=-1;
    partner_group=-1;
    parent_group=-1;
  }
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
extern UINTAHSHARE int dir3[][3];
extern UINTAHSHARE int dir2[][3];
extern UINTAHSHARE int dir1[][3];

extern UINTAHSHARE int hinv3[][8];
extern UINTAHSHARE int ginv3[][8];
extern UINTAHSHARE int minv3[][8];
extern UINTAHSHARE int hinv2[][8];
extern UINTAHSHARE int ginv2[][8];
extern UINTAHSHARE int minv2[][8];

extern UINTAHSHARE int horder3[][8];
extern UINTAHSHARE int gorder3[][8];
extern UINTAHSHARE int morder3[][8];
extern UINTAHSHARE int horder2[][8];
extern UINTAHSHARE int gorder2[][8];
extern UINTAHSHARE int morder2[][8];

extern UINTAHSHARE int horient3[][8];
extern UINTAHSHARE int gorient3[][8];
extern UINTAHSHARE int morient3[][8];
extern UINTAHSHARE int horient2[][8];
extern UINTAHSHARE int gorient2[][8];
extern UINTAHSHARE int morient2[][8];

extern UINTAHSHARE int orient1[][8];
extern UINTAHSHARE int order1[][8];
extern UINTAHSHARE int inv1[][8];

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
                std::cout << val;
                mask>>=DIM;
                bits-=DIM;
        }
}

/******************Bin helper functions*****************************/
template<int DIM, class LOCS> struct Binner
{
  static inline unsigned char Bin(LOCS *point, LOCS *center);
};
template<class LOCS> struct Binner<1,LOCS>
{
  static inline unsigned char Bin(LOCS *point, LOCS *center)
  {
    return point[0]<center[0];
  }
};
template<class LOCS> struct Binner<2,LOCS>
{
  static inline unsigned char Bin(LOCS *point, LOCS *center)
  {
    unsigned char bin=0;
    if(point[0]<center[0])
      bin|=1;
    if(point[1]<center[1])
      bin|=2;
    return bin;
  }
};
template<class LOCS> struct Binner<3,LOCS>
{
  static inline unsigned char Bin(LOCS *point, LOCS *center)
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
};

/********************************************************************/
template<class LOCS>
class SFC
{
public:
  SFC(const ProcessorGroup *d_myworld,int dim=0,Curve curve=HILBERT) : curve(curve),n(INT_MAX),refinements(-1), locsv(0), locs(0), orders(0), d_myworld(d_myworld), comm_block_size(3000), blocks_in_transit(3), merge_block_size(100), sample_percent(.1), cleanup(BATCHERS), mergemode(1)
  {
      dimensions[0]=INT_MAX;
      dimensions[1]=INT_MAX;
      dimensions[2]=INT_MAX;
      center[0]=INT_MAX;
      center[1]=INT_MAX;
      center[2]=INT_MAX;

      SetNumDimensions(dim);
  }
  void SetCurve(Curve curve);
  ~SFC() {};
  void GenerateCurve(int mode=0);
  void SetRefinements(int refinements);
  void SetLocalSize(unsigned int n);
  void SetLocations(std::vector<LOCS> *locs);
  void SetOutputVector(std::vector<DistributedIndex> *orders);
  void SetMergeParameters(unsigned int comm_block_size,unsigned int merge_block_size, unsigned int blocks_in_transit, float sample_percent);
  void SetNumDimensions(int dim);
  void SetCommBlockSize(unsigned int b) {comm_block_size=b;};
  void SetMergeBlockSize(unsigned int b) {merge_block_size=b;};
  void SetBlocksInTransit(unsigned int b) {blocks_in_transit=b;};
  void SetSamplePercent(float p) {sample_percent=p;};
  void SetCleanup(CleanupType cleanup) {this->cleanup=cleanup;};
  void SetMergeMode(int mode) {this->mergemode=mode;};
  void SetDimensions(LOCS *dimensions);
  void SetCenter(LOCS *center);
  void SetRefinementsByDelta(LOCS *deltax);

  template<int DIM> void ProfileMergeParameters(int repeat=21);
protected:

  int dim;
  Curve curve;

  SCIRun::Time *timer;

  //order and orientation arrays
  int (*order)[8];
  int (*orientation)[8];
  int (*inverse)[8];

  //direction array
  int (*dir)[3];

  //curve parameters
  LOCS dimensions[3];
  LOCS center[3];
  unsigned int n;
  int refinements;


  //XY(Z) locations of points
  std::vector<LOCS> *locsv;
  LOCS *locs;

  //Output vector
  std::vector<DistributedIndex> *orders;

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
  bool BulletProof(int mode)
  {

    bool retval=true;

    if(dim<1 || dim >3)
    {
      std::cout << "SFC curve: invalid number of dimensions (" << dim << ")\n";
      retval=false;
    }
    if(locsv==0)
    {
      std::cout << "SFC curve: location vector not set\n";
      retval=false;
    }
    if(orders==0)
    {
      std::cout << "SFC curve: orders not set\n";
      retval=false;
    }
    if(center[0]==INT_MAX && center[1]==INT_MAX && center[2]==INT_MAX)
    {
      std::cout << "SFC curve: center not set\n";
      retval=false;
    }
    if(dimensions[0]==INT_MAX && dimensions[1]==INT_MAX && dimensions[2]==INT_MAX)
    {
      std::cout << "SFC curve: dimensions not set\n";
      retval=false;
    }
    if((P>1 && mode==PARALLEL) && n==INT_MAX)
    {
      if(n==INT_MAX)
      {
        std::cout << "SFC curve: Local size not set\n";
        retval=false;
      }
      if(refinements==INT_MAX)
      {
        std::cout << "SFC curve: Refinements not set\n";
        retval=false;
      }
    }
    //check that all points fall with the bounds of the domain.
    int low[3],high[3];
    for(int d=0;d<dim;d++)
    {
      low[d]=int(center[d]-dimensions[d]/2);
      high[d]=int(ceil(center[d]+dimensions[d]/2));
    }

    for(unsigned int i=0;i<n;i++)
    {
      for(int d=0;d<dim;d++)
      {
        if(locs[i*dim+d]<low[d] || locs[i*dim+d]>high[d])
        {
          std::cout << "SFC curve: Points are not bounded by dimensions\n";
          return false;
        }
      }
    }

    return retval;
  }

  template<int DIM> void GenerateDim(int mode);
  template<int DIM> void Serial();
  template<int DIM> void SerialR(DistributedIndex* orders,std::vector<DistributedIndex> *bin, unsigned int n, LOCS *center, LOCS *dimension, unsigned int o=0);

  template<int DIM, class BITS> void SerialH(History<BITS> *histories);
  template<int DIM, class BITS> void SerialHR(DistributedIndex* orders,History<BITS>* corders,std::vector<DistributedIndex> *bin, unsigned int n, LOCS *center, LOCS *dimension,                                     unsigned int o=0, int r=1, BITS history=0);
  template<int DIM, class BITS> void Parallel();
  template<int DIM, class BITS> void Parallel0();
  template<int DIM, class BITS> void Parallel1();
  template<int DIM, class BITS> void Parallel2();
  template<int DIM, class BITS> void Parallel3();
  template<int DIM, class BITS> void ProfileMergeParametersT(int repeat);

  template<class BITS> void BlockedMerge(History<BITS>* start1, History<BITS>* end1, History<BITS>*start2,History<BITS>* end2,History<BITS>* out);


  template<class BITS> void CalculateHistogramsAndCuts(std::vector<BITS> &histograms, std::vector<BITS> &cuts, std::vector<History<BITS> > &histories);
  template<class BITS> void ComputeLocalHistogram(BITS *histogram,std::vector<History<BITS> > &histories);

  template<class BITS> int MergeExchange(int to,std::vector<History<BITS> > &histories, std::vector<History<BITS> >&rbuf, std::vector<History<BITS> > &mbuf);
  template<class BITS> void PrimaryMerge(std::vector<History<BITS> > &histories, std::vector<History<BITS> >&rbuf, std::vector<History<BITS> > &mbuf);
  template<class BITS> void PrimaryMerge2(std::vector<History<BITS> > &histories, std::vector<History<BITS> >&rbuf, std::vector<History<BITS> > &mbuf,int  procs, int base=0);
  template<class BITS> void Cleanup(std::vector<History<BITS> > &histories, std::vector<History<BITS> >&rbuf, std::vector<History<BITS> > &mbuf);
  template<class BITS> void Batchers(std::vector<History<BITS> > &histories, std::vector<History<BITS> >&rbuf, std::vector<History<BITS> > &mbuf);
  template<class BITS> void Linear(std::vector<History<BITS> > &histories, std::vector<History<BITS> >&rbuf, std::vector<History<BITS> > &mbuf);
};

/***********SFC**************************/
template<class LOCS> template<int DIM>
void SFC<LOCS>::ProfileMergeParameters(int repeat)
{
#if SCI_ASSERTION_LEVEL >= 3
  ASSERT(BulletProof(Parallel));
#endif

  rank=d_myworld->myrank();
  Comm=d_myworld->getComm();

  if((int)refinements*DIM<=(int)sizeof(unsigned char)*8)
  {
    ProfileMergeParametersT<DIM,unsigned char>(repeat);
  }
  else if((int)refinements*DIM<=(int)sizeof(unsigned short)*8)
  {
    ProfileMergeParametersT<DIM, unsigned short>(repeat);
  }
  else if((int)refinements*DIM<=(int)sizeof(unsigned int)*8)
  {
    ProfileMergeParametersT<DIM,unsigned int>(repeat);
  }
  else if((int)refinements*DIM<=(int)sizeof(unsigned long)*8)
  {
    ProfileMergeParametersT<DIM,unsigned long>(repeat);
  }
  else
  {
    if((int)refinements*DIM>(int)sizeof(unsigned long long)*8)
    {
      refinements=sizeof(unsigned long long)*8/DIM;
    }
    ProfileMergeParametersT<DIM,unsigned long long>(repeat);
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


template<class LOCS> template <int DIM, class BITS>
void SFC<LOCS>::ProfileMergeParametersT(int repeat)
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
  std::vector<History<BITS> > rbuf(max_block_size), mbuf(max_block_size);
  std::vector<History<BITS> > histories(max_block_size);
  std::vector<History<BITS> > histories_original(max_block_size);

  //sorting input and using it as our test case
  SerialH<BITS>(&histories_original[0]);  //Saves results in sendbuf

  //create merging search parameters...
  std::vector<float> times(repeat);

  //determine block size
  float last,cur;
  int last_dir=0;

  /*
  //run once to avoid some strange problem where the first time running is way off
  sample(cur,block_size,block_size,blocks,sample_size);
  sample(last,block_size+delta_block_size,block_size,blocks,sample_size);
  if(rank==0)
  {
    std::cout << "cur:" << cur << " last:" << last << std::endl;
  }
  */

  sample(cur,block_size,block_size,blocks,sample_size);
  sample(last,block_size+delta_block_size,block_size,blocks,sample_size);

//  if(rank==0)
//  {
//    std::cout << "t1: " << block_size << " " << cur << std::endl;
//    std::cout << "t2: " << block_size+delta_block_size << " " << last << std::endl;
//    std::cout << "-----------------------\n";
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
   //   std::cout << "checking with block_size: " << block_size+delta_block_size*last_dir << std::endl;
    //measure new points
    sample(cur,block_size+delta_block_size*last_dir,block_size+delta_block_size*last_dir,blocks,sample_size);
   // if(rank==0)
   // {
   //   std::cout << "last: " << block_size << " " << last << std::endl;
   //   std::cout << "cur: " << block_size+delta_block_size*last_dir << " " << cur << std::endl;
   //   std::cout << "-----------------------\n";
   // }

    if(fabs(1-cur/last)<sig)  //if t1 & t2 are not signifigantly different
    {
      //if(rank==0)
      //  std::cout << "not signifigantly different\n";
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

      std::swap(cur,last);
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
//    std::cout << "Using block_size=" << block_size <<std::endl;

  blocks++;
  //determine number of blocks to send at a time
  sample(cur,block_size,block_size,blocks,sample_size);

//  if(rank==0)
//    std::cout << "blocks:" << blocks << " " << cur << std::endl;
  //last=999999999;

  while(cur<last)
  {
    std::swap(cur,last);
    blocks++;
    sample(cur,block_size,block_size,blocks,sample_size);
  //  if(rank==0)
  //    std::cout << "blocks:" << blocks << " " << cur << std::endl;
  //  if( fabs(1-cur/last)<sig )
  //    break;
  }
  blocks--;
//  if(rank==0)
//    std::cout << "Using blocks:" << blocks << std::endl;


  //determine merge block size
  std::vector<int> merge_block_sizes;

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
  //  std::cout << "merge_block_size:" << block_size << " " << last << std::endl;
  int merge_size=block_size;

  for(unsigned int i=0;i<merge_block_sizes.size();i++)
  {
    sample(cur,block_size,merge_block_sizes[i],blocks,sample_size);
    //if(rank==0)
    //  std::cout << "merge_block_size:" << merge_block_sizes[i] << " " << cur << std::endl;
    if(fabs(1-cur/last)<sig)
    {
      //if(rank==0)
      //  std::cout << "not signigantly different\n";
    //  break;
    }
    if(cur<last)
    {
      last=cur;
      merge_size=merge_block_sizes[i];
    }
  }
  //if(rank==0)
  //  std::cout << "Using merge_block_size:" << merge_size << std::endl;

  SetMergeParameters(block_size,merge_size,blocks,sample_size);
 // if(rank==0)
 // {
 //   std::cout << "Merge Parmeters:" << block_size << " " << merge_size << " " << blocks << " " << sample_size << " " << last << std::endl;
 // }
  //determine number of procs?
  //determine merge mode?

}

template<class LOCS>
void SFC<LOCS>::GenerateCurve(int mode)
{
  P=d_myworld->size();
  ASSERT(BulletProof(mode));

  switch(dim)
  {
    case 1:
      GenerateDim<1>(mode);
      break;
    case 2:
      GenerateDim<2>(mode);
      break;
    case 3:
      GenerateDim<3>(mode);
      break;
  }
}

template<class LOCS> template<int DIM>
void SFC<LOCS>::GenerateDim(int mode)
{
  if(mode==SERIAL)
  {
    Serial<DIM>();
  }
  else
  {
    //if using cleanup only use Batchers
    if(mergemode==1)
      SetCleanup(BATCHERS);

    //make new sub group if needed?
    rank=d_myworld->myrank();
    Comm=d_myworld->getComm();
    //Pick which generate to use
    if((int)refinements*DIM<=(int)sizeof(unsigned char)*8)
    {
      Parallel<DIM,unsigned char>();
    }
    else if((int)refinements*DIM<=(int)sizeof(unsigned short)*8)
    {
      Parallel<DIM,unsigned short>();
    }
    else if((int)refinements*DIM<=(int)sizeof(unsigned int)*8)
    {
      Parallel<DIM,unsigned int>();
    }
    else if((int)refinements*DIM<=(int)sizeof(unsigned long)*8)
    {
      Parallel<DIM,unsigned long>();
    }
    else
    {
      if((int)refinements*DIM>(int)sizeof(unsigned long long)*8)
      {
        refinements=sizeof(unsigned long long)*8/DIM;
        if(rank==0)
          std::cerr << "Warning: Not enough bits to form full SFC lowering refinements to: " << refinements << std::endl;
      }
      Parallel<DIM,unsigned long long>();
    }
  }
}

template<class LOCS> template<int DIM>
void SFC<LOCS>::Serial()
{
  if(n!=0)
  {
    orders->resize(n);

    DistributedIndex *o=&(*orders)[0];

    for(unsigned int i=0;i<n;i++)
    {
      o[i]=DistributedIndex(i,0);
    }

    std::vector<DistributedIndex> bin[BINS];
    for(int b=0;b<BINS;b++)
    {
      bin[b].reserve(n/BINS);
    }
    //Recursive call
    SerialR<DIM>(o,bin,n,center,dimensions);
  }
}

template<class LOCS> template<int DIM, class BITS>
void SFC<LOCS>::SerialH(History<BITS> *histories)
{
  if(n!=0)
  {
    orders->resize(n);

    DistributedIndex *o=&(*orders)[0];

    for(unsigned int i=0;i<n;i++)
    {
      o[i]=DistributedIndex(i,rank);
    }

    std::vector<DistributedIndex> bin[BINS];
    for(int b=0;b<BINS;b++)
    {
      bin[b].reserve(n/BINS);
    }
    //Recursive call
    SerialHR<DIM,BITS>(o,histories,bin,n,center,dimensions);
  }
}

template<class LOCS> template <int DIM>
void SFC<LOCS>::SerialR(DistributedIndex* orders,std::vector<DistributedIndex> *bin, unsigned int n, LOCS *center, LOCS *dimension, unsigned int o)
{
  LOCS newcenter[BINS][DIM], newdimension[DIM];
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
    b=Binner<DIM,LOCS>::Bin(&locs[orders[i].i*DIM],center);
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
        SerialR<DIM>(orders,bin,size[b],newcenter[b],newdimension,orientation[o][b]);
      }

    }
    else if(size[b]>1 )
    {
        SerialR<DIM>(orders,bin,size[b],newcenter[b],newdimension,orientation[o][b]);
    }
    orders+=size[b];
  }
}

template<class LOCS> template<int DIM, class BITS>
void SFC<LOCS>::SerialHR(DistributedIndex* orders, History<BITS>* corders,std::vector<DistributedIndex> *bin, unsigned int n, LOCS *center, LOCS *dimension, unsigned int o, int r, BITS history)
{
  LOCS newcenter[BINS][DIM], newdimension[DIM];
  unsigned int size[BINS];

  unsigned char b;
  unsigned int i;
  unsigned int index=0;

  if(n==1)
  {
    LOCS newcenter[DIM];
    //initialize newdimension and newcenter
    for(int d=0;d<DIM;d++)
    {
      newdimension[d]=dimension[d];
      newcenter[d]=center[d];
    }
    for(;r<=refinements;r++)
    {
      //calculate history
      b=inverse[o][Binner<DIM,LOCS>::Bin(&locs[orders[0].i*DIM],newcenter)];

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
    b=inverse[o][Binner<DIM,LOCS>::Bin(&locs[orders[i].i*DIM],center)];
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
      NextHistory=((history<<DIM)|b);
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
      SerialHR<DIM,BITS>(orders,corders,bin,size[b],newcenter[b],newdimension,orientation[o][b],r+1,NextHistory);
    }
    corders+=size[b];
    orders+=size[b];
  }
}

template<class LOCS> template<class BITS>
void SFC<LOCS>::ComputeLocalHistogram(BITS *histogram,std::vector<History<BITS> > &histories)
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
  int shift=refinements*dim-b;
  mask<<=shift;

  for(unsigned int i=0;i<n;i++)
  {
    int bucket=(histories[i].bits&mask) >> shift;

    histogram[bucket]++;
  }

}
template< class LOCS> template<class BITS>
void SFC<LOCS>::CalculateHistogramsAndCuts(std::vector<BITS> &histograms, std::vector<BITS> &cuts, std::vector<History<BITS> > &histories)
{
  float max_imbalance;
  //calculate b
  b=(int)ceil(log(2.0*P)/log(2.0));
  if(b>refinements*dim)
      b=refinements*dim;

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
    b+=dim;
    /*
    if(max_imbalance>.15 && b<refinements*dim)
    {
  std::cout << "repeat: " << max_imbalance << "P:" << P << " b:" << b << " buckets:" << buckets << std::endl;
    }
    */
  }while(max_imbalance>.15 && b<refinements*dim);
}

template<class LOCS> template<class BITS>
void SFC<LOCS>::BlockedMerge(History<BITS>* start1, History<BITS>* end1, History<BITS>*start2,History<BITS>* end2,History<BITS>* out)
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

    int m1b=std::min(int(end1-start1),(int)merge_block_size);
    int m2b=std::min(int(end2-start2),(int)merge_block_size);

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
template<class LOCS> template<int DIM, class BITS>
void SFC<LOCS>::Parallel()
{
  switch (mergemode)
  {
    case 0:case 1:
            Parallel0<DIM,BITS>();
            break;
    case 2: case 3: case 4:
            Parallel1<DIM,BITS>();
            break;
    case 5:
            Parallel2<DIM,BITS>();
            break;
    case 6:
            Parallel3<DIM,BITS>();
            break;
    default:
            throw InternalError("Invalid Merge Mode",__FILE__,__LINE__);
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
  std::queue<Comm_Msg<BITS> > in_transit;
};
template<class LOCS> template<int DIM, class BITS>
void SFC<LOCS>::Parallel3()
{
  std::vector<Comm_Partner<BITS> > spartners(4), rpartners(4);
  std::vector<MPI_Request> rreqs(rpartners.size()), sreqs(spartners.size());
  std::vector<int> rindices(rreqs.size()),sindices(sreqs.size());
#ifdef _TIMESFC_
  start=timer->currentSeconds();
#endif
  std::vector<History<BITS> > myhistories(n);//,recv_histories(n),merge_histories(n),temp_histories(n);

  //calculate local curves
  SerialH<DIM,BITS>(&myhistories[0]);  //Saves results in sendbuf

#ifdef _TIMESFC_
  finish=timer->currentSeconds();
  timers[0]+=finish-start;
  start=timer->currentSeconds();
#endif
  /*
  std::cout << rank << ": histories:";
  for(unsigned int i=0;i<n;i++)
  {
    std::cout << (int)myhistories[i].bits << " ";
  }
  std::cout << std::endl;
  */

  //calculate b
  b=(int)ceil(log(2.0*P)/log(2.0));
  if(b>refinements*DIM)
      b=refinements*DIM;

  //calcualte buckets
  buckets=1<<b;
  //std::cout << rank << ": bits for histogram:" << b << " buckets:" << buckets << std::endl;

  //create local histogram and cuts
  std::vector <BITS> histogram(buckets+P+1);
  std::vector <BITS> recv_histogram(buckets+P+1);
  std::vector <BITS> sum_histogram(buckets+P+1);
  std::vector <BITS> next_recv_histogram(buckets+P+1);

  histogram[buckets]=0;
  histogram[buckets+1]=buckets;

  //std::cout << rank << ": creating histogram\n";
  ComputeLocalHistogram<BITS>(&histogram[0], myhistories);
  //std::cout << rank << ": done creating histogram\n";

#ifdef _TIMESFC_
  finish=timer->currentSeconds();
  timers[6]+=finish-start;
  start=timer->currentSeconds();
#endif


  //merging phase
  int stages=0;
  for(int p=1;p<P;p<<=1,stages++);

  //create groups
  std::vector<std::vector<Group> > groups(stages+1);

  groups[0].push_back(Group(0,P,-1,-1));

  //std::cout << rank << ": creating merging groups\n";
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

  std::vector<MPI_Request> hsreqs;
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
        //std::cout << rank << ": sending and recieving from: " << next_partner_rank << std::endl;
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
        //std::cout << rank << ": recieving from: " << next_partner_group.start_rank << std::endl;
        //start send
        MPI_Irecv(&next_recv_histogram[0],(buckets+next_partner_group.size+1)*sizeof(BITS),MPI_BYTE,next_partner_group.start_rank,0,Comm,&rreq);
      }

      if(next_group.size<next_partner_group.size && next_local_rank==0)
      {
        MPI_Request request;
        //send to last one in partner group
        //std::cout << rank << ": sending additional to: " << next_partner_group.start_rank+next_partner_group.size-1 << std::endl;
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

  std::vector<History<BITS> > recv_histories(n),merge_histories,temp_histories;

  Group group;
  //std::cout << rank << ": merging phase start\n";
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

    //std::cout << rank << ": next group:  start_rank:" << next_group.start_rank << " size:" << next_group.size << " partner_group:" << next_group.partner_group << " next_local_rank:" << next_local_rank << " next_partner_rank:" << next_partner_rank <<std::endl;


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
      std::cout << rank << ": recieved histogram: ";
      for(unsigned int i=0;i<buckets;i++)
      {
        std::cout << recv_histogram[i] << " ";
      }
      std::cout << std::endl;
      std::cout << rank << ": cuts: ";
      for(unsigned int i=0;i<partner_group.size;i++)
      {
        std::cout << recv_histogram[buckets+i] << " ";
      }
      std::cout << std::endl;
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

      //std::cout << rank << ": mean:" << mean << " target: " << target << " total:" << total << std::endl;
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
        //std::cout << rank << ": sending and recieving from: " << next_partner_rank << std::endl;
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
        //std::cout << rank << ": recieving from: " << next_partner_group.start_rank << std::endl;
        //start send
        MPI_Irecv(&next_recv_histogram[0],(buckets+next_partner_group.size+1)*sizeof(BITS),MPI_BYTE,next_partner_group.start_rank,0,Comm,&rreq);
      }

      if(next_group.size<next_partner_group.size && next_local_rank==0)
      {
        MPI_Request request;
        //send to last one in partner group
        //std::cout << rank << ": sending additional to: " << next_partner_group.start_rank+next_partner_group.size-1 << std::endl;
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
  std::cout << rank << ": error invalid parent group!!\n";
      }
*/
      std::vector<int> sendcounts(parent_group.size,0), recvcounts(parent_group.size,0), senddisp(parent_group.size,0), recvdisp(parent_group.size,0);

      BITS oldstart=histogram[buckets+local_rank],oldend=histogram[buckets+local_rank+1];
      //std::cout << rank << ": oldstart:" << oldstart << " oldend:" << oldend << std::endl;

      //calculate send count
      for(int p=0;p<parent_group.size;p++)
      {
        //i own old histogram from buckets oldstart to oldend
        //any elements between oldstart and oldend that do not belong on me according to the new cuts must be sent
        //std::cout << rank << ": sum_histogram[buckets+p]:" << sum_histogram[buckets+p] << " sum_histogram[buckets+p+1]:" << sum_histogram[buckets+p+1] << std::endl;
        BITS start=std::max(oldstart,sum_histogram[buckets+p]),end=std::min(oldend,sum_histogram[buckets+p+1]);
        for(unsigned int bucket=start;bucket<end;bucket++)
        {
           sendcounts[p]+=histogram[bucket];
        }
      }

      //calculate recv count
      //i will recieve from every processor that owns a bucket assigned to me
      //ownership is determined by that processors old histogram and old cuts

      BITS newstart=sum_histogram[buckets+next_local_rank],newend=sum_histogram[buckets+next_local_rank+1];
      //std::cout << rank << ": newstart: " << newstart << " newend:" << newend << std::endl;

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
        //std::cout << rank << ": lefthistogram[buckets+p]:" << lefthistogram[buckets+p] << " lefthistogram[buckets+p+1]:" << lefthistogram[buckets+p+1] << std::endl;
        BITS start=std::max(newstart,lefthistogram[buckets+p]), end=std::min(newend,lefthistogram[buckets+p+1]);
        for(unsigned int bucket=start;bucket<end;bucket++)
        {
          recvcounts[p]+=lefthistogram[bucket];
        }
      }
      //old histogram and cuts is recv_histogram
      for(int p=0;p<rightsize;p++)
      {
  int start=std::max(newstart,righthistogram[buckets+p]),end=std::min(newend,righthistogram[buckets+p+1]);
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
      //std::cout << rank << " resizing histories to:" << newn << std::endl;

      recv_histories.resize(newn);

#if 0
      std::cout << rank << ": sendcounts: ";
      for(int p=0;p<parent_group.size;p++)
      {
        std::cout << sendcounts[p] << " ";
      }
      std::cout << std::endl;
      std::cout << rank << ": recvcounts: ";
      for(int p=0;p<parent_group.size;p++)
      {
        std::cout << recvcounts[p] << " ";
      }
      std::cout << std::endl;
#endif
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
     //std::cout << rank << ": spartners:" << spartners.size() << " rpartners:" << rpartners.size() << std::endl;
     //begin sends
     for(unsigned int i=0;i<spartners.size();i++)
     {
        for(unsigned int b=0;b<blocks_in_transit && spartners[i].remaining>0;b++)
        {
          //create message
          Comm_Msg<BITS> msg;
          msg.size=std::min(comm_block_size,spartners[i].remaining);
          msg.buffer=spartners[i].buffer;

          //adjust partner
          spartners[i].buffer+=msg.size;
          spartners[i].remaining-=msg.size;

          //start send
          MPI_Isend(msg.buffer,msg.size*sizeof(History<BITS>),MPI_BYTE,spartners[i].rank,1,Comm,&msg.request);

          //add msg to in transit queue
          spartners[i].in_transit.push(msg);

          //std::cout << rank << ": started initial sending msg of size: " << msg.size << " to " << spartners[i].rank << std::endl;
        }
     }
     //begin recvs
     for(unsigned int i=0;i<rpartners.size();i++)
     {
        for(unsigned int b=0;b<blocks_in_transit && rpartners[i].remaining>0;b++)
        {
          //create message
          Comm_Msg<BITS> msg;
          msg.size=std::min(comm_block_size,rpartners[i].remaining);
          msg.buffer=rpartners[i].buffer;

          //adjust partner
          rpartners[i].buffer+=msg.size;
          rpartners[i].remaining-=msg.size;

          //start send
          MPI_Irecv(msg.buffer,msg.size*sizeof(History<BITS>),MPI_BYTE,rpartners[i].rank,1,Comm,&msg.request);

          //add msg to in transit queue
          rpartners[i].in_transit.push(msg);

          //std::cout << rank << ": started inital recieving msg of size: " << msg.size << " from " << rpartners[i].rank << std::endl;
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
      //std::cout << rank << ": sreqs:" << sreqs.size() << " rreqs:" << rreqs.size() << std::endl;
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
            //std::cout << rank << ": completed send to " << partner.rank << std::endl;
            partner.in_transit.pop();

            //start next send
            if(partner.remaining>0)
            {
              //create message
              Comm_Msg<BITS> new_msg;
              new_msg.size=std::min(comm_block_size,partner.remaining);
              new_msg.buffer=partner.buffer;

              //adjust partner
              partner.buffer+=new_msg.size;
              partner.remaining-=new_msg.size;

              //start send
              MPI_Isend(new_msg.buffer,new_msg.size*sizeof(History<BITS>),MPI_BYTE,partner.rank,1,Comm,&new_msg.request);

              //add msg to in transit queue
              partner.in_transit.push(new_msg);
              //std::cout << rank << ": started sending msg of size: " << new_msg.size << " to " << partner.rank << std::endl;
            }

            //reset sreqs
            if(!partner.in_transit.empty())
            {
              sreqs[sindices[i]]=partner.in_transit.front().request;
            }
            else
            {
              //std::cout << rank << ": done sending to " << partner.rank << std::endl;
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
            //std::cout << rank << ": completed recieve from " << partner.rank << std::endl;

            Comm_Msg<BITS> msg=partner.in_transit.front();
            partner.in_transit.pop();

            //start next recv
            if(partner.remaining>0)
            {
              //create message
              Comm_Msg<BITS> new_msg;
              new_msg.size=std::min(comm_block_size,partner.remaining);
              new_msg.buffer=partner.buffer;

              //adjust partner
              partner.buffer+=new_msg.size;
              partner.remaining-=new_msg.size;

              //start recv
              MPI_Irecv(new_msg.buffer,new_msg.size*sizeof(History<BITS>),MPI_BYTE,partner.rank,1,Comm,&new_msg.request);

              //add msg to in transit queue
              partner.in_transit.push(new_msg);

              //std::cout << rank << ": started recieving msg of size: " << new_msg.size << " from " << partner.rank << std::endl;
            }
            //reset rreqs
            if(!partner.in_transit.empty())
            {
              rreqs[rindices[i]]=partner.in_transit.front().request;
            }
            else
            {
              //std::cout << rank << ": done recieving from " << partner.rank << std::endl;
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
  std::cout << rank << ": final list: ";
  for(unsigned int i=0;i<myhistories.size();i++)
  {
     std::cout << (int)myhistories[i].bits << " ";
  }
  std::cout << std::endl;
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
    std::cout << "averge recvs:" << avg_recvs << " max recvs:" << max << std::endl;
  }
#endif
  //std::cout << rank << ": all done!\n";
}
template<class LOCS> template<int DIM, class BITS>
void SFC<LOCS>::Parallel2()
{
  int total_recvs=0;
  int num_recvs=0;
#ifdef _TIMESFC_
  start=timer->currentSeconds();
#endif
  std::vector<History<BITS> > myhistories(n);//,recv_histories(n),merge_histories(n),temp_histories(n);

  //calculate local curves
  SerialH<DIM,BITS>(&myhistories[0]);  //Saves results in sendbuf

#ifdef _TIMESFC_
  finish=timer->currentSeconds();
  timers[0]+=finish-start;
  start=timer->currentSeconds();
#endif
  /*
  std::cout << rank << ": histories:";
  for(unsigned int i=0;i<n;i++)
  {
    std::cout << (int)myhistories[i].bits << " ";
  }
  std::cout << std::endl;
  */

  //calculate b
  b=(int)ceil(log(2.0*P)/log(2.0));
  if(b>refinements*DIM)
      b=refinements*DIM;

  //calcualte buckets
  buckets=1<<b;
  //std::cout << rank << ": bits for histogram:" << b << " buckets:" << buckets << std::endl;

  //create local histogram and cuts
  std::vector <BITS> histogram(buckets+P+1);
  std::vector <BITS> recv_histogram(buckets+P+1);
  std::vector <BITS> sum_histogram(buckets+P+1);
  std::vector <BITS> next_recv_histogram(buckets+P+1);

  histogram[buckets]=0;
  histogram[buckets+1]=buckets;

  //std::cout << rank << ": creating histogram\n";
  ComputeLocalHistogram<BITS>(&histogram[0], myhistories);
  //std::cout << rank << ": done creating histogram\n";

#ifdef _TIMESFC_
  finish=timer->currentSeconds();
  timers[6]+=finish-start;
  start=timer->currentSeconds();
#endif


  //merging phase
  int stages=0;
  for(int p=1;p<P;p<<=1,stages++);

  //create groups
  std::vector<std::vector<Group> > groups(stages+1);

  groups[0].push_back(Group(0,P,-1,-1));

  //std::cout << rank << ": creating merging groups\n";
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

  std::vector<MPI_Request> hsreqs;
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
        //std::cout << rank << ": sending and recieving from: " << next_partner_rank << std::endl;
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
        //std::cout << rank << ": recieving from: " << next_partner_group.start_rank << std::endl;
        //start send
        MPI_Irecv(&next_recv_histogram[0],(buckets+next_partner_group.size+1)*sizeof(BITS),MPI_BYTE,next_partner_group.start_rank,stages,Comm,&rreq);
      }

      if(next_group.size<next_partner_group.size && next_local_rank==0)
      {
        MPI_Request request;
        //send to last one in partner group
        //std::cout << rank << ": sending additional to: " << next_partner_group.start_rank+next_partner_group.size-1 << std::endl;
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
  std::vector<History<BITS> > recv_histories(n),merge_histories,temp_histories;

  Group group;
  //std::cout << rank << ": merging phase start\n";
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

    //std::cout << rank << ": next group:  start_rank:" << next_group.start_rank << " size:" << next_group.size << " partner_group:" << next_group.partner_group << " next_local_rank:" << next_local_rank << " next_partner_rank:" << next_partner_rank <<std::endl;


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
      std::cout << rank << ": recieved histogram: ";
      for(unsigned int i=0;i<buckets;i++)
      {
        std::cout << recv_histogram[i] << " ";
      }
      std::cout << std::endl;
      std::cout << rank << ": cuts: ";
      for(unsigned int i=0;i<partner_group.size;i++)
      {
        std::cout << recv_histogram[buckets+i] << " ";
      }
      std::cout << std::endl;
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

      //std::cout << rank << ": mean:" << mean << " target: " << target << " total:" << total << std::endl;
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
        //std::cout << rank << ": sending and recieving from: " << next_partner_rank << std::endl;
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
        //std::cout << rank << ": recieving from: " << next_partner_group.start_rank << std::endl;
        //start send
        MPI_Irecv(&next_recv_histogram[0],(buckets+next_partner_group.size+1)*sizeof(BITS),MPI_BYTE,next_partner_group.start_rank,stage-1,Comm,&rreq);
      }

      if(next_group.size<next_partner_group.size && next_local_rank==0)
      {
        MPI_Request request;
        //send to last one in partner group
        //std::cout << rank << ": sending additional to: " << next_partner_group.start_rank+next_partner_group.size-1 << std::endl;
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
  std::cout << rank << ": error invalid parent group!!\n";
      }
*/
      std::vector<int> sendcounts(parent_group.size,0), recvcounts(parent_group.size,0), senddisp(parent_group.size,0), recvdisp(parent_group.size,0);

      BITS oldstart=histogram[buckets+local_rank],oldend=histogram[buckets+local_rank+1];
      //std::cout << rank << ": oldstart:" << oldstart << " oldend:" << oldend << std::endl;

      //calculate send count
      for(int p=0;p<parent_group.size;p++)
      {
        //i own old histogram from buckets oldstart to oldend
        //any elements between oldstart and oldend that do not belong on me according to the new cuts must be sent
        //std::cout << rank << ": sum_histogram[buckets+p]:" << sum_histogram[buckets+p] << " sum_histogram[buckets+p+1]:" << sum_histogram[buckets+p+1] << std::endl;
        BITS start=std::max(oldstart,sum_histogram[buckets+p]),end=std::min(oldend,sum_histogram[buckets+p+1]);
        for(unsigned int bucket=start;bucket<end;bucket++)
        {
           sendcounts[p]+=histogram[bucket];
        }
      }

      //calculate recv count
      //i will recieve from every processor that owns a bucket assigned to me
      //ownership is determined by that processors old histogram and old cuts

      BITS newstart=sum_histogram[buckets+next_local_rank],newend=sum_histogram[buckets+next_local_rank+1];
      //std::cout << rank << ": newstart: " << newstart << " newend:" << newend << std::endl;

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
        //std::cout << rank << ": lefthistogram[buckets+p]:" << lefthistogram[buckets+p] << " lefthistogram[buckets+p+1]:" << lefthistogram[buckets+p+1] << std::endl;
        BITS start=std::max(newstart,lefthistogram[buckets+p]), end=std::min(newend,lefthistogram[buckets+p+1]);
        for(unsigned int bucket=start;bucket<end;bucket++)
        {
          recvcounts[p]+=lefthistogram[bucket];
        }
      }
      //old histogram and cuts is recv_histogram
      for(int p=0;p<rightsize;p++)
      {
  int start=std::max(newstart,righthistogram[buckets+p]),end=std::min(newend,righthistogram[buckets+p+1]);
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
      //std::cout << rank << " resizing histories to:" << newn << std::endl;

      recv_histories.resize(newn);

#if 0
      std::cout << rank << ": sendcounts: ";
      for(int p=0;p<parent_group.size;p++)
      {
        std::cout << sendcounts[p] << " ";
      }
      std::cout << std::endl;
      std::cout << rank << ": recvcounts: ";
      for(int p=0;p<parent_group.size;p++)
      {
        std::cout << recvcounts[p] << " ";
      }
      std::cout << std::endl;
#endif
      //calculate displacements
      for(int p=1;p<parent_group.size;p++)
      {
        senddisp[p]+=senddisp[p-1]+sendcounts[p-1];
        recvdisp[p]+=recvdisp[p-1]+recvcounts[p-1];
      }

      //redistribute keys
      std::vector<MPI_Request> rreqs,sreqs;

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
          //std::cout << rank << ": sending to " << parent_group.start_rank+p << std::endl;
          if((int)myhistories.size()<senddisp[p]+sendcounts[p])
          {
            std::cout << rank << ": error sending, send size is bigger than buffer\n";
          }
          MPI_Isend(&myhistories[senddisp[p]],sendcounts[p]*sizeof(History<BITS>),MPI_BYTE,parent_group.start_rank+p,2*stages+stage,Comm,&request);
          sreqs.push_back(request);
        }

        //start recv
        if(recvcounts[p]>0)
        {
          //std::cout << rank << ": recieving from " << parent_group.start_rank+p << std::endl;
          if((int)recv_histories.size()<recvdisp[p]+recvcounts[p])
          {
            std::cout << rank << ": error reciving, recieve size is bigger than buffer\n";
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

      std::vector<MPI_Status> statuses(rreqs.size());
      std::vector<int> indices(rreqs.size());
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
  std::cout << rank << ": final list: ";
  for(unsigned int i=0;i<myhistories.size();i++)
  {
     std::cout << (int)myhistories[i].bits << " ";
  }
  std::cout << std::endl;
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
    std::cout << "averge recvs:" << avg_recvs << " max recvs:" << max << std::endl;
  }
#endif
}

template<class LOCS> template<int DIM, class BITS>
void SFC<LOCS>::Parallel1()
{

#ifdef _TIMESFC_
  start=timer->currentSeconds();
#endif
  std::vector<History<BITS> > myhistories(n), mergefrom(n), mergeto(n);


  //calculate local curves
  SerialH<DIM,BITS>(&myhistories[0]);

#ifdef _TIMESFC_
  finish=timer->currentSeconds();
  timers[0]+=finish-start;
  start=timer->currentSeconds();
#endif

  std::vector<BITS> histograms,cuts;
  CalculateHistogramsAndCuts<BITS>(histograms,cuts,myhistories);

  //build send counts and displacements
  std::vector<int> sendcounts(P,0);
  std::vector<int> recvcounts(P,0);
  std::vector<int> senddisp(P,0);
  std::vector<int> recvdisp(P,0);

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
          std::cout << histograms[p*buckets+i] << " ";
        std::cout << std::endl;
     }
     std::cout << "send:senddisp:recv counts:recev disp:\n";
     for(int p=0;p<P;p++)
     {
       std::cout << sendcounts[p] << ":" << senddisp[p] << ":" <<  recvcounts[p] << ":" << recvdisp[p] << std::endl;
     }
  }
  */
  //std::cout << rank << ": newn:" << newn << std::endl;

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
    //std::cout << rank << ": all to all\n";
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
      std::cout << rank << ":" << "lists to merge: ";
      for(int p=0;p<P;p++)
      {
        std::cout << "list:" << p << ": ";
        for(unsigned int i=recvdisp[p]/sizeof(History<BITS>);i<(recvcounts[p]+recvdisp[p])/sizeof(History<BITS>);i++)
        {
          std::cout <<  (int)newhistories[i].bits << " ";
        }
      }
      std::cout << std::endl;
    }
    */


    std::vector<int> mergesizes(P),mergepointers(P);
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
      std::cout << rank << ": lists to merge: ";
      for(int l=0;l<lists;l++)
      {
        std::cout << "list:" << l << ": ";
        for(int i=mergepointers[l];i<mergepointers[l]+mergesizes[l];i++)
        {
          std::cout <<  (int)mergefrom[i].bits << " ";
        }
      }
      std::cout << std::endl << std::endl;
      */
      int l=0;
      mergeto.resize(0);
      for(int i=0;i<lists;i+=2)
      {
        int mln=mergesizes[i]+mergesizes[i+1];
        if(mln!=0)
        {
          int mlp=mergeto.size();
          typename std::vector<History<BITS> >::iterator l1begin=mergefrom.begin()+mergepointers[i];
          typename std::vector<History<BITS> >::iterator l2begin=mergefrom.begin()+mergepointers[i+1];
          typename std::vector<History<BITS> >::iterator l1end=mergefrom.begin()+mergepointers[i]+mergesizes[i];
          typename std::vector<History<BITS> >::iterator l2end=mergefrom.begin()+mergepointers[i+1]+mergesizes[i+1];
          /*
          std::cout << rank << ": merging lists: ";
          for(std::vector<History<BITS> >::iterator iter=l1begin;iter<l1end;iter++)
          {
            std::cout << (int)iter->bits << " ";
          }
          std::cout << " and ";
          for(std::vector<History<BITS> >::iterator iter=l2begin;iter<l2end;iter++)
          {
            std::cout << (int)iter->bits << " ";
          }
          */
          merge(l1begin,l1end,l2begin,l2end,std::back_inserter(mergeto));
          mergesizes[l]=mln;
          mergepointers[l]=mlp;
          /*
          std::cout << " merged list:";
          for(std::vector<History<BITS> >::iterator iter=mergeto.begin()+mlp;iter<mergeto.begin()+mlp+mln;iter++)
          {
            std::cout << (int)iter->bits << " ";
          }
          std::cout << std::endl;
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
    std::vector<MPI_Request> sreqs;
    std::vector<MPI_Request> rreqs;

    std::vector<History<BITS> > recvbuf(newn);
    MPI_Request empty;
    //start sends
    for(int p=0;p<P;p++)
    {
      if(sendcounts[p]!=0 && p!=rank)
      {
        //std::cout << rank << " sending " << sendcounts[p] << " to rank " << p << std::endl;
        sreqs.push_back(empty);
        MPI_Isend(&myhistories[senddisp[p]],sendcounts[p]*sizeof(History<BITS>),MPI_BYTE,p,0,Comm,&sreqs.back());
      }
    }
    //start recieves
    for(int p=0;p<P;p++)
    {
      if(recvcounts[p]!=0 && p!=rank)
      {
        //std::cout << rank << " recving " << recvcounts[p] << " from rank " << p << std::endl;
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
        //std::cout << "doing waitany\n";
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
        //std::cout << "Recieved list from " << p << std::endl;
        merge(mergefrom.begin(),mergefrom.end(),recvbuf.begin()+recvdisp[p],recvbuf.begin()+recvdisp[p]+recvcounts[p],std::back_inserter(mergeto));
        //std::cout << "done merging\n";
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
       std::vector<std::vector<std::vector<History<BITS> > > > done(stages);

       //std::cout << rank << ": stages: " << stages << std::endl;

       if(recvcounts[rank]!=0)
       {
        //copy my list to buffers
        done[0].push_back(std::vector<History<BITS > >(myhistories.begin()+senddisp[rank],myhistories.begin()+senddisp[rank]+sendcounts[rank]));
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
        done[0].push_back(std::vector<History<BITS> >(recvbuf.begin()+recvdisp[p],recvbuf.begin()+recvdisp[p]+recvcounts[p]));

        //process done requests
        while(done[mstage].size()==2)
        {
          //create mergeto buffer
          done[mstage+1].push_back(std::vector<History<BITS> >(done[mstage][0].size()+done[mstage][1].size()) );
          done[mstage+1].back().resize(0);
          //std::cout << rank << ": merging:  mstage:" << mstage << " list sizes are:" << done[mstage][0].size() << " and " << done[mstage][1].size() << std::endl;
          //merge lists into new buffer
          merge(done[mstage][0].begin(),done[mstage][0].end(),done[mstage][1].begin(),done[mstage][1].end(),back_inserter(done[mstage+1].back()));
          /*
          std::cout << rank << ": done merging, list is";
          for(unsigned int i=0;i<done[mstage+1][0].size();i++)
          {
             std::cout << (int)done[mstage+1][0][i].bits << " ";
          }
          std::cout << std::endl;
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
          done[mstage+1].push_back(std::vector<History<BITS> >(done[mstage][0].begin(),done[mstage][0].end()));
         }
         else if(done[mstage].size()==2)
         {
          //create mergeto buffer
          done[mstage+1].push_back(std::vector<History<BITS> >(done[mstage][0].size()+done[mstage][1].size()) );
          done[mstage+1].back().resize(0);

          //merge lists into new buffer
          merge(done[mstage][0].begin(),done[mstage][0].end(),done[mstage][1].begin(),done[mstage][1].end(),back_inserter(done[mstage+1].back()));
         }
      }



      //std::cout << rank << ": resizing mergefrom to size:" << done.back().back().size() << std::endl;
      mergefrom.resize(done.back().back().size());
      //std::cout << rank << ": copying to mergefrom\n";
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
  std::cout << rank << ": final list: ";
  for(unsigned int i=0;i<mergefrom.size();i++)
  {
     std::cout << (int)mergefrom[i].bits << " ";
  }
   std::cout << std::endl;
 */
#ifdef _TIMESFC_
  finish=timer->currentSeconds();
  timers[3]+=finish-start;
#endif

}
template<class LOCS> template<int DIM, class BITS>
void SFC<LOCS>::Parallel0()
{
#ifdef _TIMESFC_
  start=timer->currentSeconds();
#endif
  std::vector<History<BITS> > histories(n);
  unsigned int i;

  SerialH<DIM,BITS>(&histories[0]);  //Saves results in sendbuf

#ifdef _TIMESFC_
  finish=timer->currentSeconds();
  timers[0]+=finish-start;
#endif
  std::vector<History<BITS> > rbuf, mbuf(n);

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
  std::cout << rank << ": final list: ";
  for(unsigned int i=0;i<histories.size();i++)
  {
     std::cout << (int)histories[i].bits << ":" << histories[i].index.p << ":" << histories[i].index.i << " ";
  }
  std::cout << std::endl;
  */
  /*
  for(unsigned int i=1;i<histories.size();i++)
  {
     if(histories[i].bits-1!=histories[i-1].bits)
        std::cout << rank << ": sfc error\n ";
  }
  */
}

template <class BITS>
struct MergeInfo
{
  unsigned int n;
  BITS min;
  BITS max;
  MergeInfo<BITS>() : n(0), min(0), max(0) {};
};

#define ASCENDING 0
#define DESCENDING 1
#if 1
template<class LOCS> template<class BITS>
int SFC<LOCS>::MergeExchange(int to,std::vector<History<BITS> > &sendbuf, std::vector<History<BITS> >&recievebuf, std::vector<History<BITS> > &mergebuf)
{
  //std::cout << rank <<  ": Merge Exchange started with " << to << std::endl;
  int direction= (int) (rank>to);

  MergeInfo<BITS> myinfo,theirinfo;

  MPI_Request srequest, rrequest;
  MPI_Status status;

  myinfo.n=n;
  if(n!=0)
  {
    myinfo.min=sendbuf[0].bits;
    myinfo.max=sendbuf[n-1].bits;
  }

  MPI_Isend(&myinfo,sizeof(MergeInfo<BITS>),MPI_BYTE,to,0,Comm,&srequest);
  MPI_Irecv(&theirinfo,sizeof(MergeInfo<BITS>),MPI_BYTE,to,0,Comm,&rrequest);
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
      int sendn=n;  //the number of elements to send
      int send_offset=0; //the index of the first element to send
      if(n>theirinfo.n)
      {
        //i have more elements than the other side
          //we don't want to the number of elements to change
          //so copy the first elements to the end of the merge buff

        //move last elements to merge buf
        int diff=n-theirinfo.n;
        for(int i=0;i<diff;i++)
        {
          mergebuf[n-diff+i]=sendbuf[i];
        }

        sendn=theirinfo.n;
        send_offset=diff;
      }

       //send the elements to be merged
       MPI_Isend(&sendbuf[send_offset],sendn*sizeof(History<BITS>),MPI_BYTE,to,0,Comm,&srequest);
       //recv the elements to be merged
       MPI_Irecv(&mergebuf[0],sendn*sizeof(History<BITS>),MPI_BYTE,to,0,Comm,&rrequest);
       MPI_Wait(&rrequest,&status);
       MPI_Wait(&srequest,&status);
       sendbuf.swap(mergebuf);
       return 1;
    }
  }
  else //DESCENDING
  {
    if(theirinfo.max<=myinfo.min) //no exchange needed
        return 0;
    else if(myinfo.max<=theirinfo.min) //full exchange needed
    {
      int sendn=n;  //the number of elements to send
      int send_offset=0; //the index of the first element to send

      if(n>theirinfo.n)
      {
        //i have more elements than the other side
          //we don't want to the number of elements to change
          //so copy the last elements to the begining of the merge buff

        //move last elements to merge buf
        int diff=n-theirinfo.n;

        for(int i=0;i<diff;i++)
        {
          mergebuf[i]=sendbuf[n-diff+i];
        }

        sendn=theirinfo.n;
        send_offset=diff;

      }

      //send the elements to be merged
      MPI_Isend(&sendbuf[0],sendn*sizeof(History<BITS>),MPI_BYTE,to,0,Comm,&srequest);
      //recv the elements to be merged
      MPI_Irecv(&mergebuf[send_offset],sendn*sizeof(History<BITS>),MPI_BYTE,to,0,Comm,&rrequest);
      MPI_Wait(&rrequest,&status);
      MPI_Wait(&srequest,&status);
      sendbuf.swap(mergebuf);
      return 1;
    }
  }

  //std::cout << rank << ": Max-min done\n";

  recievebuf.resize(theirinfo.n);

  History<BITS> *sbuf=&sendbuf[0], *rbuf=&recievebuf[0], *mbuf=&mergebuf[0];
  History<BITS> *msbuf=sbuf, *mrbuf=rbuf;

  unsigned int nsend=n;
  unsigned int nrecv=theirinfo.n;

  //final exchange


  if(direction==ASCENDING)
  {
    //Merge Ascending

    History<BITS> *start1=msbuf,*start2=mrbuf,*end2=mrbuf+theirinfo.n,*out=mbuf, *outend=mbuf+n;
    //position buffers

    //communicate lists
    MPI_Irecv(rbuf,nrecv*sizeof(History<BITS>),MPI_BYTE,to,1,Comm,&rrequest);
    MPI_Isend(sbuf,nsend*sizeof(History<BITS>),MPI_BYTE,to,1,Comm,&srequest);

    //wait for recieve
    MPI_Wait(&rrequest,&status);

    //merge lists

    //while there is more merging needed and I have recieved histories
    for(;out<outend;out++)
    {
      //std::cout << rank << " start1:" << (unsigned int)start1->bits << " start2:" << (unsigned int)start2->bits;
      //if there is data in the recieved buffer and it is less than my buffer
      if(start2<end2 && *start2 < *start1)
        *out=*start2++; //take from recieved buffer
      else
        *out=*start1++; //take from my buffer

      //std::cout << " Ascending took:" << (unsigned int)out->bits << std::endl;
    }

    //wait for send
    MPI_Wait(&srequest,&status);

  }
  else
  {
    //Merge Descending

    History<BITS> *start2=mrbuf,*end1=msbuf+n-1,*end2=mrbuf+theirinfo.n-1,*out=mbuf, *outend=mbuf+n-1;
    //position buffers

    //communicate lists
    MPI_Irecv(rbuf,nrecv*sizeof(History<BITS>),MPI_BYTE,to,1,Comm,&rrequest);
    MPI_Isend(sbuf,nsend*sizeof(History<BITS>),MPI_BYTE,to,1,Comm,&srequest);

    //wait for recieve
    MPI_Wait(&rrequest,&status);

    //merge lists

    //while there is more merging needed and I have recieved histories
    for(;outend>=out;outend--)
    {
      //std::cout << rank << " end1:" << (unsigned int)end1->bits << " end2:" << (unsigned int)end2->bits;
      //if there is data in the recieved buffer and it is less than my buffer
      if(start2<=end2 && *end2 > *end1)
        *outend=*end2--; //take from recieved buffer
      else
        *outend=*end1--; //take from my buffer
      //std::cout << " Descending took:" << (unsigned int)out->bits << std::endl;
    }

    //wait for send
    MPI_Wait(&srequest,&status);
  }
#if 0
  std::cout << rank << " list before: ";
  for(unsigned int i=0;i<n;i++)
    std::cout << (unsigned int)sendbuf[i].bits << " ";
  std::cout << std::endl;
#endif
  sendbuf.swap(mergebuf);
#if 0
  std::cout << rank << " list after: ";
  for(unsigned int i=0;i<n;i++)
    std::cout << (unsigned int)sendbuf[i].bits << " ";
  std::cout << std::endl;
#endif
  //std::cout << rank << ": done ME\n";
  return 1;
}
#endif
#if 0
template<class LOCS> template<class BITS>
int SFC<LOCS>::MergeExchange(int to,std::vector<History<BITS> > &sendbuf, std::vector<History<BITS> >&recievebuf, std::vector<History<BITS> > &mergebuf)
{
  float inv_denom=1.0/sizeof(History<BITS>);
  //std::cout << rank <<  ": Merge Exchange started with " << to << std::endl;
  int direction= (int) (rank>to);
  //BITS emax, emin;
  std::queue<MPI_Request> squeue, rqueue;
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
  //std::cout << rank << " n:" << n << " min:" << (int)myinfo.min << "max:" << (int)myinfo.max << std::endl;

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
  //std::cout << rank << ": Max-min done\n";

  recievebuf.resize(n2);

  History<BITS> *sbuf=&sendbuf[0], *rbuf=&recievebuf[0], *mbuf=&mergebuf[0];
  History<BITS> *msbuf=sbuf, *mrbuf=rbuf;

  unsigned int nsend=n;
  unsigned int nrecv=n2;
  //sample exchange
  unsigned int minn=std::min(n,n2);
  unsigned int sample_size=(int)(minn*sample_percent);

  if(sample_size>=5)
  {
//    std::cout << rank << " creating samples\n";
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
//    std::cout << "exchanging samples\n";
    //exchange samples
    MPI_Isend(mysample,sample_size*sizeof(BITS),MPI_BYTE,to,1,Comm,&srequest);
    MPI_Irecv(theirsample,sample_size*sizeof(BITS),MPI_BYTE,to,1,Comm,&rrequest);

    MPI_Wait(&rrequest,&status);
    MPI_Wait(&srequest,&status);

//    std::cout << "done exchanging samples\n";
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
  //std::cout << rank << ": sample done\n";

  int b;
  unsigned int block_count=0;
  int sremaining=nsend;
  int rremaining=nrecv;
//  std::cout << sremaining << " " << rremaining << std::endl;
  unsigned int sent=0, recvd=0;
  //unsigned int merged=0;
//  std::cout << rank << " Block size: " << comm_block_size << std::endl;


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
        int send=std::min(sremaining,(int)comm_block_size);
        sbuf-=send;
        MPI_Isend(sbuf,send*sizeof(History<BITS>),MPI_BYTE,to,1,Comm,&srequest);
        squeue.push(srequest);
        sent+=send;
        sremaining-=send;
      }

      //recieve
      if(rremaining>0)
      {
        int recv=std::min(rremaining,(int)comm_block_size);
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
        int send=std::min(sremaining,(int)comm_block_size);
        sbuf-=send;
        MPI_Isend(sbuf,send*sizeof(History<BITS>),MPI_BYTE,to,1,Comm,&srequest);
        squeue.push(srequest);
        sent+=send;
        sremaining-=send;
      }

      if(rremaining>0)
      {
        int recv=std::min(rremaining,(int)comm_block_size);
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
        int mb=std::min(int(end1-start1),(int)merge_block_size);
        int rb=std::min(int(end2-start2),(int)merge_block_size);

        BITS mmin=start1->bits, mmax=(start1+mb-1)->bits, bmin=start2->bits,bmax=(start2+rb-1)->bits;

        if(mmax<=bmin) //take everything from mine
        {
          int s=std::min(mb,int(outend-out));
          //std::cout << rank << ": take " << s << " from mine\n";
          memcpy(out,start1,s*sizeof(History<BITS>));
          start1+=s;
          out+=s;
        }
        else if (bmax<mmin) //take everything from theirs
        {
          int s=std::min(rb,int(outend-out));
          //std::cout << rank << ": take " << s << " from theirs\n";
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
        int send=std::min(sremaining,(int)comm_block_size);
        MPI_Isend(sbuf+sent,send*sizeof(History<BITS>),MPI_BYTE,to,1,Comm,&srequest);
        squeue.push(srequest);
        sent+=send;
        sremaining-=send;
      }

      if(rremaining>0)
      {
        int recv=std::min(rremaining,(int)comm_block_size);
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
        int send=std::min(sremaining,(int)comm_block_size);
        MPI_Isend(sbuf+sent,send*sizeof(History<BITS>),MPI_BYTE,to,1,Comm,&srequest);
        squeue.push(srequest);
        sent+=send;
        sremaining-=send;
      }
      if(rremaining>0)
      {
        int recv=std::min(rremaining,(int)comm_block_size);
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
        int mb=std::min(int(start1-end1),(int)merge_block_size);
        int rb=std::min(int(start2-end2),(int)merge_block_size);

        BITS mmin=(start1-mb+1)->bits, mmax=start1->bits, bmin=(start2-rb+1)->bits,bmax=start2->bits;

        if(mmin>bmax) //take everything from mine
        {
          int s=std::min(mb,int(out-outend));
          memcpy(out-s+1,start1-s+1,s*sizeof(History<BITS>));
          start1-=s;
          out-=s;
        }
        else if (bmin>=mmax) //take everything from theirs
        {
          int s=std::min(rb,int(out-outend));
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
//      std::cout << rank << " (DSC) merging more from " << to << std::endl;
    }
    */
  }
  while(!rqueue.empty())
  {
    MPI_Wait(&(rqueue.front()),&status);
    rqueue.pop();
//    std::cout << rank << " recieved left over block\n";
  }
  while(!squeue.empty())
  {
    MPI_Wait(&(squeue.front()),&status);
    squeue.pop();
//    std::cout << rank << " sent left over block\n";
  }

  sendbuf.swap(mergebuf);
  //std::cout << rank << ": done ME\n";
  return 1;
}
#endif
struct HC_MERGE
{
  unsigned int base;
  unsigned int P;
};

template<class LOCS> template<class BITS>
void SFC<LOCS>::PrimaryMerge(std::vector<History<BITS> > &histories, std::vector<History<BITS> >&rbuf, std::vector<History<BITS> > &mbuf)
{
  std::queue<HC_MERGE> q;
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
template<class LOCS> template<class BITS>
void SFC<LOCS>::PrimaryMerge2(std::vector<History<BITS> > &histories, std::vector<History<BITS> >&rbuf, std::vector<History<BITS> > &mbuf, int P, int base)
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
template<class LOCS> template<class BITS>
void SFC<LOCS>::Cleanup(std::vector<History<BITS> > &histories, std::vector<History<BITS> >&rbuf, std::vector<History<BITS> > &mbuf)
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
template<class LOCS> template <class BITS>
void SFC<LOCS>::Batchers(std::vector<History<BITS> > &histories, std::vector<History<BITS> >&rbuf, std::vector<History<BITS> > &mbuf)
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
template<class LOCS> template <class BITS>
void SFC<LOCS>::Linear(std::vector<History<BITS> > &histories, std::vector<History<BITS> >&rbuf, std::vector<History<BITS> > &mbuf)
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
template<class LOCS>
void SFC<LOCS>::SetMergeParameters(unsigned int comm_block_size,unsigned int merge_block_size, unsigned int blocks_in_transit, float sample_percent)
{
  this->comm_block_size=comm_block_size;
  this->blocks_in_transit=blocks_in_transit;
  this->sample_percent=sample_percent;
  this->merge_block_size=merge_block_size;
}

template<class LOCS>
void SFC<LOCS>::SetLocations(std::vector<LOCS> *locsv)
{
  if(locsv!=0)
  {
    this->locsv=locsv;
    this->locs=&(*locsv)[0];
  }
}

template<class LOCS>
void SFC<LOCS>::SetOutputVector(std::vector<DistributedIndex> *orders)
{
  if(orders!=0)
  {
    this->orders=orders;
  }
}
template<class LOCS>
void SFC<LOCS>::SetLocalSize(unsigned int n)
{
  this->n=n;
}
template<class LOCS>
void SFC<LOCS>::SetDimensions(LOCS *dimensions)
{
  for(int d=0;d<dim;d++)
  {
    this->dimensions[d]=dimensions[d];
  }
}
template<class LOCS>
void SFC<LOCS>::SetCenter(LOCS *center)
{
  for(int d=0;d<dim;d++)
  {
    this->center[d]=center[d];
  }
}
template<class LOCS>
void SFC<LOCS>::SetRefinements(int refinements)
{
  this->refinements=refinements;
}
template<class LOCS>
void SFC<LOCS>::SetRefinementsByDelta(LOCS *delta)
{
  if(dimensions[0]==INT_MAX && dimensions[1]==INT_MAX && dimensions[2]==INT_MAX)
  {
    throw InternalError("SFC Dimensions not set",__FILE__,__LINE__);
  }
  refinements=(int)ceil(log(dimensions[0]/delta[0])/log(2.0));
  for(int d=1;d<dim;d++)
  {
    refinements=std::max(refinements,(int)ceil(log(dimensions[d]/delta[d])/log(2.0)));
  }
}
template<class LOCS>
void SFC<LOCS>::SetNumDimensions(int dimensions)
{
  dim=dimensions;
  switch(dim)
  {
    case 1:
      dir=dir1;
      break;
    case 2:
      dir=dir2;
      break;
    case 3:
      dir=dir3;
      break;
  }
  SetCurve(curve);
}
template<class LOCS>
void SFC<LOCS>::SetCurve(Curve curve)
{
  switch(dim)
  {
    case 1:
      order=order1;
      orientation=orient1;
      inverse=inv1;
      break;
    case 2:
      switch(curve)
      {
        case HILBERT:
          order=horder2;
          orientation=horient2;
          inverse=hinv2;
          break;
        case MORTON:
          order=morder2;
          orientation=morient2;
          inverse=minv2;
          break;
        case GREY:
          order=gorder2;
          orientation=gorient2;
          inverse=ginv2;
           break;
      }
       break;
     case 3:
      switch(curve)
      {
        case HILBERT:
          order=horder3;
          orientation=horient3;
          inverse=hinv3;
          break;
        case MORTON:
          order=morder3;
          orientation=morient3;
          inverse=morder3;
          break;
        case GREY:
          order=gorder3;
          orientation=gorient3;
          inverse=ginv3;
          break;
      }
      break;
  }
}
} //End Namespace Uintah

#endif
