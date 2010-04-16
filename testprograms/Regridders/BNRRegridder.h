#ifndef BNRREGRIDDERTEST
  #define BNRREGRIDDERTEST
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Region.h>
#include <vector>
using namespace std;
using namespace SCIRun;
using namespace Uintah;

struct Split
{
  int d;
  int i;
  Split(int d,int i) : d(d),i(i) {};
};
class BNRRegridder
{
  public:
    BNRRegridder(double tol,IntVector rr) : rr(rr),tol(tol)
    {
      rr=IntVector(4,4,4);
    };
    virtual ~BNRRegridder() {};

    virtual void regrid(const vector<IntVector> &flags, vector<Region> &patches);

  protected:
    IntVector rr;
    double tol;
    virtual void brsplit(list<IntVector> &flags, vector<Region> &patches, int r=0);
    virtual Region computeBounds(list<IntVector> &flags);
    virtual void computeHistogram(list<IntVector> &flags, const Region &bounds, vector<vector<unsigned int> > &hist);
    Split findSplit(vector<vector<unsigned int> > &hist);
    void split(list<IntVector> &flags, const Split &split, list<IntVector> &left, list<IntVector> &right);
    virtual int computeNumFlags(list<IntVector> &flags);
  
};


void BNRRegridder::regrid(const vector<IntVector> &flags, vector<Region> &patches)
{
  patches.resize(0);
  list<IntVector> flags_tmp(flags.begin(),flags.end());

  brsplit(flags_tmp, patches); 
}
    
void BNRRegridder::brsplit(list<IntVector> &flags, vector<Region> &patches,int r)
{
  //bound flags //make as function
  Region bounds=computeBounds(flags);
  unsigned int num_flags=computeNumFlags(flags);

  //cout << getpid() << " Bounds: " << bounds << " flags: " << flags.size() << endl;
  //cout << getpid() << " thresh: " << num_flags/(double)bounds.getVolume() << endl;
  //check tolerance
  if(num_flags/(double)bounds.getVolume()>=tol)
  {
    //cout << getpid() << " adding patch: " << bounds << endl;
    patches.push_back(Region(bounds.getLow()*rr,bounds.getHigh()*rr));
    return;
  }
  
  vector<vector<unsigned int> > hist;
  //generate histogram  //make as function
  computeHistogram(flags,bounds,hist);
  
  //Find Split
  Split s=findSplit(hist);
  s.i+=bounds.getLow()[s.d];

  //cout << getpid() << " Split: d:" << s.d << " i:" << s.i << endl;  

  list<IntVector> left,right;
  split(flags,s,left,right);

  hist.clear();

  //Recurse
  brsplit(left,patches,r+1);
  brsplit(right,patches,r+1);
}
    
Region BNRRegridder::computeBounds(list<IntVector> &flags)
{
  Region bounds(*flags.begin(),*flags.begin());
  
  for(list<IntVector>::iterator iter=flags.begin();iter!=flags.end();iter++)
  {
    bounds.extend(*iter);
    //cout << "Extending bounds to: " << *iter << " new bounds: " << bounds << endl;
  }
  return bounds;
}
int BNRRegridder::computeNumFlags(list<IntVector> &flags)
{
  return flags.size();
}
    
int sign(int num)
{
  if(num<0)
    return -1;
  else if(num>0)
    return 1;
  else 
    return 0;
}
void BNRRegridder::computeHistogram(list<IntVector> &flags, const Region &bounds, vector<vector<unsigned int> > &hist)
{
  //allocate histogram
  hist.resize(3);
  IntVector size=bounds.getHigh()-bounds.getLow();
  hist[0].assign(size[0],0);
  hist[1].assign(size[1],0);
  hist[2].assign(size[2],0);

  //create histogram
  for(list<IntVector>::iterator iter=flags.begin();iter!=flags.end();iter++)
  {
    IntVector p=(*iter)-bounds.getLow();
    //cout << "adding to hist at: " << p << endl;
    hist[0][p.x()]++;
    hist[1][p.y()]++;
    hist[2][p.z()]++;
  }
#if 0
  for(size_t d=0;d<3;d++)
  {
    for(size_t i=0;i<hist[d].size();i++)
    {
      cout << "Hist[" << d << "][" << i << "]: " << hist[d][i] << endl; 
    }
  }
#endif
}
Split BNRRegridder::findSplit(vector<vector<unsigned int> > &hist)
{
  int size[3]={hist[0].size(),hist[1].size(),hist[2].size()};
  int dims[3]={0,1,2};
#if 1
  //simple bubble sort on the length of the dimension
  for(int i=0;i<3;i++)
    for(int j=0;j<i;j++)
      if(size[dims[i]]>size[dims[j]])
        swap(dims[i],dims[j]);
#endif

  //cout << "dimension order: " << dims[0] << " " << dims[1] << " " << dims[2] << endl;

  //search for a zero
  for(size_t dim=0;dim<hist.size();dim++)
  {
    int d=dims[dim];
    for(size_t i=0;i<hist[d].size();i++)
    {
      if(hist[d][i]==0)
      {
        //cout << "Zero split found at: " << d << " : " << i << endl;
        return Split(d,i);
      }
    }
  }

  int max_change=INT_MIN;
  Split split(-1,-1);

  //search for largest sign change in the second derivative
  for(size_t dim=0;dim<hist.size()-1;dim++)
  {
    int d=dims[dim];
    int last_d2=hist[d][1]-hist[d][0];
    //int last_d2=hist[d][0]+hist[d][2]-2*hist[d][1];

    for(size_t i=1;i<hist[d].size()-1;i++)
    {
      int d2=hist[d][i-1]+hist[d][i+1]-2*hist[d][i];
      int change=abs(last_d2-d2);
      //cout << "d: " << d << " i: " << i << " last_d2: " << last_d2 << " d2: " << d2 << endl;
      if(sign(last_d2)!=sign(d2))
      { 
        if(change>max_change )
        {
          max_change=change;
          split.d=d;
          split.i=i;
        }
#if 0
        else if(change==max_change && d==split.d)
        {
          double mid=hist.size()/2.0;
          if(fabs(mid-i)<fabs(mid-split.i))
          {
            split.d=d;
            split.i=i;
          }
        }
#endif
      }

      last_d2=d2;
    }
  }

  //cout << "max_change: " << max_change << endl;
  //could not find a good split, just split in half
  if(max_change<=0)
  {
    split.d=dims[0];
    split.i=hist[split.d].size()/2;
  }

  return split;
}

void BNRRegridder::split(list<IntVector> &flags, const Split &split, list<IntVector> &left, list<IntVector> &right)
{
  left.resize(0);
  right.resize(0);
  
  for(list<IntVector>::iterator iter=flags.begin();iter!=flags.end();)
  {
    list<IntVector>::iterator siter=iter;
    iter++;

    if((*siter)[split.d]<split.i)
    {
      //cout << " moving " << *iter << " to left flags\n";
      left.splice(left.end(),flags,siter);
    }
    else
    {
      //cout << " moving " << *iter << " to right flags\n";
      right.splice(right.end(),flags,siter);
    }
  }

#if 0
  cout << "Left flags: " << endl;
  for(list<IntVector>::iterator iter=left.begin();iter!=left.end();iter++)
  {
    cout << "  " << *iter << endl;
  }
  
  cout << "Right flags: " << endl;
  for(list<IntVector>::iterator iter=right.begin();iter!=right.end();iter++)
  {
    cout << "  " << *iter << endl;
  }
#endif 
}

#endif
