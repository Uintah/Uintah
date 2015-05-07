#include <iostream>
#include <cmath>
#include <ctime>
#include <cstdlib>
#include <vector>
#include <climits>
using namespace std;
#include <Core/Math/MiscMath.h>
using namespace SCIRun;
struct Range
{
  double low;
  double high;
  Range(double l, double h) {low=l;high=h;}
};

#define NUMBERS 10000
#define ITERS 10000
int main()
{
  srand(time(0));
  

  double minus_one_third=-1.0/3.0;
  vector<Range> ranges;
  vector<double> guesses;
  ranges.push_back(Range(0,1));
  guesses.push_back(.5);
  ranges.push_back(Range(.5,2));
  guesses.push_back(1);
  ranges.push_back(Range(100,1000));
  guesses.push_back(7);
  ranges.push_back(Range(10000,1000000));
  guesses.push_back(70);

  for(unsigned int r=0;r<ranges.size();r++)
  {
    double pow_time=0; 
    double cube_root_time=0;
    double avg_diff=0;
    double max_diff=0;
    double guess=guesses[r];
    Range range=ranges[r];
    for(int nn=0;nn<NUMBERS;nn++)
    {
      double n=rand()/(double)INT_MAX*(range.high-range.low)+range.low;
      clock_t start,finish;
      
      double pow_ans;
      start=clock();
      for(int i=0;i<ITERS;i++)
        pow_ans=pow(n,minus_one_third);  
      finish=clock();
      
      pow_time+=(finish-start)/double(CLOCKS_PER_SEC);

      double cube_ans;
      start=clock();
      for(int i=0;i<ITERS;i++)
        cube_ans=1.0/cubeRoot(n,guess);  
      finish=clock();
      
      cube_root_time+=(finish-start)/double(CLOCKS_PER_SEC);

      double diff=fabs(pow_ans-cube_ans);
      avg_diff+=diff;
      if(diff>max_diff)
        max_diff=diff;
    }
    double factor=double(NUMBERS)*ITERS;
    avg_diff/=factor;
    pow_time/=factor;
    cube_root_time/=factor;

    cout << "Range: [" << range.low << "," << range.high << "] initial guess: " << guess << " pow time: " << pow_time << " cubeRoot time:" << cube_root_time << " avg diff: " << avg_diff << " max diff:" << max_diff << endl;
  }

  return 0;
}
