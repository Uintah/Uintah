
#include <testprograms/Regridders/BNRRegridder.h>

class LBNRRegridder : BNRRegridder
{
  public:
    LBNRRegridder(double tol, IntVector rr) : BNRRegridder(tol,rr){};

    void regrid(const std::vector<std::list<IntVector> >&lflags, std::vector<Region> &patches);

  private:
  
};


void LBNRRegridder::regrid(const std::vector<std::list<IntVector> > &lflags, std::vector<Region> &patches)
{
  patches.resize(0);

  for(size_t p=0;p<lflags.size();p++)
  {
    if(lflags[p].size()>0)
    {
      std::list<IntVector> flags_tmp(lflags[p].begin(),lflags[p].end());
      brsplit(flags_tmp, patches); 
    }
  }
}
    
