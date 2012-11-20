
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Region.h>
#include <vector>

using namespace SCIRun;
using namespace Uintah;
class TiledRegridder
{
  public:
    TiledRegridder(IntVector patch_size,IntVector rr) : rr(rr),mps(patch_size)
    {
    };

    void regrid(const std::vector<Region> &cp, const std::vector<CCVariable<int>*> &flags, std::vector<Region> &patches);


  private:
    IntVector rr;
    IntVector mps;
};


void TiledRegridder::regrid(const std::vector<Region> &cp, const std::vector<CCVariable<int>*> &flags, std::vector<Region> &patches)
{
  patches.resize(0);
  Vector factor=rr.asVector()/mps.asVector(); 
  Vector inv_factor=mps.asVector()/rr.asVector();
  for(unsigned int patch=0;patch<cp.size();patch++)
  {
    //cout << "Coarse Patch: " << cp[patch].getLow() << " " << cp[patch].getHigh() << endl;
    //compute patch extents
    //compute possible tile index's
    
    IntVector tileLow(cp[patch].getLow()*rr/mps);
    IntVector tileHigh(cp[patch].getHigh()*rr/mps);

    //cout << "Tiles: " << tileLow << " " << tileHigh << endl; 
    //cout << "window: " << (*flags[patch]).getWindow()->getLowIndex() << " " << (*flags[patch]).getWindow()->getHighIndex()  << endl;

    for (CellIterator ti(tileLow,tileHigh); !ti.done(); ti++)
    {
      IntVector i=*ti;
      IntVector searchLow(
            static_cast<int>(i[0]*inv_factor[0]),
            static_cast<int>(i[1]*inv_factor[1]),
            static_cast<int>(i[2]*inv_factor[2])
          );

      IntVector searchHigh(
            static_cast<int>(searchLow[0]+inv_factor[0]),
            static_cast<int>(searchLow[1]+inv_factor[1]),
            static_cast<int>(searchLow[2]+inv_factor[2])
          );
      IntVector plow=searchLow*rr;
      
      IntVector phigh=searchHigh*rr;
      //cout << "  Coarse Search: " << searchLow << " " << searchHigh << endl;
      for(CellIterator c_it(searchLow,searchHigh);!c_it.done();c_it++)
      {
        if( (*flags[patch])[*c_it]==1)
        {
          //cout << "Adding Patch: " << plow << " " << phigh << endl;
          patches.push_back(Region(plow,phigh));
          break;
        }
        //cout << "    no flag f_it: " << *f_it << endl;
      }
    }
  }
}
