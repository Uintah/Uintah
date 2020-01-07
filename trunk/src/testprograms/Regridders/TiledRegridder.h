
#include <Core/Grid/Variables/CCVariable.h>
#include <Core/Grid/Region.h>
#include <vector>

using Uintah::IntVector;

class TiledRegridder
{
  public:
 TiledRegridder(Uintah::IntVector patch_size,Uintah::IntVector rr) : rr(rr),mps(patch_size)
    {
    };

  void regrid(const std::vector<Uintah::Region> &cp, const std::vector<Uintah::CCVariable<int>*> &flags, std::vector<Uintah::Region> &patches);


  private:
  IntVector rr;
  IntVector mps;
};


void TiledRegridder::regrid(const std::vector<Uintah::Region> &cp, const std::vector<Uintah::CCVariable<int>*> &flags, std::vector<Uintah::Region> &patches)
{
  patches.resize(0);
  Uintah::Vector inv_factor=mps.asVector()/rr.asVector();
  for(unsigned int patch=0;patch<cp.size();patch++)
  {
    //cout << "Coarse Patch: " << cp[patch].getLow() << " " << cp[patch].getHigh() << endl;
    //compute patch extents
    //compute possible tile index's
    
    Uintah::IntVector tileLow(cp[patch].getLow()*rr/mps);
    Uintah::IntVector tileHigh(cp[patch].getHigh()*rr/mps);

    //cout << "Tiles: " << tileLow << " " << tileHigh << endl; 
    //cout << "window: " << (*flags[patch]).getWindow()->getLowIndex() << " " << (*flags[patch]).getWindow()->getHighIndex()  << endl;

    for (Uintah::CellIterator ti(tileLow,tileHigh); !ti.done(); ti++)
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
      for(Uintah::CellIterator c_it(searchLow,searchHigh);!c_it.done();c_it++)
      {
        if( (*flags[patch])[*c_it]==1)
        {
          //cout << "Adding Patch: " << plow << " " << phigh << endl;
          patches.push_back(Uintah::Region(plow,phigh));
          break;
        }
        //cout << "    no flag f_it: " << *f_it << endl;
      }
    }
  }
}
