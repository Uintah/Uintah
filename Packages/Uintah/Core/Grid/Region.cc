
#include <Packages/Uintah/Core/Grid/Region.h>
#include <sgi_stl_warnings_off.h>
#include <iostream>
#include <sgi_stl_warnings_on.h>

#include <deque>

using namespace Uintah;
using namespace std;

//This code was pulled from Box.cc

bool
Region::overlaps(const Region& otherregion) const
{
  if(d_lowIndex.x() > otherregion.d_highIndex.x() || d_highIndex.x() < otherregion.d_lowIndex.x()) {
    return false;
  }
  if(d_lowIndex.y() > otherregion.d_highIndex.y() || d_highIndex.y() < otherregion.d_lowIndex.y()) {
    return false;
  }
  if(d_lowIndex.z() > otherregion.d_highIndex.z() || d_highIndex.z() < otherregion.d_lowIndex.z()) {
    return false;
  }
  return true;
}

//static
void Region::difference(list<Region> l1, list<Region> l2, list<Region> &diff)
{
  //bound the regions
  Region bounds(IntVector(INT_MAX,INT_MAX,INT_MAX),IntVector(INT_MIN,INT_MIN,INT_MIN));
  
  for(list<Region>::iterator patch=l1.begin();patch!=l1.end();patch++)
  {
    bounds.d_lowIndex=Min(bounds.d_lowIndex,patch->d_lowIndex);
    bounds.d_highIndex=Max(bounds.d_highIndex,patch->d_highIndex);
  }
  
  for(list<Region>::iterator patch=l2.begin();patch!=l2.end();patch++)
  {
    bounds.d_lowIndex=Min(bounds.d_lowIndex,patch->d_lowIndex);
    bounds.d_highIndex=Max(bounds.d_highIndex,patch->d_highIndex);
  }

  difference(l1,l1.begin(),l1.end(),l2,l2.begin(),l2.end(),bounds,diff);
}

//static
void Region::difference(list<Region> &l1, const list<Region>::iterator &l1_begin, const list<Region>::iterator &l1_end,
                        list<Region> &l2, const list<Region>::iterator &l2_begin, const list<Region>::iterator &l2_end,
                        const Region &bounds, list<Region> &diff)
{
  //base cases
  if (l1_begin==l1_end) //l1 is empty, difference is empty
  {
    return;
  }
  if (l2_begin==l2_end) //l2 is empty, difference is all of l1
  {
    //add all patches in l1 to difference
    for(list<Region>::iterator r=l1_begin;r!=l1_end;r++)
    {
      diff.push_back(*r);
    }
    return;
  }

  list<Region>::iterator l1_next=l1_begin, l2_next=l2_begin;
  l1_next++;
  l2_next++;
  if (l1_next==l1_end && l2_next==l2_end)  //1 patch left in each list
  {
    //add the patches l1-l2
    Region r1=*l1_begin;
    Region r2=*l2_begin;
    if(r1.overlaps(r2))
    {
      // divide the difference space into up to 6 regions, 2 in each dimension.
      // each pass, reduce the amount of space to take up.
      Region intersection = r1.intersect(r2);
      Region leftoverSpace = r1;
      for (int dim = 0; dim < 3; dim++) 
      {
        if (r1.d_lowIndex(dim) < intersection.d_lowIndex(dim)) 
        {
          Region tmp = leftoverSpace;
          tmp.d_lowIndex(dim) = r1.d_lowIndex(dim);
          tmp.d_highIndex(dim) = intersection.d_lowIndex(dim);
          diff.push_back(tmp);
        }
        if (r1.d_highIndex(dim) > intersection.d_highIndex(dim)) 
        {
          Region tmp = leftoverSpace;
          tmp.d_lowIndex(dim) = intersection.d_highIndex(dim);
          tmp.d_highIndex(dim) = r1.d_highIndex(dim);
          diff.push_back(tmp);
        }
        leftoverSpace.d_lowIndex(dim) = intersection.d_lowIndex(dim);
        leftoverSpace.d_highIndex(dim) = intersection.d_highIndex(dim);
      }
    }
    return;
  }
  //else split patches along longest dimension

  //find max dimension
  IntVector size=bounds.d_highIndex-bounds.d_lowIndex;
  int split_d=0;
  for(int d=1;d<3;d++)
  {
    if(size[d]>size[split_d])
      split_d=d;
  }
  int split=(bounds.d_highIndex[split_d]+bounds.d_lowIndex[split_d])/2;

  //reorder lists along split
  list<Region>::iterator l1_front=l1_begin,l1_back=l1_end;
  //reorder list l1
  while(l1_front!=l1_back)
  {
    //if left of split
    if(l1_front->d_highIndex[split_d]<=split)
    {
      //advance front iterator
      l1_front++;
    }
    //else if right of split
    else if (l1_front->d_lowIndex[split_d]>=split)
    {
      //move back iterator backword 1 spot
      l1_back--;
      //swap front and back
      swap(*l1_front,*l1_back);
    }
    //else region overlaps split
    else 
    {
      //split into left and right
      Region right=*l1_front;
      l1_front->d_highIndex[split_d]=split;
      right.d_lowIndex[split_d]=split;
      
      //insert right at the end
      l1.insert(l1_end,right);
      
      //advance front iterator
      l1_front++;
    }
  }   
  
  list<Region>::iterator l2_front=l2_begin,l2_back=l2_end;
  //reorder list l2
  while(l2_front!=l2_back)
  {
    //if left of split
    if(l2_front->d_highIndex[split_d]<=split)
    {
      //advance front iterator
      l2_front++;
    }
    //else if right of split
    else if (l2_front->d_lowIndex[split_d]>=split)
    {
      //move back iterator backword 1 spot
      l2_back--;
      //swap front and back
      swap(*l2_front,*l2_back);
    }
    //else region overlaps split
    else 
    {
      //split into left and right
      Region right=*l2_front;
      l2_front->d_highIndex[split_d]=split;
      right.d_lowIndex[split_d]=split;
      
      //insert right at the end
      l2.insert(l2_end,right);
      
      //advance front iterator
      l2_front++;
    }
  }   

  //calculate new bounds
  Region lbounds=bounds,rbounds=bounds;
  lbounds.d_highIndex[split_d]=split;
  rbounds.d_lowIndex[split_d]=split;

  //recurse left side
  difference(l1,l1_begin,l1_front,l2,l2_begin,l2_front,lbounds,diff);
  //recurse right side
  difference(l1,l1_back,l1_end,l2,l2_back,l2_end,rbounds,diff);
}

//static 
deque<Region> Region::difference(const Region& b1, const Region& b2)
{
  deque<Region> set1, set2;
  set1.push_back(b1);
  set2.push_back(b2);
  return difference(set1, set2);
}

//static 
deque<Region> Region::difference(deque<Region>& region1, deque<Region>& region2)
{
  // use 2 deques, as inserting into a deque invalidates the iterators
  deque<Region> searchSet(region1.begin(), region1.end());
  deque<Region> remainingRegiones;
  // loop over set2, as remainingRegiones will more than likely change
  for (unsigned i = 0; i < region2.size(); i++) {
    for (deque<Region>::iterator iter = searchSet.begin(); iter != searchSet.end(); iter++) {
      Region b1 = *iter;
      Region b2 = region2[i];
      if (b1.overlaps(b2)) {
        // divide the difference space into up to 6 regions, 2 in each dimension.
        // each pass, reduce the amount of space to take up.
        // Add new regions to the front so we don't loop over them again for this region.
        Region intersection = b1.intersect(b2);
        Region leftoverSpace = b1;
        for (int dim = 0; dim < 3; dim++) {
          if (b1.d_lowIndex(dim) < intersection.d_lowIndex(dim)) {
            Region tmp = leftoverSpace;
            tmp.d_lowIndex(dim) = b1.d_lowIndex(dim);
            tmp.d_highIndex(dim) = intersection.d_lowIndex(dim);
            remainingRegiones.push_back(tmp);
          }
          if (b1.d_highIndex(dim) > intersection.d_highIndex(dim)) {
            Region tmp = leftoverSpace;
            tmp.d_lowIndex(dim) = intersection.d_highIndex(dim);
            tmp.d_highIndex(dim) = b1.d_highIndex(dim);
            remainingRegiones.push_back(tmp);              
          }
          leftoverSpace.d_lowIndex(dim) = intersection.d_lowIndex(dim);
          leftoverSpace.d_highIndex(dim) = intersection.d_highIndex(dim);
        }
      } 
      else {
        remainingRegiones.push_back(b1);
      }
    }
    if (i+1 < region2.size()) {
      searchSet = remainingRegiones;
      remainingRegiones.clear();
    }
  }
  return remainingRegiones;
}

