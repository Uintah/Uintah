

#include <Packages/Uintah/Core/Grid/Variables/GridIterator.h>
#include <Packages/Uintah/Core/Grid/PatchBVH/PatchBVH.h>
#include <Packages/Uintah/Core/Grid/PatchRangeTree.h>
#include <Packages/Uintah/Core/Grid/Grid.h>
#include <Packages/Uintah/Core/Grid/LevelP.h>

#include <iostream>
using namespace std;
using namespace Uintah;

bool compareEqual(const Patch* p1, const Patch* p2)
{
  if(p1->getExtraNodeLowIndex__New()!=p2->getExtraNodeLowIndex__New())
    return false;
  if(p1->getExtraNodeHighIndex__New()!=p2->getExtraNodeHighIndex__New())
    return false;
  return true;
}

bool compareQueries( Level::selectType &q1, Level::selectType &q2)
{
  if(q1.size()!=q2.size())
    return false;

  for(Level::selectType::iterator iter1=q1.begin(); iter1!=q1.end(); iter1++)
  {
    bool found=false;
    for(Level::selectType::iterator iter2=q2.begin(); iter2!=q2.end(); iter2++)
    {
      if(compareEqual(*iter1,*iter2)==true);
          found=true;
    }
    if(!found)
      return false;
  }
  return true;
}
int main()
{
  IntVector PatchSize(10,10,10);
  //create a grid iterator to be a list of patches
  GridIterator iter(IntVector(0,0,0),IntVector(10,10,10));
 
  Point anchor(0,0,0);
  Vector dcell(1,1,1);
  Grid grid;

  grid.addLevel(anchor,dcell);
  vector<const Patch*> patches;

  int i=0; 
  for(;!iter.done();iter++,i++)
  {
    LevelP level=grid.getLevel(0);
    IntVector low=*iter*PatchSize;
    IntVector high=(*iter+IntVector(1,1,1))*PatchSize;

    level->addPatch(low,high,low,high);

    patches.push_back(level->getPatch(i));
  }
  
  PatchBVH bvh(patches);
  PatchRangeTree prt(patches);

  //query various regions and check results
  Level::selectType q_new, q_old;
  IntVector low,high;
  
  low=IntVector(0,0,0);
  high=IntVector(10,10,10);
  bvh.query(low,high,q_new);
  prt.query(low,high,q_old);
  if(!compareQueries(q_new,q_old))
  {
    cout << "Error queries do not match(1)\n";
    cout << "old:\n";
    for(Level::selectType::iterator iter=q_old.begin(); iter!=q_old.end(); iter++)
    {
      cout << **iter << endl;
    }
    cout << "new:\n";
    for(Level::selectType::iterator iter=q_new.begin(); iter!=q_new.end(); iter++)
    {
      cout << **iter << endl;
    }
  }
  cout << "Query returned size:" << q_new.size() << endl;
  q_new.resize(0);
  q_old.resize(0);
  
  low=IntVector(0,0,0);
  high=IntVector(12,12,1);
  bvh.query(low,high,q_new);
  prt.query(low,high,q_old);
  if(!compareQueries(q_new,q_old))
  {
    cout << "Error queries do not match(2)\n";
    cout << "old:\n";
    for(Level::selectType::iterator iter=q_old.begin(); iter!=q_old.end(); iter++)
    {
      cout << **iter << endl;
    }
    cout << "new:\n";
    for(Level::selectType::iterator iter=q_new.begin(); iter!=q_new.end(); iter++)
    {
      cout << **iter << endl;
    }
  }
  cout << "Query returned size:" << q_new.size() << endl;
  q_new.resize(0);
  q_old.resize(0);
      
  low=IntVector(34,22,28);
  high=IntVector(73,12,36);
  bvh.query(low,high,q_new);
  prt.query(low,high,q_old);
  if(!compareQueries(q_new,q_old))
  {
    cout << "Error queries do not match(3)\n";
    cout << "old:\n";
    for(Level::selectType::iterator iter=q_old.begin(); iter!=q_old.end(); iter++)
    {
      cout << **iter << endl;
    }
    cout << "new:\n";
    for(Level::selectType::iterator iter=q_new.begin(); iter!=q_new.end(); iter++)
    {
      cout << **iter << endl;
    }
  }
  cout << "Query returned size:" << q_new.size() << endl;
  q_new.resize(0);
  q_old.resize(0);
  
  low=IntVector(8,8,8);
  high=IntVector(8,8,8);
  bvh.query(low,high,q_new);
  prt.query(low,high,q_old);
  if(!compareQueries(q_new,q_old))
  {
    cout << "Error queries do not match(4)\n";
    cout << "old:\n";
    for(Level::selectType::iterator iter=q_old.begin(); iter!=q_old.end(); iter++)
    {
      cout << **iter << endl;
    }
    cout << "new:\n";
    for(Level::selectType::iterator iter=q_new.begin(); iter!=q_new.end(); iter++)
    {
      cout << **iter << endl;
    }
  }
  cout << "Query returned size:" << q_new.size() << endl;
  q_new.resize(0);
  q_old.resize(0);
  
  low=IntVector(10,10,10);
  high=IntVector(9,9,9);
  bvh.query(low,high,q_new);
  prt.query(low,high,q_old);
  if(!compareQueries(q_new,q_old))
  {
    cout << "Error queries do not match(5)\n";
    cout << "old:\n";
    for(Level::selectType::iterator iter=q_old.begin(); iter!=q_old.end(); iter++)
    {
      cout << **iter << endl;
    }
    cout << "new:\n";
    for(Level::selectType::iterator iter=q_new.begin(); iter!=q_new.end(); iter++)
    {
      cout << **iter << endl;
    }
  }
  cout << "Query returned size:" << q_new.size() << endl;
  q_new.resize(0);
  q_old.resize(0);
  
  cout << "All tests successfully passed\n";
  return 0;
}
