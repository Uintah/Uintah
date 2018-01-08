/*
 * The MIT License
 *
 * Copyright (c) 1997-2018 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */



#include <Core/Grid/Variables/GridIterator.h>
#include <Core/Grid/PatchBVH/PatchBVH.h>
#include <Core/Grid/PatchRangeTree.h>
#include <Core/Grid/Grid.h>
#include <Core/Grid/LevelP.h>

#include <iostream>


bool compareEqual(const Uintah::Patch* p1, const Uintah::Patch* p2)
{
  if(p1->getExtraNodeLowIndex()!=p2->getExtraNodeLowIndex())
    return false;
  if(p1->getExtraNodeHighIndex()!=p2->getExtraNodeHighIndex())
    return false;
  return true;
}

bool compareQueries( Uintah::Level::selectType &q1, Uintah::Level::selectType &q2)
{
  if(q1.size()!=q2.size())
    return false;

  for(Uintah::Level::selectType::iterator iter1=q1.begin(); iter1!=q1.end(); iter1++)
  {
    bool found=false;
    for(Uintah::Level::selectType::iterator iter2=q2.begin(); iter2!=q2.end(); iter2++)
    {
      if(compareEqual(*iter1,*iter2)==true)
          found=true;
    }
    if(!found)
      return false;
  }
  return true;
}
int main()
{
  Uintah::IntVector PatchSize(10,10,10);
  //create a grid iterator to be a list of patches
  Uintah::GridIterator iter(Uintah::IntVector(0,0,0),Uintah::IntVector(10,10,10));
 
  Uintah::Point anchor(0,0,0);
  Uintah::Vector dcell(1,1,1);
  Uintah::Grid grid;

  grid.addLevel(anchor,dcell);
  std::vector<const Uintah::Patch*> patches;

  int i=0; 
  for(;!iter.done();iter++,i++)
  {
    Uintah::LevelP level=grid.getLevel(0);
    Uintah::IntVector low=*iter*PatchSize;
    Uintah::IntVector high=(*iter+Uintah::IntVector(1,1,1))*PatchSize;

    level->addPatch(low,high,low,high,&grid);

    patches.push_back(level->getPatch(i));
  }
  
  Uintah::PatchBVH bvh(patches);
  Uintah::PatchRangeTree prt(patches);

  //query various regions and check results
  Uintah::Level::selectType q_new, q_old;
  Uintah::IntVector low,high;
  
  low=Uintah::IntVector(0,0,0);
  high=Uintah::IntVector(10,10,10);
  bvh.query(low,high,q_new);
  prt.query(low,high,q_old);
  if(!compareQueries(q_new,q_old))
  {
    std::cout << "Error queries do not match(1)\n";
    std::cout << "old:\n";
    for(Uintah::Level::selectType::iterator iter=q_old.begin(); iter!=q_old.end(); iter++)
    {
      std::cout << **iter << std::endl;
    }
    std::cout << "new:\n";
    for(Uintah::Level::selectType::iterator iter=q_new.begin(); iter!=q_new.end(); iter++)
    {
      std::cout << **iter << std::endl;
    }
  }
  std::cout << "Query returned size:" << q_new.size() << std::endl;
  q_new.resize(0);
  q_old.resize(0);
  
  low=Uintah::IntVector(0,0,0);
  high=Uintah::IntVector(12,12,1);
  bvh.query(low,high,q_new);
  prt.query(low,high,q_old);
  if(!compareQueries(q_new,q_old))
  {
    std::cout << "Error queries do not match(2)\n";
    std::cout << "old:\n";
    for(Uintah::Level::selectType::iterator iter=q_old.begin(); iter!=q_old.end(); iter++)
    {
      std::cout << **iter << std::endl;
    }
    std::cout << "new:\n";
    for(Uintah::Level::selectType::iterator iter=q_new.begin(); iter!=q_new.end(); iter++)
    {
      std::cout << **iter << std::endl;
    }
  }
  std::cout << "Query returned size:" << q_new.size() << std::endl;
  q_new.resize(0);
  q_old.resize(0);
      
  low=Uintah::IntVector(34,22,28);
  high=Uintah::IntVector(73,12,36);
  bvh.query(low,high,q_new);
  prt.query(low,high,q_old);
  if(!compareQueries(q_new,q_old))
  {
    std::cout << "Error queries do not match(3)\n";
    std::cout << "old:\n";
    for(Uintah::Level::selectType::iterator iter=q_old.begin(); iter!=q_old.end(); iter++)
    {
      std::cout << **iter << std::endl;
    }
    std::cout << "new:\n";
    for(Uintah::Level::selectType::iterator iter=q_new.begin(); iter!=q_new.end(); iter++)
    {
      std::cout << **iter << std::endl;
    }
  }
  std::cout << "Query returned size:" << q_new.size() << std::endl;
  q_new.resize(0);
  q_old.resize(0);
  
  low=Uintah::IntVector(8,8,8);
  high=Uintah::IntVector(8,8,8);
  bvh.query(low,high,q_new);
  prt.query(low,high,q_old);
  if(!compareQueries(q_new,q_old))
  {
    std::cout << "Error queries do not match(4)\n";
    std::cout << "old:\n";
    for(Uintah::Level::selectType::iterator iter=q_old.begin(); iter!=q_old.end(); iter++)
    {
      std::cout << **iter << std::endl;
    }
    std::cout << "new:\n";
    for(Uintah::Level::selectType::iterator iter=q_new.begin(); iter!=q_new.end(); iter++)
    {
      std::cout << **iter << std::endl;
    }
  }
  std::cout << "Query returned size:" << q_new.size() << std::endl;
  q_new.resize(0);
  q_old.resize(0);
  
  low=Uintah::IntVector(10,10,10);
  high=Uintah::IntVector(9,9,9);
  bvh.query(low,high,q_new);
  prt.query(low,high,q_old);
  if(!compareQueries(q_new,q_old))
  {
    std::cout << "Error queries do not match(5)\n";
    std::cout << "old:\n";
    for(Uintah::Level::selectType::iterator iter=q_old.begin(); iter!=q_old.end(); iter++)
    {
      std::cout << **iter << std::endl;
    }
    std::cout << "new:\n";
    for(Uintah::Level::selectType::iterator iter=q_new.begin(); iter!=q_new.end(); iter++)
    {
      std::cout << **iter << std::endl;
    }
  }
  std::cout << "Query returned size:" << q_new.size() << std::endl;
  q_new.resize(0);
  q_old.resize(0);
  
  std::cout << "All tests successfully passed\n";
  return 0;
}
