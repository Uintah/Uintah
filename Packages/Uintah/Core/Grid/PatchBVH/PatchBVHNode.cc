
#include <Packages/Uintah/Core/Grid/PatchBVH/PatchBVHNode.h>
#include <Packages/Uintah/Core/Grid/PatchBVH/PatchBVHLeaf.h>

namespace Uintah {

  /**************************************

    CLASS
    PatchBVHNode

    A Bounding Volume Hiearchy for querying patches that are 
    within a given range.  This class is a general node of the tree.

    GENERAL INFORMATION

    PatchBVHNode.cc

    Justin Luitjens
    Department of Computer Science
    University of Utah

    Center for the Simulation of Accidental Fires and Explosions (C-SAFE)

    Copyright (C) 2008 SCI Group

    KEYWORDS
    PatchBVH

    DESCRIPTION
    The PatchBVH is used for querying patches within a given range.
    WARNING

   ****************************************/

  PatchBVHNode::PatchBVHNode(std::vector<PatchKeyVal>::iterator begin, std::vector<PatchKeyVal>::iterator end) : left_(NULL), right_(NULL)
  {
    //set bounding box
    low_=begin->patch->getCellLowIndex__New();
    high_=begin->patch->getCellHighIndex__New();

    for(std::vector<PatchKeyVal>::iterator iter=begin+1; iter<end; iter++)
    {
       low_=Min(low_,iter->patch->getCellLowIndex__New());
       high_=Max(high_,iter->patch->getCellHighIndex__New());
    }

    //find maximum dimension
    IntVector range=high_-low_;
    int maxd=0;

    if(range[1]>range[maxd])
      maxd=1;
    if(range[2]>range[maxd])
      maxd=2;
     
    //sort on maiximum dimension
    sortDim_=maxd;
    sort(begin,end);

    //split the list in half

    //create left and right nodes/leafs
    unsigned int size=(end-begin);
    unsigned int left_size=size/2;
    unsigned int right_size=size-left_size;

    if(left_size>PatchBVHBase::getLeafSize())
    {
      //create new node
      left_=new PatchBVHNode(begin,begin+left_size);
    }
    else
    {
      //create new leaf
      left_=new PatchBVHLeaf(begin,begin+left_size);
    }

    if(right_size>PatchBVHBase::getLeafSize())
    {
      //create new node
      right_=new PatchBVHNode(begin+left_size,end);
    }
    else
    {
      //create new leaf
      right_=new PatchBVHLeaf(begin+left_size,end);
    }
  }

  PatchBVHNode::~PatchBVHNode()
  {
    if(left_!=NULL)
      delete left_;
    if(right_!=NULL)
      delete right_;
  }

  void PatchBVHNode::query(const IntVector& low, const IntVector& high, Level::selectType& patches)
  {
    //check that the query intersects my bounding box
    if(!intersects(low,high,low_,high_))
      return;

    //intersect with left and right trees
    left_->query(low,high,patches);
    right_->query(low,high,patches);
  }

} // end namespace Uintah
