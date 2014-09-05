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



#include <CCA/Components/Regridder/PatchFixer.h>

#include <stack>

using namespace std;
using namespace Uintah;

void
PatchFixer::BuildLattice(const vector<Region> &patches)
{
  //bound patches
  bounds_=patches[0];
  for(unsigned int p=1;p<patches.size();p++)
    {
      bounds_.high()=Max(patches[p].getHigh(),bounds_.getHigh());
      bounds_.low()=Min(patches[p].getLow(),bounds_.getLow());
    }

  //allocate celltolattice mapping
  csize_=bounds_.getHigh()-bounds_.getLow()+IntVector(1,1,1);
  cellstolattice_[0].resize(csize_[0]);
  cellstolattice_[1].resize(csize_[1]);
  cellstolattice_[2].resize(csize_[2]);

  latticetocells_[0].resize(0);  
  latticetocells_[1].resize(0);  
  latticetocells_[2].resize(0);  

  //initialize celltolattice mapping
  cellstolattice_[0].assign(csize_[0],0);
  cellstolattice_[1].assign(csize_[1],0);
  cellstolattice_[2].assign(csize_[2],0);

  //mark each lattice point that exists
  for(unsigned int p=0;p<patches.size();p++)
    {
      for(int d=0;d<3;d++)
        {
          cellstolattice_[d][patches[p].getLow()[d]-bounds_.getLow()[d]]=1;
          cellstolattice_[d][patches[p].getHigh()[d]-bounds_.getLow()[d]]=1;
        }
    }

  //create mappings
  for(unsigned int d=0;d<3;d++)
    {
      int l=-1; //lattice coordinate
      for(int i=0;i<csize_[d];i++)
        {
          if(cellstolattice_[d][i]==1)          //edge exists
            {
              l++;
              latticetocells_[d].push_back(i+bounds_.getLow()[d]);              //map lattice coordinate to cell coordinate
            }
          cellstolattice_[d][i]=l;
        }       
      lsize_[d]=l;
    }
  //make lattice
  int size=lsize_[0]*lsize_[1]*lsize_[2];
  lattice_.resize(size);
  lattice_.assign(size,-1);
                                
  //Fill lattice
  for(unsigned int p=0;p<patches.size();p++)
    {
      Fill(patches[p],p);
    }
}

//Fill the lattice in the volume of patch with the value id
void PatchFixer::Fill(const Region patch,const int id)
{
  int Y=lsize_[0],Z=Y*lsize_[1];
  int b[3]={patch.getLow()[0],patch.getLow()[1],patch.getLow()[2]};
  int e[3]={patch.getHigh()[0],patch.getHigh()[1],patch.getHigh()[2]};

  CellToLattice(b);
  CellToLattice(e);

  int index;
  for(int z=b[2];z<e[2];z++)
    {
      for(int y=b[1];y<e[1];y++)
        {
          index=b[0]+y*Y+z*Z;
          for(int x=b[0];x<e[0];x++)
            {
              lattice_[index]=id;
              index++;
            }
        }
    }
}

void
PatchFixer::FixUp(vector<Region> &patches)
{
  //search lattice
  int size=patches.size()/d_myworld->size();
  int rem=patches.size()%d_myworld->size();
  int mystart=0,myend;
  int mysize=size;
        
  if(d_myworld->myrank()<rem)
    mysize++;

  if(mysize==0)
    {
      patches.resize(0);
    }
  else
    {
      for(int p=0;p<d_myworld->size();p++)
        {
          if(p==d_myworld->myrank())
            break;
          if(p<rem)
            mystart+=size+1;
          else
            mystart+=size;
        }
      myend=mystart+size;

      //move my patches to the front
      for(int p=0;p<mysize;p++)
        {
          swap(patches[p],patches[mystart+p]);  
        }       
        
      //make lattice
      BuildLattice(patches);
        
      //delete all patches that are not mine
      patches.resize(mysize);
        
      //make fixup search area
      stack <Region> search;
      for(int p=0;p<mysize;p++)
        {
          search.push(patches[p]);
        }       

      Region current;
      //search area and fix each face
      while(!search.empty())
        {
          current=search.top();
          search.pop();
                        
          FixFace(patches,current,0,-1);
          FixFace(patches,current,0,1);
          FixFace(patches,current,1,-1);
          FixFace(patches,current,1,1);
          FixFace(patches,current,2,-1);
          FixFace(patches,current,2,1);
        }
    }
  int my_patch_size=patches.size();

  //it may be worth moving splitting patches below the threashold to here
 
  //allgather patchset sizes
  if(d_myworld->size()>1)
    {
      vector<int> patch_sizes(d_myworld->size());
      vector<int> displacements(d_myworld->size());
      MPI_Allgather(&my_patch_size,1,MPI_INT,&patch_sizes[0],1,MPI_INT,d_myworld->getComm());   

      int total_size=patch_sizes[0];
      displacements[0]=0;
      patch_sizes[0]*=sizeof(Region);
      for(int p=1;p<d_myworld->size();p++)
        {
          displacements[p]=total_size*sizeof(Region);
          total_size+=patch_sizes[p];
          patch_sizes[p]*=sizeof(Region);
        }
      vector<Region> mypatches(patches);
      patches.resize(total_size);
      //allgatherv patchsets
      MPI_Allgatherv(&mypatches[0],my_patch_size*sizeof(Region),MPI_BYTE,&patches[0],&patch_sizes[0],&displacements[0],MPI_BYTE,d_myworld->getComm());
    }
}

void
PatchFixer::FixFace(vector<Region> &patches,Region patch, int dim, int side)
{
  int Y=lsize_[0],Z=Y*lsize_[1];
  int xdim=0,ydim=0,zdim=0,Xm=0,Ym=0,Zm=0;
  int x,by,bz,ey,ez;
  int p1,p2;

  switch(dim)
    {
    case 0:
      xdim=0;Xm=1;
      ydim=1;Ym=Y;
      zdim=2;Zm=Z;
      break;
    case 1:
      xdim=1;Xm=Y;
      ydim=0;Ym=1;
      zdim=2;Zm=Z;
      break;
    case 2:
      xdim=2;Xm=Z;
      ydim=0;Ym=1;
      zdim=1;Zm=Y;
      break;
    default:
      cout << "Invalid dimension in FixFace\n";
      exit(0);
    }
  if(side==-1)
    {
      x=CellToLattice(patch.getLow()[xdim],xdim)-1;
    }
  else if(side==1)
    {
      x=CellToLattice(patch.getHigh()[xdim],xdim);
    }
  else
    {
      cout << "error invalid side in fixup\n";
      exit(0);
    }
  if(x>=0 && x<lsize_[xdim])    //only search if i'm not beyond the bounds_
    {
      by=CellToLattice(patch.getLow()[ydim],ydim);
      ey=CellToLattice(patch.getHigh()[ydim],ydim);
      bz=CellToLattice(patch.getLow()[zdim],zdim);
      ez=CellToLattice(patch.getHigh()[zdim],zdim);
      //search along face in one dimension
      for(int y=by;y<ey;y++)
        {
          bool state=lattice_[x*Xm+y*Ym+bz*Zm]!=-1;
          bool newstate;
          for(int z=bz+1;z<ez;z++)
            {
              newstate=lattice_[x*Xm+y*Ym+z*Zm]!=-1;
              if(state!=newstate) //change of state indicates edge needed
                {
                  state=newstate;
                  p1=lattice_[(x-side)*Xm+y*Ym+z*Zm];
                  p2=lattice_[(x-side)*Xm+y*Ym+(z-1)*Zm];
                  if(p1==p2)        //no edge exists so split
                    {
                      Split split;
                      split.d=zdim;
                      split.index=LatticeToCell(z,zdim);
                      SplitPatch(p1,patches,split);
                    }
                }
            }
        }
      //search along face in the other dimension
      for(int z=bz;z<ez;z++)
        {
          bool state=lattice_[x*Xm+by*Ym+z*Zm]!=-1;
          bool newstate;
          for(int y=by+1;y<ey;y++)
            {
              newstate=lattice_[x*Xm+y*Ym+z*Zm]!=-1;
              if(state!=newstate) //change of state indicates edge needed
                {
                  state=newstate;
                  p1=lattice_[(x-side)*Xm+y*Ym+z*Zm];
                  p2=lattice_[(x-side)*Xm+(y-1)*Ym+z*Zm];
                  if(p1==p2)        //no edge exists so split
                    {
                      Split split;
                      split.d=ydim;
                      split.index=LatticeToCell(y,ydim);
                      SplitPatch(p1,patches,split);
                    }
                }
            }
        }
    }
} // end FixFace()

void
PatchFixer::SplitPatch(int index, vector<Region> &patches, const Split &split)
{
  Region right=patches[index];
  patches[index].high()[split.d]=right.low()[split.d]=split.index;

  patches.push_back(right);

  //update lattice
  Fill(right,(int)patches.size()-1);
}
                                
