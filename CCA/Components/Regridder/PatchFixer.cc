
#include <Packages/Uintah/CCA/Components/Regridder/PatchFixer.h>
using namespace Uintah;

#include <stack>

using namespace std;

void PatchFixer::BuildLattice(const vector<PseudoPatch> &patches)
{
	//bound patches

	bounds=patches[0];
	for(unsigned int p=1;p<patches.size();p++)
	{
		for(unsigned int d=0;d<3;d++)
		{
			if(patches[p].high[d]>bounds.high[d])
				bounds.high[d]=patches[p].high[d];
			if(patches[p].low[d]<bounds.low[d])
				bounds.low[d]=patches[p].low[d];
		}								
	}

	//allocate celltolattice mapping
	csize=bounds.high-bounds.low+IntVector(1,1,1);
	cellstolattice[0].resize(csize[0]);
	cellstolattice[1].resize(csize[1]);
	cellstolattice[2].resize(csize[2]);
	

	//initialize celltolattice mapping
	cellstolattice[0].assign(csize[0],-99999999);
	cellstolattice[1].assign(csize[1],-99999999);
	cellstolattice[2].assign(csize[2],-99999999);

	//mark each lattice point that exists
	for(unsigned int p=0;p<patches.size();p++)
	{
		for(int d=0;d<3;d++)
		{
			cellstolattice[d][patches[p].low[d]-bounds.low[d]]=1;
			cellstolattice[d][patches[p].high[d]-bounds.low[d]]=1;
		}
	}

	//create mappings
	for(unsigned int d=0;d<3;d++)
	{
		int l=-1;	//lattice coordinate
		for(int i=0;i<csize[d];i++)
		{
			if(cellstolattice[d][i]==1)		//edge exists
			{
				l++;
				latticetocells[d].push_back(i+bounds.low[d]);		//map lattice coordinate to cell coordinate
			}
			cellstolattice[d][i]=l;
		}	
		lsize[d]=l;
	}
				
	//make lattice
	int size=lsize[0]*lsize[1]*lsize[2];
	lattice.resize(size);
	lattice.assign(size,-1);
				
	//Fill lattice
	for(unsigned int p=0;p<patches.size();p++)
	{
		Fill(patches[p],p);
	}

}
void PatchFixer::Fill(const PseudoPatch patch,const int id)
{
	int Y=lsize[0],Z=Y*lsize[1];
	int b[3]={patch.low[0],patch.low[1],patch.low[2]};
	int e[3]={patch.high[0],patch.high[1],patch.high[2]};

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
						lattice[index]=id;
						index++;
					}
			}
	}
}
void PatchFixer::FixUp(vector<PseudoPatch> &patches)
{
	//search lattice
  int size=patches.size()/numprocs;
	int rem=patches.size()%numprocs;
	int mystart=0,myend;
	int mysize=size;
	
	if(rank<rem)
	  mysize++;

	if(mysize==0)
	{
		patches.resize(0);
	}
	else
	{
		for(int p=0;p<numprocs;p++)
		{
		  if(p==rank)
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
	
		stack <PseudoPatch> search;
		for(int p=0;p<mysize;p++)
		{
			search.push(patches[p]);
		}	

		PseudoPatch current;
		while(!search.empty())
		{
			current=search.top();
			search.pop();
			/*
			cout << "rank:" << rank << ": searching patch: {"  
					 << current.low[0] << "-" << current.high[0] << ", "
					 << current.low[1] << "-" << current.high[1] << ", "
					 << current.low[2] << "-" << current.high[2] << "}\n";
			*/
			FixFace(patches,current,0,-1);
			FixFace(patches,current,0,1);
			FixFace(patches,current,1,-1);
			FixFace(patches,current,1,1);
			FixFace(patches,current,2,-1);
			FixFace(patches,current,2,1);
		}
	}
	vector<int> patch_sizes(numprocs);
	vector<int> displacements(numprocs);
	int my_patch_size=patches.size();
	//allgather patchset sizes
	MPI_Allgather(&my_patch_size,1,MPI_INT,&patch_sizes[0],1,MPI_INT,comm);	
	
	int total_size=patch_sizes[0];
	displacements[0]=0;
	patch_sizes[0]*=sizeof(PseudoPatch);
	for(int p=1;p<numprocs;p++)
	{
		displacements[p]=total_size*sizeof(PseudoPatch);
		total_size+=patch_sizes[p];
		patch_sizes[p]*=sizeof(PseudoPatch);
	}
	vector<PseudoPatch> mypatches(patches);
	
	patches.resize(total_size);
	/*
	cout << "rank:" << rank << ": total size:" << total_size << endl;	
	
	cout << "rank:" << rank << ": displacements: " ;	
	for(int p=0;p<numprocs;p++)
	{
			cout << displacements[p] << " ";
	}
	cout << endl;
	cout << "rank:" << rank << ": counts: " ;	
	for(int p=0;p<numprocs;p++)
	{
			cout << patch_sizes[p] << " ";
	}
	cout << endl;
	*/
	//allgatherv patchsets
	MPI_Allgatherv(&mypatches[0],my_patch_size*sizeof(PseudoPatch),MPI_BYTE,&patches[0],&patch_sizes[0],&displacements[0],MPI_BYTE,comm);
}

void PatchFixer::FixFace(vector<PseudoPatch> &patches,PseudoPatch patch, int dim, int side)
{
	int Y=lsize[0],Z=Y*lsize[1];
	int xdim,ydim,zdim,Xm,Ym,Zm;
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
	}
	if(side==-1)
	{
		x=CellToLattice(patch.low[xdim],xdim)-1;
	}
	else if(side==1)
	{
		x=CellToLattice(patch.high[xdim],xdim);
	}
	else
	{
		cout << "error invalid side in fixup\n";
		exit(0);
	}
	if(x>=0 && x<lsize[xdim])	//only search if i'm not beyond the bounds
	{
		by=CellToLattice(patch.low[ydim],ydim);
		ey=CellToLattice(patch.high[ydim],ydim);
		bz=CellToLattice(patch.low[zdim],zdim);
		ez=CellToLattice(patch.high[zdim],zdim);

		for(int y=by;y<ey;y++)
		{
			bool state=lattice[x*Xm+y*Ym+bz*Zm]!=-1;
			bool newstate;
			for(int z=bz+1;z<ez;z++)
			{
				newstate=lattice[x*Xm+y*Ym+z*Zm]!=-1;
				if(state!=newstate)
				{
					state=newstate;
					p1=lattice[(x-side)*Xm+y*Ym+z*Zm];
					p2=lattice[(x-side)*Xm+y*Ym+(z-1)*Zm];
					if(p1==p2)
					{
						Split split;
						split.d=zdim;
						split.index=LatticeToCell(z,zdim);
						SplitPatch(p1,patches,split);
					}
				}
			}
		}
		
		for(int z=bz;z<ez;z++)
		{
			bool state=lattice[x*Xm+by*Ym+z*Zm]!=-1;
			bool newstate;
			for(int y=by+1;y<ey;y++)
			{
				newstate=lattice[x*Xm+y*Ym+z*Zm]!=-1;
				if(state!=newstate)
				{
					state=newstate;
					p1=lattice[(x-side)*Xm+y*Ym+z*Zm];
					p2=lattice[(x-side)*Xm+(y-1)*Ym+z*Zm];
					if(p1==p2)
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
	
}

void PatchFixer::SplitPatch(int index, vector<PseudoPatch> &patches, const Split &split)
{
	PseudoPatch right=patches[index];
	
	patches[index].high[split.d]=right.low[split.d]=split.index;

	patches.push_back(right);

	Fill(right,(int)patches.size()-1);
}
				
