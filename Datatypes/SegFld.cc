/*
 *  SegFldToSurfs.cc:  SegFldToSurfs a set of points to a surface
 *
 *  Written by:
 *   David Weinstein
 *   Department of Computer Science
 *   University of Utah
 *   July 1996
 *
 *  Copyright (C) 1996 SCI Group
 */

#include <Classlib/NotFinished.h>
#include <Classlib/Queue.h>
#include <Classlib/String.h>
#include <Datatypes/ScalarFieldRGchar.h>
#include <Datatypes/ScalarFieldRG.h>
#include <Malloc/Allocator.h>
#include <iostream.h>
#include <stdio.h>

#include <Datatypes/SegFld.h>

static Persistent* maker()
{
    return scinew SegFld;
}

PersistentTypeID SegFld::type_id("SegFld", "ScalarFieldRGint", maker);

SegFld::SegFld()
: ScalarFieldRGint()
{
}

void SegFld::printComponents() {
    for (int i=0; i<comps.size(); i++) {
	if (comps[i] != 0)
	    cerr << "Component "<<i<<": size="<<get_size(comps[i])<<" type="<<get_type(comps[i])<<"\n";
    }
}

void SegFld::compress() {
    Array1<int> invmap;
    Array1<int> map(comps.size());
    map.initialize(-1);

    for (int i=0; i<comps.size(); i++)
	if (comps[i]) {
	    map[i]=invmap.size();
	    invmap.add(i);
	}
    if (invmap.size() == comps.size()) return;
    Array1<int> newcomps(invmap.size());
    for (i=0; i<newcomps.size(); i++)
	newcomps[i]=comps[invmap[i]];
    comps.resize(newcomps.size());
    for (i=0; i<comps.size(); i++)
	comps[i]=newcomps[i];

    Array1<Array1<tripleInt> *> newcompMembers;
    newcompMembers.resize(newcomps.size());
    for (i=0; i<newcompMembers.size(); i++)
	newcompMembers[i]=compMembers[invmap[i]];
    for (i=0; i<compMembers.size(); i++)
	compMembers[i]=0;
    compMembers.resize(newcompMembers.size());
    for (i=0; i<compMembers.size(); i++)
	compMembers[i]=newcompMembers[i];
    
    for (i=0; i<nx; i++)
	for (int j=0; j<ny; j++)
	    for (int k=0; k<nz; k++)
		grid(i,j,k)=map[grid(i,j,k)];
}

void SegFld::audit() {
    Array1<int> cc(comps.size());
    cc.initialize(0);
    for (int i=0; i<nx; i++)
	for (int j=0; j<ny; j++)
	    for (int k=0; k<nz; k++)
	        cc[grid(i,j,k)]++;
    int passed=1;
    for (i=0; i<cc.size(); i++) {
	if (cc[i] != get_size(comps[i])) {
	    cerr << "Audit error in component "<<i<<" should be "<<cc[i]<<" but found "<<get_size(comps[i])<<"\n";
	    passed=0;
	}
    }
    if (!passed) cerr << "Audit failed.\n";
    else cerr << "Audit passed.\n";
}

SegFld::SegFld(const SegFld& copy)
: ScalarFieldRGint(copy)
{
}

SegFld::SegFld(ScalarFieldRGchar* sf) {
    resize(sf->nx, sf->ny, sf->nz);
    Point min, max;
    sf->get_bounds(min,max);
    set_bounds(min,max);
    bldFromChar(sf);
}

SegFld::~SegFld()
{
}

ScalarField* SegFld::clone()
{
    return scinew SegFld(*this);
}

#define SegFld_VERSION 1

void SegFld::io(Piostream& stream)
{
    int version=stream.begin_class("SegFld", SegFld_VERSION);
	// Do the base class first...
    ScalarFieldRGint::io(stream);
    Pio(stream, comps);
    Pio(stream, compMembers);
    stream.end_class();
}

ScalarFieldRGchar* SegFld::getTypeFld() {
    ScalarFieldRGchar* c=scinew ScalarFieldRGchar();
    Point min, max;
    get_bounds(min, max);
    c->resize(nx, ny, nz);
    c->set_bounds(min, max);
    for (int i=0; i<nx; i++)
	for (int j=0; j<ny; j++)
	    for (int k=0; k<nz; k++)
		c->grid(i,j,k)=get_type(comps[grid(i,j,k)]);
    return c;
}

ScalarFieldRG* SegFld::getBitFld() {
    ScalarFieldRG* c=scinew ScalarFieldRG();
    Point min, max;
    get_bounds(min, max);
    c->resize(nx, ny, nz);
    c->set_bounds(min, max);
    for (int i=0; i<nx; i++)
	for (int j=0; j<ny; j++)
	    for (int k=0; k<nz; k++)
		if (get_type(comps[grid(i,j,k)]) == 0) c->grid(i,j,k)=0;
		else c->grid(i,j,k)=1<<(get_type(comps[grid(i,j,k)])-1);
    return c;
}

//void SegFld::setCompsFromGrid() {
//    comps.resize(0);
//    compMembers.resize(0);
//    Array1<char> valid;
//    int i,j,k;
//    int vox;
//    for (i=0; i<nx; i++) {
//	for (j=0; j<ny; j++) {
//	    for (k=0; k<nz; k++) {
//		vox=grid(i,j,k);
//		int comp=get_component(vox);
//		grid(i,j,k)=comp;
//		if (comp>=comps.size()) {
//		    comps.resize(comp+1);
//		    compMembers.resize(comp+1);
//		    comps[comp]=vox;
//		    int was=valid.size();
//		    valid.resize(comp+1);
//		    for (int ii=was; ii<valid.size(); ii++) valid[ii]=0;
//		    valid[comp]=1;
//		} else if (!valid[comp]) {
//		    comps[comp]=comp;
//		    valid[comp]=1;
//		}
//		compMembers[comp].add(tripleInt(i,j,k));
//	    }
//	}
//    }
//}

//void SegFld::setGridFromComps() {
//    int i,j,k;
//    int c;
//    for (i=0; i<nx; i++) {
//	for (j=0; j<ny; j++) {
//	    for (k=0; k<nz; k++) {
//		c=grid(i,j,k);
//		grid(i,j,k)=get_index(comps[c].x,comps[c].y,comps[c].z);
//	    }
//	}
//   }
//}

void SegFld::annexComponent(int old_comp, int new_comp) {
//    cerr << "setting component "<<old_comp<<" to new comp:"<<new_comp<<"\n";
    int i;
    tripleInt idx;
    for (i=0; i<compMembers[old_comp]->size(); i++) {
	idx=(*compMembers[old_comp])[i];
	grid(idx.x, idx.y, idx.z)=new_comp;
	compMembers[new_comp]->add(idx);
    }
    delete compMembers[old_comp];
    compMembers[old_comp]=0;
}

void SegFld::bldFromChar(ScalarFieldRGchar* ch) {
    int i,j,k,ii,jj,kk;
    comps.resize(0);
    cerr << "Number of independent components: ";
    for (i=0; i<compMembers.size(); i++) delete compMembers[i];
    compMembers.resize(0);
    Queue<tripleInt> q;
    Array3<char> visited(nx, ny, nz);
    visited.initialize(0);
    Array1<int> is, js, ks;
    for (i=0; i<nx; i++) {
	for (j=0; j<ny; j++) {
	    for (k=0; k<nz; k++) {
		if (!visited(i,j,k)) {
		    cerr << comps.size()<<" ";
		    q.append(tripleInt(i,j,k));
		    int compNum=comps.size();
		    char currTypeChar=ch->grid(i,j,k);
		    int currType=currTypeChar-'0';
		    grid(i,j,k)=compNum;
		    comps.add(get_index(currType, 0));
		    compMembers.resize(compNum+1);
		    compMembers[compNum]=scinew Array1<tripleInt>;
		    compMembers[compNum]->add(tripleInt(i,j,k));
		    visited(i,j,k)=1;
		    while(!q.is_empty()) {
			// enqueue non-visited neighbors
			tripleInt next=q.pop();
			compMembers[compNum]->add(next);
			comps[compNum]++;
			int iii=next.x;
			int jjj=next.y;
			int kkk=next.z;

			is.resize(0); js.resize(0); ks.resize(0);

#if 0
			// this is for vertex (diagonally) connected components
			for (ii=Max(iii-1,0); ii<Min(iii+2,nx); ii++)
			    for (jj=Max(jjj-1,0); jj<Min(jjj+2,ny); jj++)
				for (kk=Max(kkk-1,0); kk<Min(kkk+2,nz); kk++)
				    { is.add(ii); js.add(jj); ks.add(kk); }
#endif

			// this version is for face-connected components
			if (iii>0)
			    { is.add(iii-1); js.add(jjj); ks.add(kkk); }
			if (iii<nx-1)
    			    { is.add(iii+1); js.add(jjj); ks.add(kkk); }
			if (jjj>0)
			    { is.add(iii); js.add(jjj-1); ks.add(kkk); }
			if (jjj<ny-1)
    			    { is.add(iii); js.add(jjj+1); ks.add(kkk); }
			if (kkk>0)
			    { is.add(iii); js.add(jjj); ks.add(kkk-1); }
			if (kkk<nz-1)
    			    { is.add(iii); js.add(jjj); ks.add(kkk+1); }
			
			for (int nbr=0; nbr<is.size(); nbr++) {
			    ii=is[nbr]; jj=js[nbr]; kk=ks[nbr];
			    if (!visited(ii,jj,kk) &&
				(ch->grid(ii,jj,kk)==currTypeChar)) {
				grid(ii,jj,kk)=compNum;
				q.append(tripleInt(ii,jj,kk));
				visited(ii,jj,kk)=1;
			    }
			}
		    }
		}
	    }
	}
    }
    cerr << "DONE!\n";
}

void SegFld::killSmallComponents(int min) {
    int i;
    int ii,jj,kk;
    tripleInt idx;
    int min_sz=0;
    Array3<char> visited(nx,ny,nz);
    int min_comp=-1;
    Queue<tripleInt> visit_q;
    Array1<int> bdry_comps(comps.size());
    Array1<int> type_count(6);
    while (min_sz<min) {
	min_sz=get_size(comps[0]);
	i=1;
	while(min_sz==0) {
	    min_sz=get_size(comps[i]);
	    i++;
	}
	for (; i<comps.size(); i++) {
	    if ((get_size(comps[i]) != 0) && (get_size(comps[i]) < min_sz)) {
		min_comp=i;
		min_sz=get_size(comps[i]);
	    }
	}
	if (min_sz>min) break;
	idx=(*compMembers[min_comp])[0];
	cerr << "C: "<<min_comp<<", size="<<get_size(comps[min_comp])<<", type="<<get_type(comps[min_comp]);
	visited.initialize(0);
	visit_q.append(idx);
	int max_sz=0;
	int max_comp;
	int max_type;
	bdry_comps.initialize(0);
	while(!visit_q.is_empty()) {
	    // enqueue non-visited neighbors
	    tripleInt next=visit_q.pop();
	    int iii=next.x;
	    int jjj=next.y;
	    int kkk=next.z;
	    for (ii=Max(iii-1,0); ii<Min(iii+2,nx); ii++) {
		for (jj=Max(jjj-1,0); jj<Min(jjj+2,ny); jj++) {
		    for (kk=Max(kkk-1,0); kk<Min(kkk+2,nz); kk++) {
			if (!visited(ii,jj,kk)) {
			    visited(ii,jj,kk)=1;
			    if(grid(ii,jj,kk)==min_comp) {
				visit_q.append(tripleInt(ii,jj,kk));
			    } else {
				bdry_comps[grid(ii,jj,kk)]=1;
			    }
			}
		    }
		}
	    }
	}
	cerr << " ok ";
	int bdryCompsSize=0;
	type_count.initialize(0);
//	cerr << "   Here are all of the bdry components/types/sizes: ";
	for (ii=0; ii<bdry_comps.size(); ii++) {
	    if (bdry_comps[ii] != 0) {
//		cerr << ii <<"/"<<get_type(comps[ii])<<"/"<<get_size(comps[ii])<<" ";
//		if (type_count[get_type(comps[ii])] != 0)
//		    cerr << "**** GOT IT!!!! ****\n";
		bdryCompsSize++;
		type_count[get_type(comps[ii])]+=get_size(comps[ii]);
	    }
	}	
	cerr << " #nbrs="<<bdryCompsSize;
//	cerr << "\n   Here are the counts type/size: ";
	for (ii=0; ii<type_count.size(); ii++) {
//	    if (type_count[ii] != 0) cerr << ii <<"/" << type_count[ii]<<" ";
	    if (type_count[ii]>max_sz) {
		max_type=ii;
		max_sz=type_count[ii];
	    }
	}

	max_comp=-1;

//	cerr << "\n   Max type/size=: "<<max_type<<"/"<<max_sz<<"\n";
	for (ii=0; ii<bdry_comps.size(); ii++)
	    if (bdry_comps[ii] != 0) {
		if (get_type(comps[ii]) == max_type) {
		    if (max_comp == -1 ||
			get_size(comps[ii]) > get_size(comps[max_comp])) {
			max_comp=ii;
		    }
		}
	    }
//	cerr << "  new_comp="<<max_comp<<" (size was "<<get_size(comps[max_comp])<<", now it's "<<max_sz+min_sz<<"  ";
	for (ii=0; ii<bdry_comps.size(); ii++)
	    if (bdry_comps[ii] != 0) {
		if ((get_type(comps[ii]) == max_type) &&
		    (ii != max_comp)) {
//		    cerr << "annexing comp: "<<ii<<" (was "<<get_size(comps[ii])<<")  ";
		    comps[ii]=0;
		    annexComponent(ii, max_comp);
		}
	    }
	cerr << " -- annexed by "<<max_comp<<", newsize="<<min_sz+max_sz<<"\n";
	comps[min_comp]=0;
	annexComponent(min_comp, max_comp);
	comps[max_comp] = get_index(max_type, max_sz+min_sz);
    }
    compress();
}

void Pio(Piostream& stream, tripleInt& t) {
    int i;
    if (stream.reading()) {
	Pio(stream, i);
	t.x=i>>20;
	t.y=(i>>10)&1023;
	t.z=i & 1023;
    } else {
	i=(t.x<<20)+(t.y<<10)+t.z;
	Pio(stream, i);
    }
}

#ifdef __GNUG__

#include <Classlib/Array1.cc>

template class Array1<tripleInt>;
template void Pio(Piostream&, Array1<tripleInt>&);

#endif


#ifdef __sgi
#if _MIPS_SZPTR == 64
#include <Classlib/Array1.cc>

static void _dummy_(Piostream& p1, Array1<tripleInt>& p2)
{
    Pio(p1, p2);
}

#endif
#endif

