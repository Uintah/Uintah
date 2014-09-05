/*
  The contents of this file are subject to the University of Utah Public
  License (the "License"); you may not use this file except in compliance
  with the License.
  
  Software distributed under the License is distributed on an "AS IS"
  basis, WITHOUT WARRANTY OF ANY KIND, either express or implied. See the
  License for the specific language governing rights and limitations under
  the License.
  
  The Original Source Code is SCIRun, released March 12, 2001.
  
  The Original Source Code was developed by the University of Utah.
  Portions created by UNIVERSITY are Copyright (C) 2001, 1994 
  University of Utah. All Rights Reserved.
*/

/*
 *  SegLatVolField.cc
 *
 *  Written by:
 *   David Weinstein
 *   School of Computing
 *   University of Utah
 *
 *  Copyright (C) 2003 SCI Institute
 */

#include <Packages/BioPSE/Core/Datatypes/SegLatVolField.h>
#include <Core/Datatypes/FieldInterface.h>

#include <queue>
using std::queue;

namespace BioPSE {

// Pio defs.
const int SEG_LAT_VOL_FIELD_VERSION = 1;

Persistent*
SegLatVolField::maker()
{
  return scinew SegLatVolField;
}

PersistentTypeID 
SegLatVolField::type_id(type_name(-1), 
			LatVolField<int>::type_name(-1),
			maker);


void 
SegLatVolField::io(Piostream& stream)
{
  /*int version=*/stream.begin_class(type_name(-1), 
				     SEG_LAT_VOL_FIELD_VERSION);
  LatVolField<int>::io(stream);
  Pio(stream, comps_);
  Pio(stream, compMembers_);
  Pio(stream, maxMatl_);
  stream.end_class();
}

SegLatVolField *SegLatVolField::clone() const {
  return new SegLatVolField(*this);
}

SegLatVolField::SegLatVolField(const SegLatVolField &copy) :
  LatVolField<int>(copy), maxMatl_(copy.maxMatl_)
{
  comps_ = copy.comps_;
  compMembers_.resize(copy.compMembers_.size());
  for (int i=0; i<copy.compMembers_.size(); i++)
    compMembers_[i] = new Array1<LatVolMesh::Cell::index_type>(*(copy.compMembers_[i]));
}
  
const string 
SegLatVolField::type_name(int n)
{
  ASSERT((n >= -1) && n <= 1);
  if (n == -1)
  {
    static const string name = type_name(0) + FTNS + type_name(1) + FTNE;
    return name;

  }
  else if (n == 0)
  {
    return "SegLatVolField";
  }
  else
  {
    return find_type_name((int *)0);
  }
}

const TypeDescription*
SegLatVolField::get_type_description(int n) const
{
  ASSERT((n >= -1) && n <= 1);

  TypeDescription* td = 0;
  static string name( type_name(0) );
  static string namesp("BioPSE");
  static string path(__FILE__);

  if(!td){
    if (n == -1) {
      const TypeDescription *sub = SCIRun::get_type_description((int*)0);
      TypeDescription::td_vec *subs = scinew TypeDescription::td_vec(1);
      (*subs)[0] = sub;
      td = scinew TypeDescription(name, subs, path, namesp);
    }
    else if(n == 0) {
      td = scinew TypeDescription(name, 0, path, namesp);
    }
    else {
      td = (TypeDescription *) SCIRun::get_type_description((int*)0);
    }
  }
  return td;
}

void SegLatVolField::compress() {
  Array1<int> invmap;
  Array1<int> map(comps_.size());
  map.initialize(-1);

  int i;
  for (i=0; i<comps_.size(); i++)
    if (compSize(i) != -1) {
      map[i]=invmap.size();
      invmap.add(i);
    }
  if (invmap.size() == comps_.size()) return;
  Array1<pair<int, long> > newcomps(invmap.size());
  for (i=0; i<newcomps.size(); i++)
    newcomps[i]=comps_[invmap[i]];
  comps_.resize(newcomps.size());
  for (i=0; i<comps_.size(); i++)
    comps_[i]=newcomps[i];

  Array1<Array1<LatVolMesh::Cell::index_type> *> newcompMembers;
  newcompMembers.resize(newcomps.size());
  for (i=0; i<newcompMembers.size(); i++)
    newcompMembers[i]=compMembers_[invmap[i]];
  for (i=0; i<compMembers_.size(); i++)
    compMembers_[i]=0;
  compMembers_.resize(newcompMembers.size());
  for (i=0; i<compMembers_.size(); i++)
    compMembers_[i]=newcompMembers[i];
    
  for (i=0; i<fdata().dim1(); i++)
    for (int j=0; j<fdata().dim2(); j++)
      for (int k=0; k<fdata().dim3(); k++)
	fdata()(i,j,k)=map[fdata()(i,j,k)];
}

void SegLatVolField::audit() {
  Array1<long int> cc(comps_.size());
  cc.initialize(0);
  int i;
  for (i=0; i<fdata().dim1(); i++)
    for (int j=0; j<fdata().dim2(); j++)
      for (int k=0; k<fdata().dim3(); k++)
	cc[fdata()(i,j,k)]++;
  int passed=1;
  for (i=0; i<cc.size(); i++) {
    if (cc[i] != compSize(i)) {
      cerr << "Audit error in component "<<i<<" should be "<<cc[i]<<" but found "<<compSize(i)<<"\n";
      passed=0;
    }
  }
  if (!passed) cerr << "Audit failed.\n";
  else cerr << "Audit passed.\n";
}

void SegLatVolField::absorbComponent(int old_comp, int new_comp) {
  int i;
  LatVolMesh::Cell::index_type idx;
  for (i=0; i<compMembers_[old_comp]->size(); i++) {
    idx=(*compMembers_[old_comp])[i];
    fdata()(idx.i_, idx.j_, idx.k_)=new_comp;
    compMembers_[new_comp]->add(idx);
  }
  delete compMembers_[old_comp];
  compMembers_[old_comp]=0;
}

void buildAssocList(Array1<Array1<int> > &full, int idx, Array1<int> &equiv) {
  int i,j;
  for (i=0; i<full[idx].size(); i++) {
    int nidx=full[idx][i];
    for (j=0; j<equiv.size(); j++)
      if (equiv[j]==nidx) break;
    if (j == equiv.size()) { 
      equiv.add(nidx);
      buildAssocList(full, nidx, equiv);
    }
  }
}
	
/* returns the lowest equivalency of the label i
 * that is, the smallest label that is equivalent to this label */
int SegLatVolField::lowestValue(int i, Array1<int>& workingLabels) {
  if (i == workingLabels[i])
    return i;
  else 
    return (lowestValue(workingLabels[i], workingLabels));
}


/* sets label i and all labels it knows are equivalent to it to a newValue */
void SegLatVolField::setAll(int i, int newValue, Array1<int>& workingLabels) { 
  if (i == workingLabels[i]) {
    workingLabels[i] = newValue;
  }
  else {
    setAll(workingLabels[i], newValue, workingLabels);
    workingLabels[i] = newValue;
  }
}

void SegLatVolField::setEquiv(int larger, int smaller, Array1<int>& workingLabels) {
  int temp;

  if (larger == smaller)
    return;

  /* make sure larger number really is larger */
  if (larger < smaller) {
    temp = larger;
    larger = smaller;
    smaller = temp;
  }

  if (smaller != workingLabels[smaller]) {
    /* case 1:  when smaller number has an equivalency */
    if (larger == workingLabels[larger]) 
      workingLabels[larger] = lowestValue(smaller, workingLabels);

    /* case 2: when both numbers already have an equivalency */
    else {
      if (lowestValue(smaller, workingLabels) < 
	  lowestValue(larger, workingLabels))
        setAll(larger, lowestValue(smaller, workingLabels), workingLabels);
      else if (lowestValue(smaller, workingLabels) > 
	       lowestValue(larger, workingLabels))
        setAll(smaller, lowestValue(larger, workingLabels), workingLabels);
      /* avoid doing anything if smaller and larger are already equivalent */
    }
  }

  /* case 3: when larger number already has an equivalency */
  else if (larger != workingLabels[larger]) {
    if (lowestValue(larger, workingLabels) < smaller)
      workingLabels[smaller] = lowestValue(larger, workingLabels);
    else if (lowestValue(larger, workingLabels) > smaller)
      setAll(larger, smaller, workingLabels);
    /* avoid doing anything if smaller and larger are already equivalent */
  }

  /* case 4: when neither has an equivalency yet */
  else {
    workingLabels[larger] = smaller;
  }
}


void SegLatVolField::initialize() {
  int i,j,k;
  bool firstMatl=true;
  Array1<int> workingMatls;	// materials of components
  Array1<int> workingLabels;	// labels of components
  Array3<int> id(fdata().dim1(),fdata().dim2(),fdata().dim3());	// component index for each voxel
  id.initialize(-1);

  // traverse all of the voxels
  // for each voxel, if my material is the same as a neighbor, store their
  // component index as "newid".  if we're the same as multiple neighbors
  // with different component indices, add the bigger index to the "same"
  // list of the small index.
  // if we're not the same as any of our neighbors, we're a new component.
  for (i=0; i<fdata().dim1(); i++)
    for (j=0; j<fdata().dim2(); j++)
      for (k=0; k<fdata().dim3(); k++) {
	int currMatl=fdata()(i,j,k);
	if (firstMatl || currMatl>maxMatl_) { 
	  maxMatl_=currMatl; 
	  firstMatl=false; 
	}
	int newid=-1;
	Array1<int> equiv;
	if (i && (fdata()(i-1,j,k)==currMatl)) {
	  newid=id(i,j,k)=id(i-1,j,k);
	  equiv.add(newid);
	}
	if (j && (fdata()(i,j-1,k)==currMatl)) {
	  newid=id(i,j,k)=id(i,j-1,k);
	  int xx;
	  for (xx=0; xx<equiv.size(); xx++) 
	    if (equiv[xx]==newid) break;
	  if (xx == equiv.size()) equiv.add(newid);
	}
	if (k && (fdata()(i,j,k-1)==currMatl)) {
	  newid=id(i,j,k)=id(i,j,k-1);
	  int xx;
	  for (xx=0; xx<equiv.size(); xx++) 
	    if (equiv[xx]==newid) break;
	  if (xx == equiv.size()) equiv.add(newid);
	}
	if (newid == -1) {
	  id(i,j,k)=workingLabels.size();
	  workingLabels.add(workingLabels.size());
	  workingMatls.add(currMatl);
	} else {
	  int xx;
	  for (xx=0; xx<equiv.size()-1; xx++)
	    setEquiv(equiv[xx], equiv[xx+1], workingLabels);
	}
      }
  cerr << "Initial pass found "<<workingLabels.size()<<" components.  Removing equivalences...\n";

  int numLabels=workingLabels.size();

  // remove duplicates
  int changed = 1;
  int smallest;
  while (changed) {
    changed = 0;
    for (i = numLabels-1; i >= 0; i--) 
      if (i != workingLabels[i]) {
	smallest = lowestValue(i, workingLabels);
	if (workingLabels[i] != smallest) {
	  changed = 1;
	  setAll(i, smallest, workingLabels);
	}
      }
  }

  Array1<int> finalLabels(numLabels);

  // CreateNewLabels() code
  int counter = 0;
  for (i = 0; i < numLabels; i++) {
    if (i == workingLabels[i])
      finalLabels[i] = counter++;
    else 
      finalLabels[i] = finalLabels[workingLabels[i]];
  }
    
  for (i=0; i<compMembers_.size(); i++) delete compMembers_[i];
  compMembers_.resize(counter);
  for (i=0; i<compMembers_.size(); i++) compMembers_[i]=new Array1<LatVolMesh::Cell::index_type>;
  for (i=0; i<fdata().dim1(); i++)
    for (j=0; j<fdata().dim2(); j++)
      for (k=0; k<fdata().dim3(); k++) {
	id(i,j,k)=fdata()(i,j,k)=finalLabels[id(i,j,k)];
	compMembers_[id(i,j,k)]->add(LatVolMesh::Cell::index_type(get_typed_mesh().get_rep(),i,j,k));
      }

  comps_.resize(0);
  for (i = 0; i < numLabels; i++)
    if (i == workingLabels[i])
      comps_.add(pair<int, long>(workingMatls[i], 
				compMembers_[finalLabels[i]]->size()));

  cerr << "... ended up with "<<comps_.size()<<" components.\n";
//  printComponents();
}

void SegLatVolField::absorbSmallComponents(int min) {
  int i;
  int ii,jj,kk;
  LatVolMesh::Cell::index_type idx;
  int min_sz=0;
  Array3<char> visited(fdata().dim1(),fdata().dim2(),fdata().dim3());
  int min_comp=-1;
  queue<LatVolMesh::Cell::index_type> visit_q;
  Array1<int> bdry_comps(comps_.size());

  while (min_sz<min) {
    min_sz=compSize(0);
    min_comp=0;
    i=1;
    while(min_sz==-1) {
      min_sz=compSize(i);
      min_comp=i;
      i++;
    }
    for (; i<comps_.size(); i++) {
      if ((compSize(i) != -1) && (compSize(i) < min_sz)) {
	min_sz=compSize(i);
	min_comp=i;
      }
    }
    if (min_sz>min) break;
    idx=(*compMembers_[min_comp])[0];
    cerr << "C: "<<min_comp<<", size="<<compSize(min_comp)<<", material="<<compMatl(min_comp)<<"\n";
//    cerr << "idx.i_="<<idx.i_<<" idx.j_="<<idx.j_<<" idx.k_="<<idx.k_<<"\n";
    visited.initialize(0);
    visit_q.push(idx);
    int max_sz=0;
    int max_comp;
    int max_matl;
    bdry_comps.initialize(0);
    while(!visit_q.empty()) {
      // enqueue non-visited neighbors
      LatVolMesh::Cell::index_type next=visit_q.front();
      visit_q.pop();
      int iii=next.i_;
      int jjj=next.j_;
      int kkk=next.k_;
//      cerr << "iii="<<iii<<" jjj="<<jjj<<" kkk="<<kkk<<"\n";
      for (ii=Max(iii-1,0); ii<Min(iii+2,fdata().dim1()); ii++) {
	for (jj=Max(jjj-1,0); jj<Min(jjj+2,fdata().dim2()); jj++) {
	  for (kk=Max(kkk-1,0); kk<Min(kkk+2,fdata().dim3()); kk++) {
	    if (!visited(ii,jj,kk)) {
	      visited(ii,jj,kk)=1;
	      if(fdata()(ii,jj,kk)==min_comp) {
		visit_q.push(LatVolMesh::Cell::index_type(get_typed_mesh().get_rep(),ii,jj,kk));
	      } else {
//		cerr << "neighbor component:"<<fdata()(ii,jj,kk)<<"\n";
		bdry_comps[fdata()(ii,jj,kk)]=1;
	      }
	    }
	  }
	}
      }
    }
//    cerr << " ok ";
    int bdryCompsSize=0;
//    cerr << "  MAXMATL="<<maxMatl_<<"  ";
    Array1<int> matl_count(maxMatl_+1);
    matl_count.initialize(0);
//    cerr << "   Here are all of the bdry components/materials/sizes: ";
    for (ii=0; ii<bdry_comps.size(); ii++) {
//      cerr << ii << " of "<< bdry_comps.size() <<"       ";
      if (bdry_comps[ii] != 0) {
//	cerr << ii <<"/"<<compMatl(ii)<<"/"<<compSize(ii)<<" ";
	if (matl_count[compMatl(ii)] != 0)
//	  cerr << "**** GOT IT!!!! ****";
	bdryCompsSize++;
	matl_count[compMatl(ii)]+=compSize(ii);
      }
//      cerr << "\n";
    }	
//    cerr << " #nbrs="<<bdryCompsSize;
    //	cerr << "\n   Here are the counts material/size: ";
    for (ii=0; ii<matl_count.size(); ii++) {
      //	    if (matl_count[ii] != 0) cerr << ii <<"/" << matl_count[ii]<<" ";
      if (matl_count[ii]>max_sz) {
	max_matl=ii;
	max_sz=matl_count[ii];
      }
    }

    max_comp=-1;

    //	cerr << "\n   Max material/size=: "<<max_matl<<"/"<<max_sz<<"\n";
    for (ii=0; ii<bdry_comps.size(); ii++)
      if (bdry_comps[ii] != 0) {
	if (compMatl(ii) == max_matl) {
	  if (max_comp == -1 ||
	      compSize(ii) > compSize(max_comp)) {
	    max_comp=ii;
	  }
	}
      }
    //	cerr << "  new_comp="<<max_comp<<" (size was "<<get_size(comps_[max_comp])<<", now it's "<<max_sz+min_sz<<"  ";
    for (ii=0; ii<bdry_comps.size(); ii++)
      if (bdry_comps[ii] != 0) {
	if ((compMatl(ii) == max_matl) &&
	    (ii != max_comp)) {
	  //		    cerr << "absorbing comp: "<<ii<<" (was "<<get_size(comps_[ii])<<")  ";
	  comps_[ii].second=-1;
	  absorbComponent(ii, max_comp);
	}
      }
//    cerr << " -- absorbed by "<<max_comp<<", newsize="<<min_sz+max_sz<<"\n";
    comps_[min_comp].second=-1;
    absorbComponent(min_comp, max_comp);
    comps_[max_comp] = pair<int, long>(max_matl, max_sz+min_sz);
  }
  compress();
}

void SegLatVolField::printComponents() {
  for (int i=0; i<comps_.size(); i++) {
    if (compSize(i) != -1)
      cerr << "Component "<<i<<": size="<<compSize(i)<<" material="<<compMatl(i)<<"\n";
  }
}

void SegLatVolField::setData(const Array3<int> &data) {
  if (fdata().dim1() != data.dim1() ||
      fdata().dim2() != data.dim2() ||
      fdata().dim3() != data.dim3()) {
    cerr << "Error -- can't initialize from different sized data.\n";
    return;
  }
  for (int i=0; i<data.dim1(); i++)
    for (int j=0; j<data.dim2(); j++)
      for (int k=0; k<data.dim3(); k++)
	fdata()(i,j,k)=data(i,j,k);
  initialize();
}

} // End namespace BioPSE
