
#ifndef Uintah_Core_Grid_ComputeSet_h
#define Uintah_Core_Grid_ComputeSet_h

#include <Core/Util/Assert.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Exceptions/InternalError.h>
#include <Packages/Uintah/Core/ProblemSpec/RefCounted.h>
#include <Packages/Uintah/Core/ProblemSpec/constHandle.h>
#include <vector>
#include <algorithm>
#include <iostream>
#include <sstream>

namespace Uintah {
  using SCIRun::InternalError;
  using std::vector;
  using std::cerr;
  using std::ostringstream;

  template<class T>
  class ComputeSubset : public RefCounted {
  public:
    ComputeSubset(int size = 0)
      : items(size)
    {
    }
    ComputeSubset(const vector<T>& items)
      : items(items)
    {
    }

    int size() const {
      return (int)items.size();
    }

    T& operator[](int i) {
      return items[i];
    }
    const T& operator[](int i) const {
      return items[i];
    }
    const T& get(int i) const {
      return items[i];
    }
    void add(const T& i) {
      items.push_back(i);
    }
    bool empty() const {
      return items.size() == 0;
    }
    bool contains(T elem) const {
      for(int i=0;i<(int)items.size();i++){
	if(items[i] == elem)
	  return true;
      }
      return false;
    }
    void sort() {
      std::sort(items.begin(), items.end());
    }
    const vector<T>& getVector() const 
    { return items; }

    const ComputeSubset<T>* equals(const ComputeSubset<T>* s2) const;
    
    constHandle< ComputeSubset<T> >
    intersection(constHandle< ComputeSubset<T> > s2) const
    { return intersection(this, s2); }

    // May return the same Handle as one that came in.
    static constHandle< ComputeSubset<T> >
    intersection(const constHandle< ComputeSubset<T> >& s1,
		 const constHandle< ComputeSubset<T> >& s2)
    {
      constHandle< ComputeSubset<T> > dummy = 0;
      return intersectionAndMaybeDifferences<false>(s1, s2, dummy, dummy);
    }

    // May pass back Handles to same sets that came in.    
    static void
    intersectionAndDifferences(const constHandle< ComputeSubset<T> >& A,
			       const constHandle< ComputeSubset<T> >& B,
			       constHandle< ComputeSubset<T> >& intersection,
			       constHandle< ComputeSubset<T> >& AminusB,
			       constHandle< ComputeSubset<T> >& BminusA)
    {
      intersection =
	intersectionAndMaybeDifferences<true>(A, B, AminusB, BminusA);
    }

    static constHandle< ComputeSubset<T> > emptySubset;
  private:
    // May pass back Handles to same sets that came in.
#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1424 // template parameter not used in declaring arguments
#endif  
    template <bool passBackDifferences>
    static constHandle< ComputeSubset<T> >
    intersectionAndMaybeDifferences(const constHandle< ComputeSubset<T> >& s1,
				    const constHandle< ComputeSubset<T> >& s2,
			      constHandle< ComputeSubset<T> >& setDifference1,
			      constHandle< ComputeSubset<T> >& setDifference2);
#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1424
#endif  

 
    vector<T> items;

    ComputeSubset(const ComputeSubset&);
    ComputeSubset& operator=(const ComputeSubset&);
  };

  template<class T>
  class ComputeSet : public RefCounted {
  public:
    ComputeSet();
    ~ComputeSet();

    void addAll(const vector<T>&);
    void addEach(const vector<T>&);
    void add(const T&);

    int size() const {
       return (int)set.size();
    }
    ComputeSubset<T>* getSubset(int idx) {
      return set[idx];
    }
    const ComputeSubset<T>* getSubset(int idx) const {
      return set[idx];
    }
    const ComputeSubset<T>* getUnion() const;
    void createEmptySubsets(int size);
    int totalsize() const;
  private:
    vector<ComputeSubset<T>*> set;
    mutable ComputeSubset<T>* un;

    ComputeSet(const ComputeSet&);
    ComputeSet& operator=(const ComputeSet&);
  };

  class Patch;
  typedef ComputeSet<const Patch*>    PatchSet;
  typedef ComputeSet<int>             MaterialSet;
  typedef ComputeSubset<const Patch*> PatchSubset;
  typedef ComputeSubset<int>          MaterialSubset;
  
  template<class T>
  ComputeSet<T>::ComputeSet()
  {
    un=0;
  }

  template<class T>
  ComputeSet<T>::~ComputeSet()
  {
    for(int i=0;i<(int)set.size();i++)
      if(set[i]->removeReference())
	delete set[i];
    if(un && un->removeReference())
      delete un;
  }

  template<class T>
  void ComputeSet<T>::addAll(const vector<T>& sub)
  {
    ASSERT(!un);
    ComputeSubset<T>* subset = scinew ComputeSubset<T>(sub);
    subset->sort();
    subset->addReference();
    set.push_back(subset);
  }

  template<class T>
  void ComputeSet<T>::addEach(const vector<T>& sub)
  {
    ASSERT(!un);
    for(int i=0;i<(int)sub.size();i++){
      ComputeSubset<T>* subset = scinew ComputeSubset<T>(1);
      subset->addReference();
      (*subset)[0]=sub[i];
      set.push_back(subset);
    }
  }

  template<class T>
  void ComputeSet<T>::add(const T& item)
  {
    ASSERT(!un);
    ComputeSubset<T>* subset = scinew ComputeSubset<T>(1);
    subset->addReference();
    (*subset)[0]=item;
    set.push_back(subset);
  }

  template<class T>
  void ComputeSet<T>::createEmptySubsets(int n)
  {
    ASSERT(!un);
    for(int i=0;i<n;i++){
      ComputeSubset<T>* subset = scinew ComputeSubset<T>(0);
      subset->addReference();
      set.push_back(subset);
    }
  }

  template<class T>
  const ComputeSubset<T>* ComputeSet<T>::getUnion() const {
    if(!un){
      un = new ComputeSubset<T>;
      un->addReference();
      for(int i=0;i<(int)set.size();i++){
	ComputeSubset<T>* ss = set[i];
	for(int j=0;j<ss->size();j++){
	  un->add(ss->get(j));
	}
      }
      un->sort();
    }
    return un;
  }

  template<class T>
  int ComputeSet<T>::totalsize() const {
    if(un)
      return un->size();
    int total=0;
    for(int i=0;i<(int)set.size();i++)
      total+=set[i]->size();
    return total;
  }

  template <class T>
  static constHandle< ComputeSubset<T> > ComputeSubset<T>::emptySubset =
  scinew ComputeSubset<T>(0);
  
  
#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma set woff 1424 // template parameter not used in declaring arguments
#pragma set woff 1209 // constant controlling expressions (passBackDifference)
#endif  
  
  template<class T>
  template<bool passBackDifferences>
  constHandle< ComputeSubset<T> > ComputeSubset<T>::
  intersectionAndMaybeDifferences(const constHandle< ComputeSubset<T> >& s1,
				  const constHandle< ComputeSubset<T> >& s2,
			    constHandle< ComputeSubset<T> >& setDifference1,
			    constHandle< ComputeSubset<T> >& setDifference2)
  {
    if (s1 == s2) {
      // for efficiency -- expedite when s1 and s2 point to the same thing 
      setDifference1 = setDifference2 = ComputeSubset<T>::emptySubset;
      return s1;
    }
    
    if (passBackDifferences) {      
      setDifference1 = s1;
      setDifference2 = s2;
    }
   
    if (!s1) {
      setDifference2 = 0; // arbitrarily
      return s2; // treat null as everything -- intersecting with it gives all
    }
    if (!s2) {
      setDifference1 = 0; // arbitrarily
      return s1; // treat null as everything -- intersecting with it gives all
    }
    
    if (s1->size() == 0)
      return s1; // return an empty set
    if (s2->size() == 0)
      return s2; // return an empty set
    
    Handle< ComputeSubset<T> > intersection = scinew ComputeSubset<T>;
    Handle< ComputeSubset<T> > s1_minus_s2, s2_minus_s1;        
    
    if (passBackDifferences) {      
      setDifference1 = s1_minus_s2 = scinew ComputeSubset<T>;
      setDifference2 = s2_minus_s1 = scinew ComputeSubset<T>;
    }
    
    T el1 = s1->get(0);
    for(int i=1;i<s1->size();i++){
      T el = s1->get(i);
      if(el <= el1){
	ostringstream msgstr;
	msgstr << "Set not sorted: " << el1 << ", " << el;
	throw InternalError(msgstr.str());
      }
      el1=el;
    }
    T el2 = s2->get(0);
    for(int i=1;i<s2->size();i++){
      T el = s2->get(i);
      if(el <= el2){
	ostringstream msgstr;
	msgstr << "Set not sorted: " << el2 << ", " << el;
	throw InternalError(msgstr.str()); 
      }
      el2=el;
    }
    int i1=0;
    int i2=0;
    for(;;){
      if(s1->get(i1) == s2->get(i2)){
	intersection->add(s1->get(i1));
	i1++; i2++;
      } else if(s1->get(i1) < s2->get(i2)){
	if (passBackDifferences) {
	  s1_minus_s2->add(s1->get(i1)); // alters setDifference1
	}
	i1++;
      } else {
	if (passBackDifferences) {
	  s2_minus_s1->add(s2->get(i2)); // alters setDifference2
	}	
	i2++;
      }
      if(i1 == s1->size() || i2 == s2->size())
	break;
    }

    if (passBackDifferences) {
      if (intersection->empty()) {
	// if the intersection is empty, then the set differences are
	// the same as the sets that came in.
	setDifference1 = s1;
	setDifference2 = s2;
      }
      else {
	// get the rest of whichever difference set wasn't finished (if any)
	for (; i1 != s1->size(); i1++) {
	  s1_minus_s2->add(s1->get(i1)); // alters setDifference1
	}
	for (; i2 != s2->size(); i2++) {
	  s2_minus_s1->add(s2->get(i2));  // alters setDifference2
	}
      }
    }

    if (intersection->size() == s1->size()) {
      return s1;
    }
    else if (intersection->size() == s2->size()) {
      return s2;
    }
    else {
      return intersection;
    }
  }

#if defined(__sgi) && !defined(__GNUC__) && (_MIPS_SIM != _MIPS_SIM_ABI32)
#pragma reset woff 1424
#pragma reset woff 1209  
#endif  
  
}

#endif
