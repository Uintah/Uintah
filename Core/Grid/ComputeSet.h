
#ifndef Uintah_Core_Grid_ComputeSet_h
#define Uintah_Core_Grid_ComputeSet_h

#include <Core/Util/Assert.h>
#include <Core/Malloc/Allocator.h>
#include <Packages/Uintah/Core/ProblemSpec/RefCounted.h>
#include <vector>
#include <algorithm>
#include <iostream>

namespace Uintah {
  using std::vector;
  using std::cerr;

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

    const ComputeSubset<T>* intersection(const ComputeSubset<T>* s2) const
    { return intersection(this, s2); }

    static const ComputeSubset<T>* intersection(const ComputeSubset<T>* s1,
						const ComputeSubset<T>* s2);
  private:
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

  // Will need to do something about const-ness here.
  template<class T>
  const ComputeSubset<T>*
  ComputeSubset<T>::intersection(const ComputeSubset<T>* s1,
				 const ComputeSubset<T>* s2)
  {
    if(!s1)
      return s2;
    if(!s2)
      return s1;
    
    ComputeSubset<T>* result = scinew ComputeSubset<T>;
    
    if(s1->size() == 0 || s2->size() == 0)
      return result;
    T el1 = s1->get(0);
    for(int i=1;i<s1->size();i++){
      T el = s1->get(i);
      if(el <= el1){
	cerr << "Set not sorted: " << el1 << ", " << el << '\n';
      }
      el1=el;
    }
    T el2 = s2->get(0);
    for(int i=1;i<s2->size();i++){
      T el = s2->get(i);
      if(el <= el2){
	cerr << "Set not sorted: " << el2 << ", " << el << '\n';
      }
      el2=el;
    }
    int i1=0;
    int i2=0;
    for(;;){
      if(s1->get(i1) == s2->get(i2)){
	result->add(s1->get(i1));
	i1++; i2++;
      } else if(s1->get(i1) < s2->get(i2)){
	i1++;
      } else {
	i2++;
      }
      if(i1 == s1->size() || i2 == s2->size())
	break;
    }
    return result;
  }
}

#endif
