/*
 * Copyright (c) 1997-2012 The University of Utah
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the \"Software\"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and\/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED \"AS IS\", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 */


#ifndef Uintah_Core_Grid_ComputeSet_h
#define Uintah_Core_Grid_ComputeSet_h

#include <Core/Util/Assert.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Exceptions/InternalError.h>
#include <Core/Util/RefCounted.h>
#include <Core/Util/constHandle.h>

#include   <vector>
#include   <algorithm>
#include   <sstream>

namespace Uintah {

  using SCIRun::InternalError;

  using std::vector;
  using std::ostringstream;

  /**************************************

    CLASS
    ComputeSet

    Provides similar functionality to std::set.  An exception is that
    a ComputeSet stores data in groups of ComputeSubsets.   

    A ComputeSubset is much more similar to a std::set and provides 
    functionality necessary for ComputeSet.

    GENERAL INFORMATION

    ComputeSet.h

    Steven G. Parker
    Department of Computer Science
    University of Utah

    Center for the Simulation of Accidental Fires and Explosions (C-SAFE)


    KEYWORDS
    Level

    DESCRIPTION
    Long description...

    WARNING

   ****************************************/
  class Patch;
  template<class T> class ComputeSubset;
  template<class T> class ComputeSet;

  typedef ComputeSet<const Patch*>    PatchSet;
  typedef ComputeSet<int>             MaterialSet;
  typedef ComputeSubset<const Patch*> PatchSubset;
  typedef ComputeSubset<int>          MaterialSubset;


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

        template <class InputIterator>
          ComputeSubset(InputIterator begin, InputIterator end)
          : items(begin, end)
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

        void sort();
        bool is_sorted() const;

        const vector<T>& getVector() const 
        { return items; }

        bool equals(const ComputeSubset<T>* s2) const
        {
          //check that the sets are equvalent
          if(items.size() != s2->items.size())
            return false;

          for(unsigned int i=0;i<items.size();i++)
            if(items[i]!=s2->items[i])
              return false;
          
          return true;
        };

        constHandle< ComputeSubset<T> >
          intersection(constHandle< ComputeSubset<T> > s2) const
          { return intersection(this, s2); }

        // May return the same Handle as one that came in.
        static constHandle< ComputeSubset<T> >
          intersection(const constHandle< ComputeSubset<T> >& s1,
              const constHandle< ComputeSubset<T> >& s2);

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
        
        static void 
          difference(const constHandle< ComputeSubset<T> >& A,
              const constHandle< ComputeSubset<T> >& B,
              constHandle< ComputeSubset<T> >& diff)
          {
            diff=difference(A,B);
          }

        
        static bool overlaps(const ComputeSubset<T>* s1,
            const ComputeSubset<T>* s2);

        static bool compareElems(T e1, T e2);
      private:
        
        // May pass back Handles to same sets that came in.
        template <bool passBackDifferences>
          static constHandle< ComputeSubset<T> >
          intersectionAndMaybeDifferences(const constHandle< ComputeSubset<T> >& s1,
              const constHandle< ComputeSubset<T> >& s2,
              constHandle< ComputeSubset<T> >& setDifference1,
              constHandle< ComputeSubset<T> >& setDifference2);
        
        static constHandle< ComputeSubset<T> >
          difference(const constHandle< ComputeSubset<T> >& A,
              const constHandle< ComputeSubset<T> >& B);
        
        vector<T> items;

        ComputeSubset(const ComputeSubset&);
        ComputeSubset& operator=(const ComputeSubset&);
    };  // end class ComputeSubset

  template<class T>
    class ComputeSet : public RefCounted {
      public:
        ComputeSet();
        ~ComputeSet();

        // adds all unique elements of vector in one subset
        void addAll_unique(const vector<T>&);
        
        // adds all elements of vector in one subset
        void addAll(const vector<T>&);

        // adds each element of vector as a separate individual subset
        void addEach(const vector<T>&);

        // adds one element as a new subset
        void add(const T&);

        void sortSubsets();

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

         friend std::ostream& operator<<(std::ostream& out, const Uintah::PatchSet&);
         friend std::ostream& operator<<(std::ostream& out, const Uintah::MaterialSet&);
         friend std::ostream& operator<<(std::ostream& out, const Uintah::PatchSubset&);
         friend std::ostream& operator<<(std::ostream& out, const Uintah::MaterialSubset&);

    };  // end class ComputeSet

   std::ostream& operator<<(std::ostream& out, const Uintah::PatchSet&);
   std::ostream& operator<<(std::ostream& out, const Uintah::MaterialSet&);
   std::ostream& operator<<(std::ostream& out, const Uintah::PatchSubset&);
   std::ostream& operator<<(std::ostream& out, const Uintah::MaterialSubset&);

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
    void ComputeSet<T>::addAll_unique(const vector<T>& sub)
    {
       // Only insert unique entries into the set
      vector<T> sub_unique;
      sub_unique.push_back(sub[0]);

      for(int i=1;i<(int)sub.size();i++){
        if( find(sub_unique.begin(), sub_unique.end(), sub[i] ) == sub_unique.end()){
          sub_unique.push_back(sub[i]);
        }
      }
      
      ASSERT(!un);
      ComputeSubset<T>* subset = scinew ComputeSubset<T>(sub_unique);
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
    void ComputeSet<T>::sortSubsets()
    {
      for(int i=0;i<(int)set.size();i++){
        ComputeSubset<T>* ss = set[i];
        ss->sort();
      }
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
        un = scinew ComputeSubset<T>;
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

  // Note: sort is specialized in ComputeSet_special for const Patch*'s
  // to use Patch::Compare.
  template<>
     void ComputeSubset<const Patch*>::sort();

  template<class T>
    void ComputeSubset<T>::sort() {
      std::sort(items.begin(), items.end());
    }

  // specialized for patch in ComputeSet_special.cc
  template<>  
     bool ComputeSubset<const Patch*>::compareElems(const Patch* e1,
        const Patch* e2);

  template<class T>
    bool ComputeSubset<T>::compareElems(T e1, T e2)
    { return e1 < e2; }

  //compute the interesection between s1 and s2
  template<class T>
    constHandle< ComputeSubset<T> > ComputeSubset<T>::
    intersection(const constHandle< ComputeSubset<T> >& s1,
        const constHandle< ComputeSubset<T> >& s2)
    {
      if (s1 == s2) {
        // for efficiency -- expedite when s1 and s2 point to the same thing 
        return s1;
      }

      if (!s1) {
        return s2; // treat null as everything -- intersecting with it gives all
      }
      if (!s2) {
        return s1; // treat null as everything -- intersecting with it gives all
      }

      if (s1->size() == 0)
        return s1; // return an empty set
      if (s2->size() == 0)
        return s2; // return an empty set

      Handle< ComputeSubset<T> > intersection = scinew ComputeSubset<T>;

#if SCI_ASSERTION_LEVEL>0
      if (!s1->is_sorted()) {
        SCI_THROW(InternalError("ComputeSubset s1 not sorted in ComputeSubset<T>::intersectionAndMaybeDifference", __FILE__, __LINE__));
      }
      if (!s2->is_sorted()) {
        SCI_THROW(InternalError("ComputeSubset s2 not sorted in ComputeSubset<T>::intersectionAndMaybeDifference", __FILE__, __LINE__));
      }

      T el2 = s2->get(0);
      for(int i=1;i<s2->size();i++){
        T el = s2->get(i);
        if(!compareElems(el2, el)) {
          ostringstream msgstr;
          msgstr << "Set not sorted: " << el2 << ", " << el;
          SCI_THROW(InternalError(msgstr.str(), __FILE__, __LINE__)); 
        }
        el2=el;
      }
#endif
      int i1=0;
      int i2=0;
      for(;;){
        if(s1->get(i1) == s2->get(i2)){
          intersection->add(s1->get(i1));
          i1++; i2++;
        } else if(compareElems(s1->get(i1), s2->get(i2))){
          i1++;
        } else {
          i2++;
        }
        if(i1 == s1->size() || i2 == s2->size())
          break;
      }

      return intersection;
    }

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
        setDifference1 = setDifference2 = scinew ComputeSubset<T>(0);
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

#if SCI_ASSERTION_LEVEL>0
      if (!s1->is_sorted()) {
        SCI_THROW(InternalError("ComputeSubset s1 not sorted in ComputeSubset<T>::intersectionAndMaybeDifference", __FILE__, __LINE__));
      }
      if (!s2->is_sorted()) {
        SCI_THROW(InternalError("ComputeSubset s2 not sorted in ComputeSubset<T>::intersectionAndMaybeDifference", __FILE__, __LINE__));
      }

      T el2 = s2->get(0);
      for(int i=1;i<s2->size();i++){
        T el = s2->get(i);
        if(!compareElems(el2, el)) {
          ostringstream msgstr;
          msgstr << "Set not sorted: " << el2 << ", " << el;
          SCI_THROW(InternalError(msgstr.str(), __FILE__, __LINE__)); 
        }
        el2=el;
      }
#endif
      int i1=0;
      int i2=0;
      for(;;){
        if(s1->get(i1) == s2->get(i2)){
          intersection->add(s1->get(i1));
          i1++; i2++;
        } else if(compareElems(s1->get(i1), s2->get(i2))){
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
  
  template<class T>
    constHandle< ComputeSubset<T> > ComputeSubset<T>::
    difference(const constHandle< ComputeSubset<T> >& s1,
        const constHandle< ComputeSubset<T> >& s2)
    {
      if (s1 == s2 || !s1 || !s2) {
        // for efficiency -- expedite when s1 and s2 point to the same thing or are null
        return Handle< ComputeSubset<T> >(scinew ComputeSubset<T>);
      }

      if (s1->size() == 0 || s1->size()==0 || s2->size()==0) 
      {
        return s1;  // return s1
      }

#if SCI_ASSERTION_LEVEL>0
      if (!s1->is_sorted()) {
        SCI_THROW(InternalError("ComputeSubset s1 not sorted in ComputeSubset<T>::intersectionAndMaybeDifference", __FILE__, __LINE__));
      }
      if (!s2->is_sorted()) {
        SCI_THROW(InternalError("ComputeSubset s2 not sorted in ComputeSubset<T>::intersectionAndMaybeDifference", __FILE__, __LINE__));
      }

      T el2 = s2->get(0);
      for(int i=1;i<s2->size();i++){
        T el = s2->get(i);
        if(!compareElems(el2, el)) {
          ostringstream msgstr;
          msgstr << "Set not sorted: " << el2 << ", " << el;
          SCI_THROW(InternalError(msgstr.str(), __FILE__, __LINE__)); 
        }
        el2=el;
      }
#endif
      
      Handle< ComputeSubset<T> > diff = scinew ComputeSubset<T>;

      int i1=0;
      int i2=0;
      for(;;){
        if(s1->get(i1) == s2->get(i2)){
          //in both s1 and s2
          i1++; i2++;
        } else if(compareElems(s1->get(i1), s2->get(i2))){
          //in s1 but not s2
          diff->add(s1->get(i1)); // alters setDifference1
          i1++;
        } else {
          //in s2 but not s1
          i2++;
        }
        if(i1 == s1->size() || i2 == s2->size())
          break;
      }

      // get the rest of s1 that wasn't finished (if any)
      for (; i1 != s1->size(); i1++) {
        diff->add(s1->get(i1)); // alters setDifference1
      }

      return diff;
    }


  template<class T>
    bool ComputeSubset<T>::overlaps(const ComputeSubset<T>* s1,
        const ComputeSubset<T>* s2)
    {
      if (s1 == s2) {
        return true;
      }
      if(s1->size() == 0 || s2->size() == 0) {
        return false;
      }
      if (!s1->is_sorted()) {
        SCI_THROW(InternalError("ComputeSubset s1 not sorted in ComputeSubset<T>::overlaps", __FILE__, __LINE__));
      }
      if (!s2->is_sorted()) {
        SCI_THROW(InternalError("ComputeSubset s2 not sorted in ComputeSubset<T>::overlaps", __FILE__, __LINE__));
      }
      int i1=0;
      int i2=0;
      for(;;){
        if(s1->get(i1) == s2->get(i2)){
          return true;
        } else if(compareElems(s1->get(i1), s2->get(i2))){
          if (++i1 == s1->size())
            break;
        } else {
          if (++i2 == s2->size())
            break;
        }
      }
      return false;
    }

  template<class T>
    bool ComputeSubset<T>::is_sorted() const
    {
      T cur = get(0);
      for(int i=1;i<size();i++){
        T next = get(i);
        if(!compareElems(cur, next)) {
          return false;
        }
        cur=next;
      }
      return true;
    }

} // end namespace Uintah

// The following #include doesn't seem to be necessary anymore... (It
// isn't needed on Ranger with pgCC version 7.1-2.)
//
//#if defined( __PGI ) && !defined( REDSTORM )
//#  include <Core/Grid/Variables/ComputeSet_special.cc>
//#endif


#endif
