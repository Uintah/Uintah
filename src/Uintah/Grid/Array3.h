#ifndef UINTAH_HOMEBREW_ARRAY3_H
#define UINTAH_HOMEBREW_ARRAY3_H

#include "Array3Window.h"
#include <SCICore/Geometry/IntVector.h>
#include <SCICore/Malloc/Allocator.h>

/**************************************

CLASS
   Array3
   
GENERAL INFORMATION

   Array3.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Array3

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

namespace Uintah {
   using SCICore::Geometry::IntVector;
   template<class T> class Array3 {
      
   public:
      Array3() {
	 d_window = 0;
      }
      Array3(int size1, int size2, int size3) {
	 d_window=scinew Array3Window<T>(new Array3Data<T>(size1, size2, size3));
	 d_window->addReference();
      }
      Array3(const IntVector& lowIndex, const IntVector& highIndex);
      virtual ~Array3();
      Array3(const Array3& copy)
	 : d_window(copy.d_window)
      {
	 if(d_window)
	    d_window->addReference();
      }
      
      Array3& operator=(const Array3& copy) {
	 if(copy.d_window)
	    copy.d_window->addReference();
	 if(d_window && d_window->removeReference()){
	    delete d_window;
	 }
	 d_window = copy.d_window;
	 return *this;
      }

      IntVector size() const {
	 return d_window->size();
      }
      void initialize(const T& value) {
	 d_window->initialize(value);
      }
      
      void initialize(const T& value, const IntVector& s,
		      const IntVector& e) {
	 d_window->initialize(value, s, e);
      }
      
      void resize(const IntVector& lowIndex, const IntVector& highIndex) {
	 if(d_window && d_window->removeReference())
	    delete d_window;
	 IntVector size = highIndex-lowIndex;
	 d_window=scinew Array3Window<T>(new Array3Data<T>(size), lowIndex, highIndex);
	 d_window->addReference();
      }
      T& operator[](const IntVector& idx) const {
	 return d_window->get(idx);
      }
      
      inline Array3Window<T>* getWindow() const {
	 return d_window;
      }
      T& operator[](const IntVector& idx) {
	 return d_window->get(idx);
      }
      
      IntVector getLowIndex() const {
	 return d_window->getLowIndex();
      }

      IntVector getHighIndex() const {
	return d_window->getHighIndex();
      }
      
   private:
      Array3Window<T>* d_window;
   };
   
   template<class T>
      Array3<T>::~Array3()
      {
	 if(d_window && d_window->removeReference()){
	    delete d_window;
	 }
      }
}
   
//
// $Log$
// Revision 1.11  2000/05/30 20:19:27  sparker
// Changed new to scinew to help track down memory leaks
// Changed region to patch
//
// Revision 1.10  2000/05/15 19:39:46  sparker
// Implemented initial version of DataArchive (output only so far)
// Other misc. cleanups
//
// Revision 1.9  2000/05/10 20:02:57  sparker
// Added support for ghost cells on node variables and particle variables
//  (work for 1 patch but not debugged for multiple)
// Do not schedule fracture tasks if fracture not enabled
// Added fracture directory to MPM sub.mk
// Be more uniform about using IntVector
// Made regions have a single uniform index space - still needs work
//
// Revision 1.8  2000/05/06 10:56:25  guilkey
// Filled in body of getLowIndex and getHighIndex.  Works for one
// region, should double check it for multiple regions.
//
// Revision 1.7  2000/05/02 06:07:21  sparker
// Implemented more of DataWarehouse and SerialMPM
//
// Revision 1.6  2000/04/26 06:48:46  sparker
// Streamlined namespaces
//
// Revision 1.5  2000/03/22 23:41:27  sparker
// Working towards getting arches to compile/run
//
// Revision 1.4  2000/03/22 00:32:12  sparker
// Added Face-centered variable class
// Added Per-region data class
// Added new task constructor for procedures with arguments
// Use IntVector more often
//
// Revision 1.3  2000/03/21 02:22:57  dav
// few more updates to make it compile including moving Array3 stuff out of namespace as I do not know where it should be
//
// Revision 1.2  2000/03/16 22:07:57  dav
// Added the beginnings of cocoon docs.  Added namespaces.  Did a few other coding standards updates too
//
//

#endif
