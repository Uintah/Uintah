#ifndef UINTAH_HOMEBREW_Array3Window_H
#define UINTAH_HOMEBREW_Array3Window_H

#include <SCICore/Geometry/IntVector.h>
#include "RefCounted.h"
#include "Array3Data.h"

/**************************************

CLASS
   Array3Window
   
GENERAL INFORMATION

   Array3Window.h

   Steven G. Parker
   Department of Computer Science
   University of Utah

   Center for the Simulation of Accidental Fires and Explosions (C-SAFE)
  
   Copyright (C) 2000 SCI Group

KEYWORDS
   Array3Window

DESCRIPTION
   Long description...
  
WARNING
  
****************************************/

namespace Uintah {
   using SCICore::Geometry::IntVector;
   template<class T> class Array3Window : public RefCounted {
   public:
      Array3Window(Array3Data<T>*);
      Array3Window(Array3Data<T>*, const IntVector& lowIndex, const IntVector& highIndex);
      virtual ~Array3Window();
      
      inline Array3Data<T>* getData() const {
	 return data;
      }
      
      void initialize(const T&);
      void initialize(const T&, const IntVector& s, const IntVector& e);
      inline IntVector getLowIndex() const {
	 return lowIndex;
      }
      inline IntVector getHighIndex() const {
	 return highIndex;
      }
      inline T& get(const IntVector& idx) {
	 //ASSERT(data);
	 CHECKARRAYBOUNDS(idx.x(), lowIndex.x(), highIndex.x());
	 CHECKARRAYBOUNDS(idx.y(), lowIndex.y(), highIndex.y());
	 CHECKARRAYBOUNDS(idx.z(), lowIndex.z(), highIndex.z());
	 return data->get(idx-lowIndex);
      }
      
   private:
      
      Array3Data<T>* data;
      IntVector lowIndex;
      IntVector highIndex;
      Array3Window(const Array3Window<T>&);
      Array3Window<T>& operator=(const Array3Window<T>&);
   };
   
   template<class T>
      void Array3Window<T>::initialize(const T& val)
      {
	 data->initialize(val, IntVector(0,0,0), highIndex-lowIndex);
      }
   
   template<class T>
      void Array3Window<T>::initialize(const T& val,
				       const IntVector& s,
				       const IntVector& e)
      {
	 CHECKARRAYBOUNDS(s.x(), lowIndex.x(), highIndex.x());
	 CHECKARRAYBOUNDS(s.y(), lowIndex.y(), highIndex.y());
	 CHECKARRAYBOUNDS(s.z(), lowIndex.z(), highIndex.z());
	 CHECKARRAYBOUNDS(e.x(), s.x(), highIndex.x()+1);
	 CHECKARRAYBOUNDS(e.y(), s.y(), highIndex.y()+1);
	 CHECKARRAYBOUNDS(e.z(), s.z(), highIndex.z()+1);
	 data->initialize(val, s-lowIndex, e-lowIndex);
      }
   
   template<class T>
      Array3Window<T>::Array3Window(Array3Data<T>* data)
      : data(data), lowIndex(0,0,0), highIndex(data->size())
      {
	 data->addReference();
      }
   
   template<class T>
      Array3Window<T>::Array3Window(Array3Data<T>* data,
				    const IntVector& lowIndex,
				    const IntVector& highIndex)
      : data(data), lowIndex(lowIndex), highIndex(highIndex)
      {
	 CHECKARRAYBOUNDS(lowIndex.x(), 0, data->size().x());
	 CHECKARRAYBOUNDS(lowIndex.y(), 0, data->size().y());
	 CHECKARRAYBOUNDS(lowIndex.z(), 0, data->size().z());
	 CHECKARRAYBOUNDS(highIndex.x()-lowIndex.x(), 0, data->size().x()+1);
	 CHECKARRAYBOUNDS(highIndex.y()-lowIndex.y(), 0, data->size().y()+1);
	 CHECKARRAYBOUNDS(highIndex.z()-lowIndex.z(), 0, data->size().z()+1);
	 data->addReference();
      }
   
   template<class T>
      Array3Window<T>::~Array3Window()
      {
	 if(data && data->removeReference())
	    delete data;
      }
   
}

//
// $Log$
// Revision 1.5  2000/05/10 20:02:58  sparker
// Added support for ghost cells on node variables and particle variables
//  (work for 1 patch but not debugged for multiple)
// Do not schedule fracture tasks if fracture not enabled
// Added fracture directory to MPM sub.mk
// Be more uniform about using IntVector
// Made regions have a single uniform index space - still needs work
//
// Revision 1.4  2000/04/26 06:48:46  sparker
// Streamlined namespaces
//
// Revision 1.3  2000/03/21 02:22:57  dav
// few more updates to make it compile including moving Array3 stuff out of namespace as I do not know where it should be
//
// Revision 1.2  2000/03/16 22:07:58  dav
// Added the beginnings of cocoon docs.  Added namespaces.  Did a few other coding standards updates too
//
//

#endif
