
/*----------------------------------------------------------------------
CLASS
    CameraView

    Datatype for wrapping View object

GENERAL INFORMATION

    Created by:
    Alexei Samsonov
    Department of Computer Science
    University of Utah
    August 2000

    Copyright (C) 2000 SCI Group

KEYWORDS
    Quaternion

DESCRIPTION
    Datatype for passing View as datatype

PATTERNS

WARNING


POSSIBLE REVISIONS 
----------------------------------------------------------------------*/

#ifndef SCI_Core_Datatypes_CameraView_h
#define SCI_Core_Datatypes_CameraView_h 1

#include <Core/share/share.h>
#include <Core/Datatypes/Datatype.h>
#include <Core/Containers/LockingHandle.h>
#include <Core/Containers/String.h>

#include <Core/Geom/View.h>

namespace SCIRun {


    class CameraView;
    typedef LockingHandle<CameraView> CameraViewHandle;
    
    class SCICORESHARE CameraView: public Datatype {
      View theView;
    public:
      CameraView();
      CameraView(const View&);      
     
      inline const View& get_view() const;
      void set_view(const View&);

      virtual CameraView* clone();
      //////////
      // Persistent representation...
      virtual void io(Piostream&);
      static PersistentTypeID type_id;
    };
    
    inline const View& CameraView::get_view() const {
      return theView;
    }

} // End namespace SCIRun

#endif












