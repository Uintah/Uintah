
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
    Datatype for passing SCICore::Geom::View as datatype

PATTERNS

WARNING


POSSIBLE REVISIONS 
----------------------------------------------------------------------*/

#ifndef SCI_SCICore_Datatypes_CameraView_h
#define SCI_SCICore_Datatypes_CameraView_h 1

#include <SCICore/share/share.h>
#include <SCICore/Datatypes/Datatype.h>
#include <SCICore/Containers/LockingHandle.h>
#include <SCICore/Containers/String.h>

#include <SCICore/Geom/View.h>

namespace SCICore {
  namespace Datatypes {

    using SCICore::Containers::LockingHandle;
    using SCICore::GeomSpace::View;
    using SCICore::Containers::clString;
    using SCICore::PersistentSpace::Piostream;
    using SCICore::PersistentSpace::PersistentTypeID;

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

  } // end namespace Datatypes
}   // end namespace SCICore

#endif












