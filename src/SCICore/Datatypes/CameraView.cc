



#include <SCICore/Datatypes/CameraView.h>
#include <SCICore/Geometry/Point.h>
#include <SCICore/Geometry/Vector.h>
#include <SCICore/Malloc/Allocator.h>

namespace SCICore {
  namespace Datatypes {

    using SCICore::Geometry::Point;
    using SCICore::Geometry::Vector;
    using SCICore::GeomSpace::Pio;

    static Persistent* make_CameraView()
    {
      return scinew CameraView;
    }
    
    PersistentTypeID CameraView::type_id("CameraView", "Datatype", make_CameraView);

    CameraView::CameraView()
      : theView(Point(1, 0, 0), Point(0, 0, 0), Vector(0, 1, 0), 20)
    {}
    
    CameraView::CameraView(const View& vv)
      : theView(vv) 
    {}
    
    void CameraView::set_view(const View& vv){
      theView=vv;
    }
    
    CameraView* CameraView::clone(){
      return scinew CameraView(*this);
    }
    
    #define CAMERAVIEW_VERSION 1

    void CameraView::io(Piostream& stream){    
      /*int version=*/stream.begin_class("CameraView", CAMERAVIEW_VERSION);
      Pio(stream, theView);
      stream.end_class();
    }
  } // end Datatypes
}   // end SCICore
