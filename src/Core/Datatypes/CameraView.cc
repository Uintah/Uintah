



#include <Core/Datatypes/CameraView.h>
#include <Core/Geometry/Point.h>
#include <Core/Geometry/Vector.h>
#include <Core/Malloc/Allocator.h>

namespace SCIRun {


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
} // End namespace SCIRun
