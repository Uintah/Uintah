#for babelType functions
require 'template/CCAtoBabel.util.rb'

#using the conversion functions
def BabelArrToVtkContourData( arg )
return  "::sidl::array<double> _table(" + arg.name + ");
  double zval;
  vtkImageCanvasSource2D* canvas = vtkImageCanvasSource2D::New();
  canvas->SetImageData(vtkID);
  canvas->SetExtent(0,100,0,100,0,100);
  canvas->SetDrawColor(70,70,70);
  for(int x=0; x <= _table.upper(0); x++) {
    for(int y=0; y <= _table.upper(1); y++) {
      for(int z=0; z <= 100; z++) {
        zval = _table.get(x,y)*100;
        if(fabs(zval-z) < 3) {
          canvas->SetDefaultZ(z);
          canvas->DrawPoint(x,y);
        }
      }
    }
  }"
end
                                                                                                                               
registerConvertFunc("array<double,2>","ContourData",method(:BabelArrToVtkContourData))

