#include <Core/Persistent/PersistentSTL.h>
#include <Core/Geometry/Tensor.h>
#include <Core/Geometry/Vector.h>
#include <Core/Datatypes/GenericField.h>
#include <Core/Datatypes/ScanlineField.h>
#include <Core/Datatypes/PointCloudField.h>

using namespace SCIRun;

template class GenericField<ScanlineMesh, vector<Tensor> >;
template class GenericField<ScanlineMesh, vector<Vector> >;
template class GenericField<ScanlineMesh, vector<double> >;
template class GenericField<ScanlineMesh, vector<float> >;
template class GenericField<ScanlineMesh, vector<int> >;
template class GenericField<ScanlineMesh, vector<short> >;
template class GenericField<ScanlineMesh, vector<char> >;
template class GenericField<ScanlineMesh, vector<unsigned int> >;
template class GenericField<ScanlineMesh, vector<unsigned short> >;
template class GenericField<ScanlineMesh, vector<unsigned char> >;

template class GenericField<PointCloudMesh, vector<Tensor> >;
template class GenericField<PointCloudMesh, vector<Vector> >;
template class GenericField<PointCloudMesh, vector<double> >;
template class GenericField<PointCloudMesh, vector<float> >;
template class GenericField<PointCloudMesh, vector<int> >;
template class GenericField<PointCloudMesh, vector<short> >;
template class GenericField<PointCloudMesh, vector<char> >;
template class GenericField<PointCloudMesh, vector<unsigned int> >;
template class GenericField<PointCloudMesh, vector<unsigned short> >;
template class GenericField<PointCloudMesh, vector<unsigned char> >;

template class ScanlineField<Tensor>;
template class ScanlineField<Vector>;
template class ScanlineField<double>;
template class ScanlineField<float>;
template class ScanlineField<int>;
template class ScanlineField<short>;
template class ScanlineField<char>;
template class ScanlineField<unsigned int>;
template class ScanlineField<unsigned short>;
template class ScanlineField<unsigned char>;

const TypeDescription* get_type_description(ScanlineField<Tensor> *);
const TypeDescription* get_type_description(ScanlineField<Vector> *);
const TypeDescription* get_type_description(ScanlineField<double> *);
const TypeDescription* get_type_description(ScanlineField<float> *);
const TypeDescription* get_type_description(ScanlineField<int> *);
const TypeDescription* get_type_description(ScanlineField<short> *);
const TypeDescription* get_type_description(ScanlineField<char> *);
const TypeDescription* get_type_description(ScanlineField<unsigned int> *);
const TypeDescription* get_type_description(ScanlineField<unsigned short> *);
const TypeDescription* get_type_description(ScanlineField<unsigned char> *);

template class PointCloudField<Tensor>;
template class PointCloudField<Vector>;
template class PointCloudField<double>;
template class PointCloudField<float>;
template class PointCloudField<int>;
template class PointCloudField<short>;
template class PointCloudField<char>;
template class PointCloudField<unsigned int>;
template class PointCloudField<unsigned short>;
template class PointCloudField<unsigned char>;

const TypeDescription* get_type_description(PointCloudField<Tensor> *);
const TypeDescription* get_type_description(PointCloudField<Vector> *);
const TypeDescription* get_type_description(PointCloudField<double> *);
const TypeDescription* get_type_description(PointCloudField<float> *);
const TypeDescription* get_type_description(PointCloudField<int> *);
const TypeDescription* get_type_description(PointCloudField<short> *);
const TypeDescription* get_type_description(PointCloudField<char> *);
const TypeDescription* get_type_description(PointCloudField<unsigned int> *);
const TypeDescription* get_type_description(PointCloudField<unsigned short> *);
const TypeDescription* get_type_description(PointCloudField<unsigned char> *);
