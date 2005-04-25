/*
 *  TensorField.cc: Data structure to represent tensor fields
 *  ---------------
 *
 *
 *  Eric Lundberg - 1998
 *  
 */

#include <Packages/DaveW/Core/Datatypes/General/TensorField.h>
#include <Core/Containers/String.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Math/MiscMath.h>
#include <Core/Util/NotFinished.h>

namespace DaveW {

static Persistent* make_TensorField() {
    return scinew TensorField<double>(0,0,0);
}

template<class DATA>
PersistentTypeID TensorField<DATA>::type_id("TensorField", "TensorFieldBase", make_TensorField);

/* The default constructor...currently not taking any args...*/
template<class DATA>
TensorField<DATA>::TensorField(int in_slices, int in_width, int in_height)
{
  m_slices = in_slices;
  m_width = in_width;
  m_height = in_height;
  int ii;
  
  /* Set up the arrays for the tensors*/
  m_tensor_field.resize(TENSOR_ELEMENTS);
  
  for (ii = 0; ii < TENSOR_ELEMENTS; ii ++)
    m_tensor_field[ii].resize(m_slices, m_width, m_height);
  
  /* Set up the arrays for the eigenvectors*/
  m_e_vectors.resize(EVECTOR_ELEMENTS);
    
  /* Set up the arrays for the eigenvalues*/
  m_e_values.resize(EVECTOR_ELEMENTS);
    
  m_tensorsGood = m_vectorsGood = m_valuesGood = 0;
}

template<class DATA>
TensorField<DATA>::TensorField(int in_width, int in_height, int in_slices,
			       float *data, int just_tensors)
{
  m_width = in_width;
  m_height = in_height;
  m_slices = in_slices;

  int ii;

  /* Set up the arrays for the tensors*/
  m_tensor_field.resize(TENSOR_ELEMENTS);
  
  for (ii = 0; ii < TENSOR_ELEMENTS; ii ++)
      m_tensor_field[ii].resize(m_width, m_height, m_slices);
  
  if (!just_tensors) {
      /* Set up the arrays for the eigenvectors*/
      m_e_vectors.resize(EVECTOR_ELEMENTS);
    
      /* Set up the arrays for the eigenvalues*/
      m_e_values.resize(EVECTOR_ELEMENTS);
      for (ii = 0; ii < EVECTOR_ELEMENTS; ii ++) {
	  m_e_vectors[ii].resize(m_width, m_height, m_slices);
	  m_e_values[ii].resize(m_width, m_height, m_slices);
      }
  }

  m_inside.resize(m_width, m_height, m_slices);

  float *p=&(data[0]);
  for (int z=0; z<in_slices; z++) {
      for (int y=0; y<in_height; y++) {
	  for (int x=0; x<in_width; x++) {
	      if (just_tensors) m_inside(x,y,z)=*p++;

	      for (int w=0; w<TENSOR_ELEMENTS; w++)
		  (m_tensor_field[w])(x,y,z)=*p++;
	      if (!just_tensors) 
		  for (int v=0; v<3; v++) {
		      Vector vec;
		      vec.x(*p++);
		      vec.y(*p++);
		      vec.z(*p++);
		      double vl=vec.length();
		      if (vl>0.000000001) vec.normalize();
		      m_e_vectors[v].grid(x,y,z)=vec;
		      m_e_values[v].grid(x,y,z)=vl;
		  }
	      if (!just_tensors) m_inside(x,y,z)=*p++;
	  }
      }
  }
  m_tensorsGood = 1;
  m_vectorsGood = m_valuesGood = !(just_tensors);
}

template<class DATA>
int TensorField<DATA>::interpolate(const Point& p, double t[][3], int &, int) 
{
    return interpolate(p, t);
}

// just do linear interpolation of the matrix entries
template<class DATA>
int TensorField<DATA>::interpolate(const Point& p, double t[][3])
{
    int nx=m_width;
    int ny=m_height;
    int nz=m_slices;
    Vector pn=p-bmin;
    double dx=diagonal.x();
    double dy=diagonal.y();
    double dz=diagonal.z();
    double x=pn.x()*(nx-1)/dx;
    double y=pn.y()*(ny-1)/dy;
    double z=pn.z()*(nz-1)/dz;
    int ix=(int)x;
    int iy=(int)y;
    int iz=(int)z;
    int ix1=ix+1;
    int iy1=iy+1;
    int iz1=iz+1;
    if(ix<0 || ix1>=nx)return 0;
    if(iy<0 || iy1>=ny)return 0;
    if(iz<0 || iz1>=nz)return 0;
    double fx=x-ix;
    double fy=y-iy;
    double fz=z-iz;
    double v000, v001, v010, v011, v100, v101, v110, v111,
	   x00, x01, x10, x11, y0, y1, zz[TENSOR_ELEMENTS];
    for (int i=0; i<TENSOR_ELEMENTS; i++) {
	v000=m_tensor_field[i](ix, iy, iz);
	v001=m_tensor_field[i](ix, iy, iz1);
	v010=m_tensor_field[i](ix, iy1, iz);
	v011=m_tensor_field[i](ix, iy1, iz1);
	v100=m_tensor_field[i](ix1, iy, iz);
	v101=m_tensor_field[i](ix1, iy, iz1);
	v110=m_tensor_field[i](ix1, iy1, iz);
	v111=m_tensor_field[i](ix1, iy1, iz1);
	x00=Interpolate(v000, v100, fx);
	x01=Interpolate(v001, v101, fx);
	x10=Interpolate(v010, v110, fx);
	x11=Interpolate(v011, v111, fx);
	y0=Interpolate(x00, x10, fy);
	y1=Interpolate(x01, x11, fy);
	zz[i]=Interpolate(y0, y1, fz);
    }
    t[0][0]=zz[0]; t[1][0]=t[0][1]=zz[1]; t[2][0]=t[0][2]=zz[2];
    t[1][1]=zz[3]; t[2][1]=t[1][2]=zz[4]; t[2][2]=zz[5];
    return 1;
}

/*note all index values should be from 0 - (value)*/
template<class DATA>
int TensorField<DATA>::AddSlice(int in_slice, int in_tensor_component, FILE* in_file)
{
  if (in_slice > m_slices)
    return 0;

  fread(&(m_tensor_field[in_tensor_component](in_slice,0,0)), sizeof(DATA), m_height*m_width, in_file);
  
  /*for (int yy = 0; yy < m_height; yy++)
    for (int xx = 0; xx < m_width; xx++)
    if (fread(&(m_tensor_field[in_tensor_component](in_slice,yy,xx)), sizeof(DATA), 1, in_file) != 1) 
	{	
	printf("TensorComponent file wasn't big enough");
	return 0;
	}*/
  return 1;
}

/* Constructor used to clone objects...either through a previous
   TensorFields .clone call, or explicitly with scinew TensorField(TFtoCopy)
   as such this function should actually copy all the data in t.x to this.x
   in a deep fashion */
template<class DATA>
TensorField<DATA>::TensorField(const TensorField<DATA>&)
{
  /*NOTE - IMPEMENT ME!*/
    NOT_FINISHED("TensorField copy ctor\n");
}

/*Destructor*/
template<class DATA>
TensorField<DATA>::~TensorField()
{ 
}

/*returns a totally different, but logically equivalent version of 'this'*/
template<class DATA>
#ifdef __GNUG__
TensorField<DATA>* TensorField<DATA>::clone() const
#else
TensorFieldBase* TensorField<DATA>::clone() const
#endif
{
    return scinew TensorField<DATA>(*this);
}

/* Set what version of the tensorfield we are currently working with - so we
   can support old verions of this datatype on the IO ports */
#define TENSORFIELD_VERSION 1

/* And do the actual IO stuff....*/
template<class DATA>
void TensorField<DATA>::io(Piostream& stream)
{
    stream.begin_class("TensorField", TENSORFIELD_VERSION);
    TensorFieldBase::io(stream);
    if (m_tensorsGood) {
	Pio(stream, m_tensor_field);
    }
    stream.end_class();
}
} // End namespace DaveW


