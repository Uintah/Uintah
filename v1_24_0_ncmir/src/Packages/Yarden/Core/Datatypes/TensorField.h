
/*
 *  Tensor.h
 *
 *  Written by:
 *   Author: Packages/Yarden Livnat
 *   
 *   Department of Computer Science
 *   University of Utah
 *   Date: Oct 2000
 *
 *  Copyright (C) 2000 SCI Group
 */

#ifndef SCI_Datatypes_Tensor_h
#define SCI_Datatypes_Tensor_h 

#include <iostream>
#include <stdio.h>
#include <errno.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>

#include <Core/Datatypes/Datatype.h>
#include <Core/Containers/Array1.h>
#include <Core/Containers/Array3.h>
#include <Core/Containers/LockingHandle.h>

#include <Core/Containers/String.h>
#include <Core/Malloc/Allocator.h>
#include <Core/Util/NotFinished.h>


namespace Yarden {
using namespace SCIRun;
    
    // Tensors

    class TensorBase  {
    public:
      TensorBase() {}
      ~TensorBase() {}
    };

    
    template<class T, int N>
    class SymTensor : public TensorBase {
    public:
      T data[N*(N+1)/2];

    public:
      SymTensor() {}
      ~SymTensor() {}

      T& operator[] (int i) { return data[i]; }
      T& operator() (int i, int j) const { return data[(N+N-i-1)*i+j]; } 

      static int get_len() { return N*(N+1)/2; }
      void get_minmax( double &, double &);
      void check_minmax( int, double &, double &);
      friend void TEMPLATE_TAG Pio TEMPLATE_BOX (Piostream&, SymTensor<T,N>*&);
      friend void TEMPLATE_TAG Pio TEMPLATE_BOX (Piostream&, SymTensor<T,N>&);

    };

    template<class T, int N>
    void SymTensor<T,N>::get_minmax( double &min, double &max )
    {
      min = max = data[0];
      for (int i=1; i<N*(N+1)/2; i++)
	if (data[i] < min ) min = data[i];
	else if (data[i] > max ) max = data[i];
    }

    template<class T, int N>
    void SymTensor<T,N>::check_minmax( int i, double &min, double &max )
    {
      if ( data[i] < min ) min = data[i];
      if ( data[i] > max ) max = data[i];
    }

    template<class T, int N>
    void Pio(Piostream& stream, Datatypes::SymTensor<T,N>*& tensor)
    {
      for (int i=0; i<N*(N+1)/2; i++)
	Pio( stream, tensor->data[i] );
    }

    template<class T, int N>
    void Pio(Piostream& stream, Datatypes::SymTensor<T,N>& tensor)
    {
      for (int i=0; i<N*(N+1)/2; i++)
	Pio( stream, tensor.data[i] );
    }

    // Tensor Fields

    class TensorFieldBase;
    typedef LockingHandle<TensorFieldBase> TensorFieldHandle;

    class TensorFieldBase : public Datatype {
    public:
      double min,max;
      bool has_minmax;

      int separate_raw;
      clString raw_filename;

    public:
      TensorFieldBase() :has_minmax(false) {}
      virtual ~TensorFieldBase() {}

      void set_raw(int r) { separate_raw = r; }
      int  get_raw() {return separate_raw; }
      virtual void compute_minmax() = 0;
      virtual void get_minmax( double &min, double &max) = 0;
      
      static PersistentTypeID type_id;
    };

    template<class T>
    class TensorField : public TensorFieldBase {
    public:
      Array3< T > data;
      
    public:
      TensorField( int i, int j, int k) { newsize(i,j,k);}
      TensorField( const TensorField<T> & );
      virtual ~TensorField() {};

      T &operator() (int i, int j, int k) { return data(i,j,k); }

      void newsize(int i, int j, int k) { data.newsize(i,j,k); }
      int dim1() const { return data.dim1(); }
      int dim2() const { return data.dim2(); }
      int dim3() const { return data.dim3(); }

      virtual void compute_minmax();
      virtual void get_minmax( double &mn, double &mx);
      virtual void get_minmax( double *mn, double *mx);

#ifdef __GNUG__
      virtual TensorField<T>* clone() const; /*makes a real copy of this*/
#else
      virtual TensorFieldBase* clone() const; /*makes a real copy of this*/
#endif
      
      // Persistent representation...
      virtual void io(Piostream&);
      static PersistentTypeID type_id;
    };
    


    template<class T>
    TensorField<T>::TensorField(const TensorField<T>&)
    {
      NOT_FINISHED("TensorField copy ctor\n");
    }


    template<class T>
#ifdef __GNUG__
    TensorField<T>* TensorField<T>::clone() const
#else
    TensorFieldBase* TensorField<T>::clone() const
#endif
    {
      return scinew TensorField<T>(*this);
    }

    template<class T>
    void TensorField<T>::compute_minmax()
    {
      min = max = data(0,0,0)[0];
      
      for (int z=0; z<data.dim1(); z++)
	for (int y=0; y<data.dim2(); y++)
	  for (int x=0; x<data.dim3(); x++) {
	    T &tensor = data(z,y,x);
	    double mn, mx;
	    tensor.get_minmax( mn, mx );
	    if ( mn < min ) min = mn;
	    if ( mx > max ) max = mx;
	  }

      has_minmax = true;
    }
	    
	
    template<class T>
    void TensorField<T>::get_minmax( double &mn, double &mx)
    {
      if ( !has_minmax ) 
	compute_minmax();

      mn = min;
      mx = max;
    }
	
    template<class T>
    void TensorField<T>::get_minmax( double *mn, double *mx)
    {
      int len = T::get_len();
      
      for (int i=0; i<len; i++)
	mn[i] = mx[i] = data(0,0,0)[i];

      for (int z=0; z<data.dim1(); z++)
	for (int y=0; y<data.dim2(); y++)
	  for (int x=0; x<data.dim3(); x++) {
	    T &tensor = data(z,y,x);
	    for (int i=0; i<len; i++)
	      tensor.check_minmax(i, mn[i], mx[i] );
	  }
    }
	
	

#define TensorField_VERSION 1

    template<class T>
    void TensorField<T>::io(Piostream& stream)
    {
      int split;
      clString filename;
      int file = -1;

      /*int version = */stream.begin_class( type_id.type, TensorField_VERSION);
      
      if ( stream.reading() ) {
	Pio(stream, separate_raw);
	
	split = separate_raw;
	
	if ( separate_raw == 1) {
	  Pio(stream,raw_filename);
	  if ( raw_filename(0) == '/' )
	    filename = raw_filename;
	  else
	    filename = pathname( stream.file_name ) + "/" + raw_filename;
	  std::cerr << "reading... rawfile=" << filename <<std::endl;
	  file=open( filename(), O_RDONLY, 0666);
	  if ( file == -1 ) {
	    printf("TensorField: can not open file [%s]\n", filename());
	    return;
	  }
	}
      }
      else { // writing
	filename = raw_filename;
	split = separate_raw ;
	if ( separate_raw == 1) {
	  std::cerr << "TensorField write: split [" << filename << "]\n";
/* 	  if ( filename == "" ) { */
	    if ( stream.file_name() ) { 
	      char *tmp=strdup(stream.file_name());
	      char *dot = strrchr( tmp, '.' );
	      if (!dot ) dot = strrchr( tmp, 0);
	      
	      filename = stream.file_name.substr(0,dot-tmp)+clString(".raw");
	      cerr << "split file = " << filename << endl;
	      delete tmp;
	    }
	    else 
	      split = 0;
//* 	  } */
	}
	if ( split ) {
	  Pio(stream, split);
	  Pio(stream, filename);
	  file=open( filename(), O_WRONLY|O_CREAT|O_TRUNC, 0666);
	  if ( file == -1 ) {
	    printf("TensorField: can not open file [%s]\n", filename());
	    return;
	  }
	}
	else 
	  Pio(stream, split);
      }
      
      
      if ( split )
	Pio( stream, data, file );
      else 
	Pio( stream, data );
      
      stream.end_class();
    }
} // End namespace Yarden
    



#endif /* SCI_Datatypes_TensorField_h */
