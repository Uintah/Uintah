#ifndef __ERROR_ESTIMATOR_H__
#define __ERROR_ESTIMATOR_H__

class ErrorEstimator {
 public:
  virtual void estimateError() const = 0;
  virtual bool needRefining() const = 0;
  virtual void needCoarsening() const = 0;

};

#endif __ERROR_ESTIMATOR_H__

// $Log$
// Revision 1.2  2000/03/15 21:58:21  jas
// Added logging and put guards in.
//
