#ifndef __ERROR_ESTIMATOR_H__
#define __ERROR_ESTIMATOR_H__

class ErrorEstimator {
 public:
  virtual void estimateError() const = 0;
  virtual bool needRefining() const = 0;
  virtual void needCoarsening() const = 0;

};

#endif __ERROR_ESTIMATOR_H__

