class ErrorEstimator {
 public:
  virtual void estimateError() const = 0;
  virtual bool needRefining() const = 0;
  virtual void needCoarsening() const = 0;

};
