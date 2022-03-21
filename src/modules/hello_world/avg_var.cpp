AnyType avg_var_transition::run(AnyType& args);
AnyType avg_var_merge_states::run(AnyType& args);
AnyType avg_var_final::run(AnyType& args);



AnyType
avg_var_transition::run(AnyType& args) {
    // get current state value
    AvgVarTransitionState<MutableArrayHandle<double> > state = args[0];
    // update state with current row value
    double x = args[1].getAs<double>();
    state += x;
    state.numRows ++;
    return state;
}


AnyType
avg_var_merge_states::run(AnyType& args) {
    AvgVarTransitionState<MutableArrayHandle<double> > stateLeft = args[0];
    AvgVarTransitionState<ArrayHandle<double> > stateRight = args[1];
 
    // Merge states together and return
    stateLeft += stateRight;
    return stateLeft;
}




AnyType
avg_var_final::run(AnyType& args) {
    AvgVarTransitionState<MutableArrayHandle<double> > state = args[0];
 
    // If we haven't seen any data, just return Null. This is the standard
    // behavior of aggregate function on empty data sets (compare, e.g.,
    // how PostgreSQL handles sum or avg on empty inputs)
    if (state.numRows == 0)
        return Null();
 
    return state;
}







  /**
     * @brief Update state with a new data point
     */
    AvgVarTransitionState &operator+=(const double x){
        double diff = (x - avg);
        double normalizer = static_cast<double>(numRows + 1);
        // online update mean
        this->avg += diff / normalizer;
        // online update variance
        double new_diff = (x - avg);
        double a = static_cast<double>(numRows) / normalizer;
        this->var = (var * a) + (diff * new_diff) / normalizer;
    }
  
/**
 * @brief Merge with another State object
 *
 * We update mean and variance in a online fashion
 * to avoid intermediate large sum.
 */
template <class OtherHandle>
AvgVarTransitionState &operator+=(
    const AvgVarTransitionState<OtherHandle> &inOtherState) {
 
    if (mStorage.size() != inOtherState.mStorage.size())
        throw std::logic_error("Internal error: Incompatible transition "
                               "states");
    double avg_ = inOtherState.avg;
    double var_ = inOtherState.var;
    uint64_t numRows_ = static_cast<uint64_t>(inOtherState.numRows);
    double totalNumRows = static_cast<double>(numRows + numRows_);
    double p = static_cast<double>(numRows) / totalNumRows;
    double p_ = static_cast<double>(numRows_) / totalNumRows;
    double totalAvg = avg * p + avg_ * p_;
    double a = avg - totalAvg;
    double a_ = avg_ - totalAvg;
 
    numRows += numRows_;
    var = p * var + p_ * var_ + p * a * a + p_ * a_ * a_;
    avg = totalAvg;
    return *this;
}
