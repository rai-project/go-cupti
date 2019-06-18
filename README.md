# go-cupti

[![Build Status](https://travis-ci.org/rai-project/go-cupti.svg?branch=master)](https://travis-ci.org/rai-project/go-cupti)

Go binding to NVIDIA CUPTI, the CUDA Performance Tool Interface.

## Example

The callback functions are publised to a tracing server. You need to have a carml_config.yml and the tracing server running to see the spans. Refer to [CarML Config](https://github.com/rai-project/carml/blob/master/docs/installation.md#carml-configuration) and [Starting Tracer Server](https://github.com/rai-project/carml/blob/master/docs/installation.md#starting-tracer-server)

Make sure `/usr/local/cuda/lib64` and `/usr/local/cuda/extras/CUPTI/lib64` are in your LD_LIBRARY_PATH.

```
cd examples/vector_add
make
cd ../cupti
go run main.go
```

Then go to `TRACER_URL:16686` to see the spans

**__Note__**: The CGO interface passes go pointers to the C API. This is an error by the CGO runtime.
If you get the following error

```
panic: runtime error: cgo argument has Go pointer to Go pointer
```

Then you need to place

```
export GODEBUG=cgocheck=0
```

in your `~/.bashrc` or `~/.zshrc` file and then run either `source ~/.bashrc` or `source ~/.zshrc`.

## References

The [CUDA Profiling Tools Interface (CUPTI)](https://docs.nvidia.com/cupti/Cupti/index.html) enables the creation of profiling and tracing tools that target CUDA applications. CUPTI provides four APIs: the Activity API, the Callback API, the Event API, and the Metric API.
1. The CUPTI Activity API allows you to asynchronously collect a trace of an application's CPU and GPU CUDA activity.
  - Activity Record. CPU and GPU activity is reported in C data structures called activity records.
  - Activity Buffer. An activity buffer is used to transfer one or more activity records from CUPTI to the client. An asynchronous buffering API is implemented by cuptiActivityRegisterCallbacks and cuptiActivityFlushAll.
2. The CUPTI Callback API allows you to register a callback into your own code. Your callback will be invoked when the application being profiled calls a CUDA runtime or driver function, or when certain events occur in the CUDA driver.
  - Callback Domain.
  - Callback ID. Each callback is given a unique ID within the corresponding callback domain so that you can identify it within your callback function.
  - Callback Function
    Your callback function must be of type CUpti_CallbackFunc. This function type has two arguments that specify the callback domain and ID so that you know why the callback is occurring. The type also has a cbdata argument that is used to pass data specific to the callback.
  - Subscriber. A subscriber is used to associate each of your callback functions with one or more CUDA API functions. There can be at most one subscriber initialized with cuptiSubscribe() at any time. Before initializing a new subscriber, the existing subscriber must be finalized with cuptiUnsubscribe().
3. The CUPTI Event API allows you to query, configure, start, stop, and read the event counters on a CUDA-enabled device.
4. The CUPTI Metric API allows you to collect application metrics calculated from one or more event values.
