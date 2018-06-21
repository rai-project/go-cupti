# CUPTI bindings in Go [![Build Status](https://travis-ci.org/rai-project/go-cupti.svg?branch=master)](https://travis-ci.org/rai-project/go-cupti)


## Example

The callback functions are publised to a tracing server. You need to have a carml_config.yml and the tracing server running to see the spans. Refer to [CarML Config](https://github.com/rai-project/carml/blob/master/docs/installation.md#carml-configuration) and [Starting Tracer Server](https://github.com/rai-project/carml/blob/master/docs/installation.md#starting-tracer-server)

```
cd examples/vector_add
make
cd ../cupti
go run main.go
```
Then go to TRACER_URL:16686 to see the spans

### Issues

The CGO interface passes go pointers to the C API. This is an error by the CGO runtime.
If you get the following error


~~~
panic: runtime error: cgo argument has Go pointer to Go pointer
~~~

Then you need to place

~~~
export GODEBUG=cgocheck=0
~~~

in your `~/.bashrc` or `~/.zshrc` file and then run either `source ~/.bashrc` or `source ~/.zshrc`

