# CUPTI bindings in Go

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

