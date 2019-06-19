//go:generate go get -v github.com/mailru/easyjson/...
//go:generate go get github.com/UnnoTed/fileb0x
//go:generate fileb0x b0x.yml
//go:generate easyjson -snake_case -disallow_unknown_fields -pkg .

package cupti
