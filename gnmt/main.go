package main

// #cgo LDFLAGS: -ltensorflow
// #cgo CFLAGS: -I${SRCDIR}/../../../tensorflow/tensorflow/
// #include <stdlib.h>
// #include "tensorflow/c/c_api.h"
import "C"

import (
	// "path/filepath"

	"unsafe"
	"github.com/k0kubun/pp"
	// utils "github.com/rai-project/tensorflow-go-examples"
	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

var (
	dir = "/home/abduld/mlperf/inference/v0.5/translation/gnmt/tensorflow/savedmodel"
)

// translate.ckpt.data-00000-of-00001  translate.ckpt.index  translate.ckpt.meta

func main() {


	pth := C.CString("_beam_search_ops.so")
	C.free(unsafe.Pointer(pth))
	stat := C.TF_NewStatus()
	C.TF_LoadLibrary(pth, stat);
	pp.Println( C.GoString(C.TF_Message(stat)))

	model, err := tf.LoadSavedModel(dir, []string{"train", "serve"}, nil)
	if err != nil {
		pp.Printf("Error loading saved model: %s\n", err.Error())
		return
	}

	defer model.Session.Close()

	// // Input op
	// inputOp := graph.Operation("input")

	// // Output ops
	// o1 := graph.Operation("MobilenetV1/Predictions/Reshape_1")

	// // Execute COCO Graph
	// output, err := model.session.Run(
	// 	map[tf.Output]*tf.Tensor{
	// 		inputOp.Output(0): tensor,
	// 	},
	// 	[]tf.Output{
	// 		o1.Output(0),
	// 	},
	// 	nil)
	// if err != nil {
	// 	log.Fatal(err)
	// }
	// // Take the first in the batched output
	// probabilities := output[0].Value().([][]float32)[0]

	// idxs := make([]int, len(probabilities))
	// for i := range probabilities {
	// 	idxs[i] = i
	// }
	// preds := utils.Predictions{Probabilities: probabilities, Indexes: idxs}
	// sort.Sort(preds)

	// for ii := 0; ii < 1; ii++ {
	// 	pp.Println(preds.Indexes[ii], labels[preds.Indexes[ii]], preds.Probabilities[ii])
	// }
}
