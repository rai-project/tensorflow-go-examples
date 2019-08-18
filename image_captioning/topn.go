package main

import (
	"container/heap"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

type caption struct {
	sentence []int
	state    tf.Tensor
	logprob  float32
	score    float32
}

type topN struct {
	captionHeap []caption
	n           int
}

func (t topN) Len() int           { return len(t.captionHeap) }
func (t topN) Less(i, j int) bool { return t.captionHeap[i].score < t.captionHeap[j].score }
func (t topN) Swap(i, j int)      { t.captionHeap[i], t.captionHeap[j] = t.captionHeap[j], t.captionHeap[i] }

func (t *topN) Push(x interface{}) {
	t.captionHeap = append(t.captionHeap, x.(caption))
}

func (t *topN) Pop() interface{} {
	old := t.captionHeap
	currLength := len(old)
	x := old[currLength-1]
	t.captionHeap = old[0 : currLength-1]
	return x
}

func (t *topN) PushTopN(x interface{}) {
	if t.Len() < t.n {
		heap.Push(t, x.(caption))
	} else {
		heap.Push(t, x.(caption))
		heap.Pop(t)
	}
}

func (t *topN) Extract() []caption {
	var result []caption
	for t.Len() > 0 {
		result = append(result, heap.Pop(t).(caption))
	}

	return result
}

func (t *topN) Reset() {
	t.captionHeap = []caption{}
}

// beamSearch performs a beam search
// func beamSearch(sess tf.Session, inputImage tf.Tensor) []caption {
//
// }

// func main() {
// 	h := &topN{n: 4, captionHeap: []caption{}}
// 	cap1 := caption{sentence: []int{1, 2, 3},
// 		state:   [][]float32{{1, 2, 3}},
// 		logprob: 3,
// 		score:   1.23}
// 	cap2 := caption{sentence: []int{1},
// 		state:   [][]float32{{4, 5, 6}},
// 		logprob: 1,
// 		score:   7.89}
// 	cap3 := caption{sentence: []int{4, 5, 6},
// 		state:   [][]float32{{7, 8, 9}},
// 		logprob: 2,
// 		score:   12}
// 	cap4 := caption{sentence: []int{1, 2, 3, 4, 5},
// 		state:   [][]float32{{10, 11, 12}},
// 		logprob: 7,
// 		score:   11}
// 	cap5 := caption{sentence: []int{7, 8, 9, 10, 11},
// 		state:   [][]float32{{13, 14, 15}},
// 		logprob: 4,
// 		score:   5.6}
// 	heap.Init(h)
// 	h.PushTopN(cap1)
// 	fmt.Println(h)
// 	h.PushTopN(cap2)
// 	fmt.Println(h)
// 	h.PushTopN(cap3)
// 	fmt.Println(h)
// 	h.PushTopN(cap4)
// 	fmt.Println(h)
// 	h.PushTopN(cap5)
// 	fmt.Println(h)

// 	// for h.Len() > 0 {
// 	// 	fmt.Println(heap.Pop(h).(caption).score)
// 	// }
// 	test := h.Extract()
// 	fmt.Println(test)
// }
