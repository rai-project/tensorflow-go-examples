package main

import (
	"container/heap"
	"fmt"
	_ "image/jpeg"
	"io/ioutil"
	"log"
	"math"
	"os"
	"path/filepath"
	"sort"
	"strings"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

func batchify(tensors []*tf.Tensor) (packedTensor *tf.Tensor) {
	s := op.NewScope()
	var pack []tf.Output
	for i := 0; i < len(tensors); i++ {
		input := op.Placeholder(s.SubScope("input"), tf.Float)
		pack = append(pack, input)
	}
	output := op.Pack(s, pack)

	graph, err := s.Finalize()
	if err != nil {
		panic(err.Error())
	}

	session, err := tf.NewSession(graph, nil)
	if err != nil {
		panic(err.Error())
	}
	defer session.Close()

	inputFeed := make(map[tf.Output]*tf.Tensor)
	for i := 0; i < len(tensors); i++ {
		inputFeed[pack[i]] = tensors[i]
	}
	batchOutput, err := session.Run(
		inputFeed,
		[]tf.Output{output},
		nil)
	if err != nil {
		fmt.Println("input image conversion to tensor error", err)
	}

	return batchOutput[0]
}

type argSort struct {
	Args []float32
	Idxs []int64
}

// Implement sort.Interface Len
func (s argSort) Len() int { return len(s.Args) }

// Implement sort.Interface Less
func (s argSort) Less(i, j int) bool { return s.Args[i] < s.Args[j] }

// Implment sort.Interface Swap
func (s argSort) Swap(i, j int) {
	// swap value
	s.Args[i], s.Args[j] = s.Args[j], s.Args[i]
	// swap index
	s.Idxs[i], s.Idxs[j] = s.Idxs[j], s.Idxs[i]
}

func main() {
	os.Setenv("TF_CPP_MIN_LOG_LEVEL", "2")
	// Parse flags
	modeldir := "./"
	jpgfile := "./COCO_val2014_000000224477.jpg"
	// outjpg := "output.jpg"

	modelpath := filepath.Join(modeldir, "im2txt_frozen_graph.pb")
	model, err := ioutil.ReadFile(modelpath)
	if err != nil {
		fmt.Println(err)
		log.Fatal(err)
	}

	// Construct the vocabulary
	// vocab, reverseVocab := constructVocabulary("word_counts_p.txt")

	// Construct an in-memory graph from the serialized form.
	graph := tf.NewGraph()
	if err := graph.Import(model, ""); err != nil {
		fmt.Println(err)
		log.Fatal(err)
	}

	// Create a session for inference over graph.
	session, err := tf.NewSession(graph, nil)
	if err != nil {
		fmt.Println("create new session error", err)
		log.Fatal(err)
	}
	defer session.Close()

	// Harness the built-in decoding.
	b, err := ioutil.ReadFile(jpgfile)
	tensor, err := tf.NewTensor(string(b))
	if err != nil {
		fmt.Println("bytes converting error:")
	}

	// Get all the input and output operations
	inputOp := graph.Operation("image_feed")
	// inputOp := graph.Operation("control_dependency")
	// inputOp := graph.Operation("decode/DecodeJpeg")
	outputOp := graph.Operation("lstm/initial_state")

	// Execute COCO Graph
	initialState, err := session.Run(
		map[tf.Output]*tf.Tensor{
			inputOp.Output(0): tensor,
		},
		[]tf.Output{
			outputOp.Output(0),
		},
		nil)
	if err != nil {
		fmt.Println("session run error", err)
		log.Fatal(err)
	}

	// Beam Search
	beamSize := 3
	maxCaptionLength := 20
	lengthNormalizationFactor := 0.0

	vocabFilename := "./word_counts_p.txt"
	vocab := constructVocabulary(vocabFilename)

	// no actual meaning. just to squeeze the 0th dimension.
	initialStateArray, _ := tf.NewTensor(initialState[0].Value().([][]float32)[0])

	initialBeam := caption{
		sentence: []int64{vocab.startID},
		state:    initialStateArray,
		logprob:  0.0,
		score:    0.0,
	}

	//initialBeam comes from the CNN feature.
	partialCaptions := &topN{n: beamSize}
	heap.Init(partialCaptions)
	partialCaptions.PushTopN(initialBeam)

	completeCaptions := &topN{n: beamSize}
	heap.Init(completeCaptions)

	intermediateInputFeed := graph.Operation("input_feed")
	intermediateStateFeed := graph.Operation("lstm/state_feed")
	softmaxOp := graph.Operation("softmax")
	stateOp := graph.Operation("lstm/state")

	for i := 0; i < maxCaptionLength-1; i++ {
		partialCaptionsList := partialCaptions.Extract(false)
		// partialCaptions.Reset()

		inputFeed := []int64{}
		stateFeed := []*tf.Tensor{}
		for _, partialCaption := range partialCaptionsList {
			inputFeed = append(inputFeed, partialCaption.sentence[len(partialCaption.sentence)-1])
			stateFeed = append(stateFeed, partialCaption.state)
		}
		inputTensor, err := tf.NewTensor(inputFeed)
		if err != nil {
			fmt.Println("inputTensor error:", err)
		}
		stateTensor := batchify(stateFeed)

		intermediateOutput, err := session.Run(
			map[tf.Output]*tf.Tensor{
				intermediateInputFeed.Output(0): inputTensor,
				intermediateStateFeed.Output(0): stateTensor,
			},
			[]tf.Output{
				softmaxOp.Output(0),
				stateOp.Output(0),
			},
			nil)
		if err != nil {
			fmt.Println("intermediate session run error:", err)
			log.Fatal(err)
		}

		softmaxOutput := intermediateOutput[0].Value().([][]float32)
		stateOutput := intermediateOutput[1].Value().([][]float32)

		for j, partialCaption := range partialCaptionsList {
			// for partialCaption, get several most probable next words (default is beamSize = 3)
			// argsort is from go-mxnet suggested by Cheng.
			// The sorting is ascending so in the following loop they are accessed reversedly.
			wordProbabilities := softmaxOutput[j]
			state := stateOutput[j]

			wordLen := len(wordProbabilities)
			idxs := []int64{}
			for idx := int64(0); idx < int64(wordLen); idx++ {
				idxs = append(idxs, idx)
			}
			arg := argSort{Args: wordProbabilities, Idxs: idxs}
			sort.Sort(arg)
			mostLikelyWords := arg.Idxs[wordLen-beamSize : wordLen]
			mostLikelyWordsProb := arg.Args[wordLen-beamSize : wordLen]

			for k := len(mostLikelyWords) - 1; k >= 0; k-- {
				w := mostLikelyWords[k]
				p := mostLikelyWordsProb[k]
				if p < 1e-12 {
					continue // avoid log(0)
				}

				sentence := make([]int64, len(partialCaption.sentence))
				copy(sentence, partialCaption.sentence)
				sentence = append(sentence, w)
				logprob := partialCaption.logprob + float32(math.Log(float64(p)))
				score := logprob
				newStateTensor, _ := tf.NewTensor(state)

				if w == vocab.endID {
					if lengthNormalizationFactor > 0 {
						score /= float32(math.Pow(float64(len(sentence)), lengthNormalizationFactor))
					}
					beam := caption{
						sentence: sentence,
						state:    newStateTensor,
						logprob:  logprob,
						score:    score,
					}
					completeCaptions.PushTopN(beam)
				} else {
					beam := caption{
						sentence: sentence,
						state:    newStateTensor,
						logprob:  logprob,
						score:    score,
					}
					partialCaptions.PushTopN(beam)
				}

				if partialCaptions.Len() == 0 {
					break
				}
			}
		}
	}

	// If we have no complete captions then fall back to the partial captions.
	// But never output a mixture of complete and partial captions because a
	// partial caption could have a higher score than all the complete captions.
	if completeCaptions.Len() == 0 {
		completeCaptions = partialCaptions
	}

	captions := completeCaptions.Extract(true)
	wordArray := []string{}
	IDArray := []int64{}
	for i := len(captions) - 1; i >= 0; i-- {
		//print the final result
		caption := captions[i]
		for _, wordID := range caption.sentence {
			wordArray = append(wordArray, vocab.reverseVocab[wordID])
			IDArray = append(IDArray, wordID)
		}
		predSentence := strings.Join(wordArray[1:len(wordArray)-1], " ")
		fmt.Printf("%d) %s (p=%f)   ", i, predSentence, math.Exp(float64(caption.logprob)))
		fmt.Println(IDArray)
		wordArray = nil
		IDArray = nil
	}
}
